import tensorflow as tf
from tensorflow import keras
import tf_keras as k3
from tf_keras import backend as K
import numpy as np
from TrackNet import TrackNet
from ResNet_Track import ResNet_Track
import os
import json
from PIL import Image
import logging
import gc
import shutil
import tempfile
from keras.saving import load_model
import datetime

def ensure_clean_dir(directory):
    """确保目录存在且为空"""
    directory = os.path.abspath(directory)  # Get absolute path
    if os.path.exists(directory):
        if os.path.isfile(directory):
            os.remove(directory)
        else:
            # Try removing content first
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to clean directory contents: {e}")
                # If cleaning fails, remove the entire directory
                shutil.rmtree(directory)
                
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory

# 使用系统临时目录作为基础目录
BASE_OUTPUT_DIR = os.path.join('/tmp', 'tracknet_distillation')
print(f"Using temporary directory for outputs: {BASE_OUTPUT_DIR}")

# Create and clean necessary directories
LOG_DIR = os.path.join(BASE_OUTPUT_DIR, 'logs')
TENSORBOARD_DIR = os.path.join(LOG_DIR, 'tensorboard')
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, 'checkpoints')

# 确保基础目录存在
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 确保目录存在且为空
LOG_DIR = ensure_clean_dir(LOG_DIR)
TENSORBOARD_DIR = ensure_clean_dir(TENSORBOARD_DIR)
CHECKPOINT_DIR = ensure_clean_dir(CHECKPOINT_DIR)

# 设置日志
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'distillation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# 定义蒸馏温度
TEMPERATURE = 4.0
BATCH_SIZE = 8
HEIGHT = 288
WIDTH = 512

# 验证目录创建是否成功
for dir_path in [LOG_DIR, TENSORBOARD_DIR, CHECKPOINT_DIR]:
    if not os.path.isdir(dir_path):
        raise RuntimeError(f"Failed to create directory: {dir_path}")
    logger.info(f"Successfully created directory: {dir_path}")

class DistillationModel(keras.Model):
    def __init__(self, student_model, teacher_model, temperature=TEMPERATURE):
        super().__init__()
        self.student_model = student_model  # Channel Last (NHWC)
        self.teacher_model = teacher_model  # Channel First (NCHW)
        self.temperature = temperature
        
        # 打印模型输入输出维度信息
        print("\n模型维度信息：")
        print("学生模型：")
        print(f"- 输入维度: (batch_size, {HEIGHT}, {WIDTH}, 3)")  # 单帧RGB图像
        print(f"- 输出维度: (batch_size, {HEIGHT}, {WIDTH}, 3)")  # 热力图输出
        
        print("\n教师模型：")
        print(f"- 输入维度: (batch_size, 9, {HEIGHT}, {WIDTH})")  # 三帧RGB图像，Channel First
        print(f"- 输出维度: (batch_size, 3, {HEIGHT}, {WIDTH})")  # 热力图输出，Channel First
        print("\n")
        
    def compile(self, optimizer, metrics=None):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_tracker = keras.metrics.Mean(name="distillation_loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")

    @property
    def metrics(self):
        return [self.distillation_loss_tracker, self.student_loss_tracker] + super().metrics

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # 获取学生模型的预测（使用当前帧，NHWC格式）
            current_frame = x[..., 3:6]  # 提取中间帧，保持NHWC格式
            student_predictions = self.student_model(current_frame, training=True)  # 输出为NHWC
            
            # 获取教师模型的预测（需要所有三帧的NCHW格式）
            if isinstance(self.teacher_model, keras.layers.TFSMLayer):
                # 对于TFSMLayer，我们需要使用serving_default签名
                # 转换为NCHW格式
                x_teacher = tf.transpose(x, [0, 3, 1, 2])  # 从(N,H,W,C)转换为(N,C,H,W)
                teacher_predictions = self.teacher_model(x_teacher)['output_0']
                # # 如果需要，将教师模型的输出从NCHW转换回NHWC
                # if teacher_predictions.shape.ndims == 4:
                #     teacher_predictions = tf.transpose(teacher_predictions, [0, 2, 3, 1])
                # 将教师模型的输出从NCHW转换回NHWC
                teacher_predictions = tf.transpose(teacher_predictions, [0, 2, 3, 1])
            else:
                # 对于普通的Keras模型
                x_teacher = tf.transpose(x, [0, 3, 1, 2])  # 从NHWC转换为NCHW
                teacher_predictions = self.teacher_model(x_teacher, training=False)  # 输出为NCHW
                # 将教师模型的输出从NCHW转换为NHWC以匹配学生模型
                teacher_predictions = tf.transpose(teacher_predictions, [0, 2, 3, 1])
            
            # 创建损失函数实例
            mse = keras.losses.MeanSquaredError()
            
            # 计算学生模型的损失（都是NHWC格式）
            student_loss = mse(y, student_predictions)
            
            # 计算蒸馏损失（都是NHWC格式）
            distillation_loss = mse(
                teacher_predictions / self.temperature,
                student_predictions / self.temperature
            )
            
            # 计算总损失
            total_loss = student_loss + distillation_loss
        
        # 计算梯度并更新学生模型的权重
        trainable_vars = self.student_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 更新指标
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.student_loss_tracker.update_state(student_loss)
        
        # 更新编译的指标
        self.compiled_metrics.update_state(y, student_predictions)
        
        # 返回包含所有指标的字典
        results = {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        x, y = data
        
        # 获取学生模型的预测（使用当前帧，NHWC格式）
        current_frame = x[..., 3:6]  # 提取中间帧
        student_predictions = self.student_model(current_frame, training=False)  # 输出为NHWC
        
        # 更新编译的指标
        self.compiled_metrics.update_state(y, student_predictions)
        
        # 返回包含所有指标的字典
        return {m.name: m.result() for m in self.metrics}

def custom_loss(y_true, y_pred):
    """自定义损失函数，使用tf操作而不是K后端"""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    y_pred_inverse = tf.clip_by_value(1 - y_pred, epsilon, 1.0)
    
    term1 = tf.square(1 - y_pred) * y_true * tf.math.log(y_pred)
    term2 = tf.square(y_pred) * (1 - y_true) * tf.math.log(y_pred_inverse)
    
    return -1.0 * (term1 + term2)

class DataGenerator:
    def __init__(self, data_dir, split, batch_size=BATCH_SIZE):
        self.data_dir = os.path.join(data_dir, split)
        self.batch_size = batch_size
        self.load_annotations()
        
    def load_annotations(self):
        annotation_file = os.path.join(self.data_dir, "_annotations.coco.json")
        print(f"尝试加载标注文件: {annotation_file}")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"找不到标注文件: {annotation_file}")
            
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
            
        # 按照文件名排序图像，以确保连续帧的顺序正确
        self.images = sorted(self.annotations['images'], key=lambda x: x['file_name'])
        self.image_ids = [img['id'] for img in self.images]
        
        # 由于需要三帧连续图像，实际可用的样本数量要减少2
        self.total_samples = len(self.image_ids) - 2
        logger.info(f"Loaded {len(self.image_ids)} images, {self.total_samples} valid sequences from {self.data_dir}")
        
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size
    
    def process_image(self, img_info):
        # 加载和处理图像
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
            
        img = Image.open(img_path)
        img = img.resize((WIDTH, HEIGHT))
        img = np.array(img, dtype=np.float32) / 255.0
        
        # 创建标签
        label = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        img_annots = [ann for ann in self.annotations['annotations'] if ann['image_id'] == img_info['id']]
        
        for ann in img_annots:
            bbox = ann['bbox']
            center_x = int(bbox[0] + bbox[2]/2)
            center_y = int(bbox[1] + bbox[3]/2)
            label[max(0, center_y-5):min(HEIGHT, center_y+5), 
                  max(0, center_x-5):min(WIDTH, center_x+5)] = 1
        
        return img, label
    
    def get_sequence(self, index):
        """获取三帧连续图像和中间帧的标签"""
        # 获取连续的三帧图像信息
        prev_img_info = self.images[index]
        curr_img_info = self.images[index + 1]
        next_img_info = self.images[index + 2]
        
        # 加载三帧图像
        prev_img, _ = self.process_image(prev_img_info)
        curr_img, curr_label = self.process_image(curr_img_info)
        next_img, _ = self.process_image(next_img_info)
        
        # 将三帧图像在通道维度上连接
        combined_input = tf.concat([prev_img, curr_img, next_img], axis=-1)
        
        return combined_input, curr_label
    
    def create_dataset(self):
        def generator():
            indices = list(range(self.total_samples))
            while True:
                # 随机打乱索引
                np.random.shuffle(indices)
                for i in range(0, len(indices), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch_inputs = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        combined_input, label = self.get_sequence(idx)
                        batch_inputs.append(combined_input)
                        batch_labels.append(label)
                    
                    # 转换为张量并清理内存
                    batch_inputs = tf.stack(batch_inputs)
                    batch_labels = tf.stack(batch_labels)
                    gc.collect()
                    
                    yield batch_inputs, batch_labels
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, HEIGHT, WIDTH, 9), dtype=tf.float32),  # 3帧 * 3通道 = 9通道
                tf.TensorSpec(shape=(None, HEIGHT, WIDTH, 3), dtype=tf.float32)
            )
        )
        
        # 配置数据集以优化性能
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def main():
    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("找到GPU设备:", len(gpus))
            # 设置默认GPU设备
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # 不启用混合精度训练，使用float32
            tf.keras.backend.set_floatx('float32')
            print("使用float32精度训练")
        except RuntimeError as e:
            print(e)
    else:
        print("未找到GPU设备，将使用CPU训练")
    
    # 基础数据目录
    base_dir = "/mnt/c/Users/Striker/Downloads/Tennis Ball detection.v2i.coco-mmdetection"
    
    # 减小批次大小
    global BATCH_SIZE
    BATCH_SIZE = 4  # 减小批次大小
    
    # 创建数据生成器
    train_gen = DataGenerator(base_dir, "train", batch_size=BATCH_SIZE)
    valid_gen = DataGenerator(base_dir, "valid", batch_size=BATCH_SIZE)
    test_gen = DataGenerator(base_dir, "test", batch_size=BATCH_SIZE)
    
    # 创建tf.data.Dataset
    train_dataset = train_gen.create_dataset()
    valid_dataset = valid_gen.create_dataset()
    test_dataset = test_gen.create_dataset()
    
    print(f"训练集大小: {train_gen.total_samples}")
    print(f"验证集大小: {valid_gen.total_samples}")
    print(f"测试集大小: {test_gen.total_samples}")
    
    # 创建和加载教师模型
    with tf.device('/GPU:0'):
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model906_30")
            model_h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model906_30.h5")
            
            # 尝试加载模型
            if os.path.exists(model_h5_path):
                print(f"正在加载H5格式的模型: {model_h5_path}")
                teacher_model = tf.keras.models.load_model(model_h5_path, custom_objects={'custom_loss': custom_loss})
            elif os.path.exists(model_path):
                print(f"正在加载SavedModel格式的模型: {model_path}")
                teacher_model = keras.layers.TFSMLayer(
                    model_path,
                    call_endpoint='serving_default'
                )
            else:
                raise FileNotFoundError("找不到模型权重文件")
            print("成功加载教师模型")
        except Exception as e:
            print(f"加载教师模型失败: {e}")
            return
        
        # 创建学生模型
        student_model = ResNet_Track(input_shape=(HEIGHT, WIDTH, 3))
        
        # 创建蒸馏模型
        distillation_model = DistillationModel(
            student_model=student_model,
            teacher_model=teacher_model,
            temperature=TEMPERATURE
        )
        
        # 编译模型，使用float32精度
        distillation_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-4,
                epsilon=1e-7,  # 增加数值稳定性
            ),
            metrics=[keras.metrics.MeanSquaredError()]
        )
        
        # 创建回调函数
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(CHECKPOINT_DIR, "best_student_model.h5"),
                save_best_only=True,
                monitor="val_mean_squared_error",
                mode="min"
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_mean_squared_error",
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_mean_squared_error",
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            keras.callbacks.TensorBoard(
                log_dir=TENSORBOARD_DIR,
                update_freq="epoch",
                write_graph=True,
                profile_batch=0
            )
        ]
        
        # 训练模型
        try:
            print("开始训练模型...")
            print(f"训练轮数: 1")
            print(f"每轮步数: {len(train_gen)}")
            print(f"验证步数: {len(valid_gen)}")
            
            history = distillation_model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=1,  # 设置为1轮
                steps_per_epoch=len(train_gen),
                validation_steps=len(valid_gen),
                callbacks=callbacks,
                verbose=1
            )
            
            print("\n训练完成，开始评估测试集...")
            
            # 在测试集上评估
            test_results = distillation_model.evaluate(
                test_dataset,
                steps=len(test_gen),
                verbose=1
            )
            print("\n测试集结果:")
            if isinstance(test_results, (list, tuple)):
                # 如果是列表或元组，按顺序打印
                for metric_name, value in zip(distillation_model.metrics_names, test_results):
                    if isinstance(value, (int, float)):
                        print(f"{metric_name}: {value:.4f}")
                        logger.info(f"{metric_name}: {value:.4f}")
                    else:
                        print(f"{metric_name}: {value}")
                        logger.info(f"{metric_name}: {value}")
            elif isinstance(test_results, dict):
                # 如果是字典，直接遍历键值对
                for metric_name, value in test_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric_name}: {value:.4f}")
                        logger.info(f"{metric_name}: {value:.4f}")
                    else:
                        print(f"{metric_name}: {value}")
                        logger.info(f"{metric_name}: {value}")
            else:
                # 如果是单个值
                print(f"测试结果: {test_results}")
                logger.info(f"测试结果: {test_results}")
            
            # 保存最终模型
            print("\n开始保存模型...")
            save_status = {
                "keras_model": False,
                "h5_model": False,
                "h5_weights": False,
                "saved_model": False,
                "tf_weights": False
            }
            
            # 1. 保存为.keras格式（推荐的原生格式）
            try:
                final_model_path_keras = os.path.join(CHECKPOINT_DIR, "final_student_model.keras")
                student_model.save(final_model_path_keras)
                save_status["keras_model"] = True
                print(f"成功保存.keras模型到: {final_model_path_keras}")
                logger.info(f"成功保存.keras模型到: {final_model_path_keras}")
            except Exception as e:
                print(f"保存.keras模型失败: {str(e)}")
                logger.error(f"保存.keras模型失败: {str(e)}")
            
            # 2. 保存为H5格式
            try:
                final_model_path_h5 = os.path.join(CHECKPOINT_DIR, "final_student_model.h5")
                student_model.save(final_model_path_h5)
                save_status["h5_model"] = True
                print(f"成功保存H5模型到: {final_model_path_h5}")
                logger.info(f"成功保存H5模型到: {final_model_path_h5}")
            except Exception as e:
                print(f"保存H5模型失败: {str(e)}")
                logger.error(f"保存H5模型失败: {str(e)}")
            
            # 3. 保存H5权重
            try:
                final_weights_path_h5 = os.path.join(CHECKPOINT_DIR, "final_student_model.weights.h5")  # 修改文件名格式
                student_model.save_weights(final_weights_path_h5)
                save_status["h5_weights"] = True
                print(f"成功保存H5权重到: {final_weights_path_h5}")
                logger.info(f"成功保存H5权重到: {final_weights_path_h5}")
            except Exception as e:
                print(f"保存H5权重失败: {str(e)}")
                logger.error(f"保存H5权重失败: {str(e)}")
            
            # 4. 保存为SavedModel格式
            try:
                saved_model_dir = os.path.join(CHECKPOINT_DIR, "final_student_model_saved_model")
                tf.saved_model.save(student_model, saved_model_dir)  # 使用tf.saved_model.save替代
                save_status["saved_model"] = True
                print(f"成功保存SavedModel到: {saved_model_dir}")
                logger.info(f"成功保存SavedModel到: {saved_model_dir}")
            except Exception as e:
                print(f"保存SavedModel失败: {str(e)}")
                logger.error(f"保存SavedModel失败: {str(e)}")
            
            # 5. 保存TensorFlow权重
            try:
                tf_weights_dir = os.path.join(CHECKPOINT_DIR, "final_student_model_tf_weights")
                student_model.save_weights(tf_weights_dir)  # 移除save_format参数
                save_status["tf_weights"] = True
                print(f"成功保存TF权重到: {tf_weights_dir}")
                logger.info(f"成功保存TF权重到: {tf_weights_dir}")
            except Exception as e:
                print(f"保存TF权重失败: {str(e)}")
                logger.error(f"保存TF权重失败: {str(e)}")
            
            # 打印保存状态总结
            print("\n模型保存状态总结:")
            for format_name, status in save_status.items():
                status_text = "成功" if status else "失败"
                print(f"{format_name}: {status_text}")
            
            # 训练完成后，将成功保存的模型复制到原始目录
            try:
                original_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
                os.makedirs(original_output_dir, exist_ok=True)
                
                # 复制所有成功保存的模型到原始目录
                if save_status["keras_model"]:
                    keras_model_dest = os.path.join(original_output_dir, "final_student_model.keras")
                    shutil.copy2(final_model_path_keras, keras_model_dest)
                    print(f"成功复制.keras模型到: {keras_model_dest}")
                    logger.info(f"成功复制.keras模型到: {keras_model_dest}")
                
                if save_status["h5_model"]:
                    h5_model_dest = os.path.join(original_output_dir, "final_student_model.h5")
                    shutil.copy2(final_model_path_h5, h5_model_dest)
                    print(f"成功复制H5模型到: {h5_model_dest}")
                    logger.info(f"成功复制H5模型到: {h5_model_dest}")
                
                if save_status["h5_weights"]:
                    h5_weights_dest = os.path.join(original_output_dir, "final_student_model_weights.h5")
                    shutil.copy2(final_weights_path_h5, h5_weights_dest)
                    print(f"成功复制H5权重到: {h5_weights_dest}")
                    logger.info(f"成功复制H5权重到: {h5_weights_dest}")
                
                if save_status["saved_model"]:
                    saved_model_dest = os.path.join(original_output_dir, "final_student_model_saved_model")
                    if os.path.exists(saved_model_dest):
                        shutil.rmtree(saved_model_dest)
                    shutil.copytree(saved_model_dir, saved_model_dest)
                    print(f"成功复制SavedModel到: {saved_model_dest}")
                    logger.info(f"成功复制SavedModel到: {saved_model_dest}")
                
                if save_status["tf_weights"]:
                    tf_weights_dest = os.path.join(original_output_dir, "final_student_model_tf_weights")
                    if os.path.exists(tf_weights_dest):
                        shutil.rmtree(tf_weights_dest)
                    shutil.copytree(tf_weights_dir, tf_weights_dest)
                    print(f"成功复制TF权重到: {tf_weights_dest}")
                    logger.info(f"成功复制TF权重到: {tf_weights_dest}")
                
                # 复制最佳模型（如果存在）
                best_model_path = os.path.join(CHECKPOINT_DIR, "best_student_model.h5")
                if os.path.exists(best_model_path):
                    best_model_dest = os.path.join(original_output_dir, "best_student_model.h5")
                    shutil.copy2(best_model_path, best_model_dest)
                    print(f"成功复制最佳模型到: {best_model_dest}")
                    logger.info(f"成功复制最佳模型到: {best_model_dest}")
                
                # 打印复制状态总结
                print("\n模型复制状态总结:")
                copied_files = 0
                for format_name, status in save_status.items():
                    if status:
                        copied_files += 1
                        print(f"{format_name}: 已复制")
                    else:
                        print(f"{format_name}: 未复制（保存失败）")
                print(f"\n共成功复制 {copied_files} 种格式的模型到输出目录")
                
            except Exception as e:
                print(f"复制模型到原始目录失败: {str(e)}")
                logger.error(f"复制模型到原始目录失败: {str(e)}")
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            logger.error(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    main() 