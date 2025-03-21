import numpy as np
import logging
import sys, getopt
import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tracknet_origin import TrackNet3_CL
# from tensorflow.keras import optimizers
from tensorflow.keras.activations import *
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import AdamW, Adadelta
import cv2
import math
import matplotlib.pyplot as plt
import gc
from ResNet_Unet import ResNet_Track
# from MobileNetV2 import LiteMobileNetV2
# from ResNet18_Unet import ResNet18_UNet
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# Setup logging
logging.basicConfig(
    filename='distillation-tracknet-0320-mobilenet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

BATCH_SIZE=3
HEIGHT=288
WIDTH=512
#HEIGHT=360
#WIDTH=640
mag = 1
sigma = 2.5

def convert_weights_channel_first_to_last(model_from, model_to):
    # Iterate over the layers in both models
    for layer_from, layer_to in zip(model_from.layers, model_to.layers):
        weights_from = layer_from.get_weights()
        weights_to = layer_to.get_weights()

        if weights_from and weights_to:
            converted_weights = []
            for w_from, w_to in zip(weights_from, weights_to):
                if w_from.shape != w_to.shape:
                    if len(w_from.shape) == 4:
                        # For Conv2D layers
                        # Transpose from (out_channels, in_channels, height, width) to (height, width, in_channels, out_channels)
                        w_from = np.transpose(w_from, (2, 3, 1, 0))
                    elif len(w_from.shape) == 1:
                        # For BatchNormalization and biases (1D arrays), shapes should match
                        pass
                    else:
                        # For other layers (e.g., Dense), transpose if necessary
                        w_from = w_from.T
                converted_weights.append(w_from)
            # Set the converted weights to the layer in the channels-last model
            layer_to.set_weights(converted_weights)
    print("Weights converted and loaded into the channels-last model.")



#Return the numbers of true positive, true negative, false positive and false negative
def outcome(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
                TN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
                FP2 += 1
            elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
                FN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
                h_pred = y_pred[i][j] * 255
                h_true = y_true[i][j] * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                #h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                #h_true
                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)

def calc_metric(TP, TN, FP1, FP2, FN):
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0
    return accuracy, precision, recall, f1


try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'load_weights=',
        'save_weights=',
        'dataDir=',
        'epochs=',
        'tol='
    ])
    if len(opts) < 4:
        raise ''
except:
    print('usage: python3 train_finetune.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

paramCount={
    'load_weights': 0,
    'save_weights': 0,
    'dataDir': 0,
    'epochs': 0,
    'tol': 0
}

for (opt, arg) in opts:
    if opt == '--load_weights':
        paramCount['load_weights'] += 1
        load_weights = arg
    elif opt == '--save_weights':
        paramCount['save_weights'] += 1
        save_weights = arg
    elif opt == '--dataDir':
        paramCount['dataDir'] += 1
        dataDir = arg
    elif opt == '--epochs':
        paramCount['epochs'] += 1
        epochs = int(arg)
    elif opt == '--tol':
        paramCount['tol'] += 1
        tol = int(arg)
    else:
        print('usage: python3 train_finetune.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
        print('argument --load_weights is required only if you want to retrain the model')
        exit(1)

if paramCount['save_weights'] == 0 or paramCount['dataDir'] == 0 or paramCount['epochs'] == 0 or paramCount['tol'] == 0:
    print('usage: python3 train_finetune.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

#Loss function
def custom_loss(y_true, y_pred): #hm_true, hm_pred
   
    #loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))

    return (loss)


def distillation_loss(y_true, y_pred, teacher_pred, T=3.0):
    # 原始任务损失
    # task_loss = K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + \
    #             K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    task_losses = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))

    task_loss = K.mean(task_losses)
    # ====================== distillation loss ======================
    # 对每个像素的通道维度单独应用 softmax
    teacher_probs = K.softmax(teacher_pred / T, axis=-1)  # shape=(batch, H, W, 3)
    student_probs = K.softmax(y_pred / T, axis=-1)        # shape=(batch, H, W, 3)
    
    # 计算每个像素的 KL 散度，再在空间维度求平均
    kl_per_pixel = teacher_probs * (K.log(teacher_probs) - K.log(student_probs))  # shape=(batch, H, W, 3)
    kl_sum = K.sum(kl_per_pixel, axis=-1)  # shape=(batch, H, W)
    dist_loss = K.mean(kl_sum)
    
    # ====================== combination loss======================
    return 0.7 * task_loss + 0.3 * dist_loss


#Training for the first time
if paramCount['load_weights'] == 0:
    model=TrackNet3_CL(HEIGHT, WIDTH)
    ADADELTA = Adadelta(lr=1.0)
    model.compile(loss=custom_loss, optimizer=ADADELTA, metrics=['accuracy'])
    # Use AdamW optimizer
    # adamw = AdamW(learning_rate=0.01, weight_decay=0.001)  # Adjust weight_decay as needed
    # model.compile(loss=custom_loss, optimizer=adamw, metrics=['accuracy'])
#Retraining
else:
    model_channel_first = load_model(load_weights, custom_objects={'custom_loss':custom_loss})
    teacher_model = TrackNet3_CL(HEIGHT, WIDTH)
    convert_weights_channel_first_to_last(model_channel_first, teacher_model)
    teacher_model.trainable = False
    print("load pre-trained model")

student_model=ResNet_Track(input_shape=(HEIGHT, WIDTH, 9))
student_model.build(input_shape=(None, HEIGHT, WIDTH, 9))
# student_model = ResNet18_UNet(input_shape=(288, 512, 9))
# input_shape = (288, 512, 9)
# student_model.build(input_shape=(None, *input_shape))
adamw = AdamW(learning_rate=1e-4, weight_decay=1e-5)
student_model.compile(
    loss=custom_loss, optimizer=adamw, metrics=['accuracy']
)


r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npz'))
num = len(path) // 2
idx = np.arange(num, dtype='int') + 1

best_f1_score = 0
best_weights_path = None
train_idx, eval_idx = train_test_split(np.arange(num, dtype='int') + 1, test_size=0.2, random_state=42)

print('Beginning training......')

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # teacher predicts the soft label
        teacher_pred = teacher_model(x_batch)
  
        student_pred = student_model(x_batch)
        # print(student_pred.shape)

        # print(y_batch.shape)
       
        loss = distillation_loss(y_batch, student_pred, teacher_pred)
    
   
    gradients = tape.gradient(loss, student_model.trainable_variables)
    student_model.optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
    return loss

loss_list = []
for epoch in range(epochs):
    print(f"============ Epoch {epoch + 1} ================")
    epoch_loss = 0.0
    # loss = 0
    np.random.shuffle(train_idx)

    # Training loop
    for j in train_idx:
        # Load training data (channel-first format)
        data = np.load(os.path.abspath(os.path.join(dataDir, f"data_{j}.npz")))
        x_train = data["x_data"]  # Shape:  (batch_size, height, width, channels)
        y_train = data["y_data"]  # Shape:  (batch_size, height, width, channels)

        num_samples = x_train.shape[0]
        if num_samples == 0:
            continue
    
        num_batches = num_samples // BATCH_SIZE
        if num_samples % BATCH_SIZE != 0:
            num_batches += 1
        
       
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, num_samples)
           
           
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            
           
            batch_loss = train_step(x_batch, y_batch)
            
            epoch_loss += batch_loss.numpy()
            
        
        del x_train, y_train
        gc.collect()

    avg_epoch_loss = epoch_loss / len(train_idx)
    loss_list.append(avg_epoch_loss)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
    
    # Evaluation loop
    TP = TN = FP1 = FP2 = FN = 0
    for j in eval_idx:
        # Load evaluation data (channel-first format)
        data = np.load(os.path.abspath(os.path.join(dataDir, f"data_{j}.npz")))
        x_eval = data["x_data"]  # Shape: (batch_size, height, width, channels)
        y_eval = data["y_data"]  # Shape: (batch_size, height, width, channels)
        y_eval = np.transpose(y_eval, (0, 3, 1, 2)) # (batch_size, channels, height, width)
        
        # Predict and calculate metrics
        y_pred = student_model.predict(x_eval, batch_size=BATCH_SIZE)
        y_pred = (y_pred > 0.5).astype("float32")  # Threshold the predictions
        y_pred = np.transpose(y_pred, (0, 3, 1, 2))  # Convert to channel-first (NCHW)
        tp, tn, fp1, fp2, fn = outcome(y_pred, y_eval, tol)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn

        del x_eval, y_eval, y_pred
        gc.collect()

    # Print metrics for the current epoch
    print(f"Outcome of training data for epoch {epoch + 1}:")
    print("loss:", avg_epoch_loss)
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP1): {FP1}")
    print(f"False Positives (FP2): {FP2}")
    print(f"False Negatives (FN): {FN}")

    acc, prec, recall, f1 = calc_metric(TP, TN, FP1, FP2, FN)
    print(f"F1 Score for epoch {epoch + 1}: {f1:.4f}")

    logger.info(f"Outcome of training data of epoch {epoch + 1}:")
    logger.info(f"loss: {avg_epoch_loss}")
    logger.info(f"Number of true positives: {TP}")
    logger.info(f"Number of true negatives: {TN}")
    logger.info(f"Number of false positives FP1: {FP1}")
    logger.info(f"Number of false positives FP2: {FP2}")
    logger.info(f"Number of false negatives: {FN}")
    logger.info(f"F1 Score for epoch {epoch + 1}: {f1:.4f}")

    # Save weights if F2 score is the best
    if f1 > best_f1_score:
        best_f1_score = f1
        best_weights_path = save_weights + 'tracknet_distillation_best_0320_mobilenet'
        student_model.save(best_weights_path, include_optimizer=False)
        print(f"New best F1 score: {f1:.4f}. Weights saved to {best_weights_path}")
        logger.info(f"New best F1 score: {f1:.4f}. Weights saved to {best_weights_path}")


print('Saving weights......')
logger.info('Saving final weights...')
last_weights_path = save_weights + 'tracknet_distillation_last_0320_mobilenet'
student_model.save_weights(last_weights_path)
print(f"Final weights saved to {last_weights_path}")
logger.info(f"Final weights saved to {last_weights_path}")
print('Done......')

