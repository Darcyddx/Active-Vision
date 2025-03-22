import os
import numpy as np
import tensorflow as tf
import pathlib
from glob import glob
from trackNet_origin import TrackNet3_CL

HEIGHT = 288
WIDTH = 512
def representative_data_gen():
    # Path to your data directory
    data_dir = "/root/autodl-tmp/dataset/npz-33"
    # List all npz files in the directory
    data_files = glob(os.path.join(data_dir, '*.npz'))
    
    # Number of samples to use for calibration
    num_calibration_samples = 100  # You can adjust this number based on your dataset size
    
    # Counter to keep track of how many samples have been used
    sample_count = 0
    
    for data_file in data_files:
        # Load the data file
        data = np.load(data_file)
        x_data = data['x_data']  # Shape: (batch_size, channels, height, width)
        
        for i in range(x_data.shape[0]):
            print(sample_count)
            if sample_count >= num_calibration_samples:
                return
            input_value = x_data[i:i+1, ...]  # Get a single sample, keep batch dimension
            
            input_value = input_value.astype(np.float32)
            input_value = np.transpose(input_value, (0, 2, 3, 1))
            print(input_value.shape)
            # Yield the input data as a list
            yield [input_value]
            sample_count += 1


def quantization_fp16():
    model_weight = "/root/workspace/tracknet-finetune/tracknet-finetune-1227/tracknetv2_best_finetune_1227.h5"
    model = TrackNet3_CL(HEIGHT, WIDTH)
    model.load_weights(model_weight)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16] 
    converter.experimental_new_converter = False
    quant_and_pruned_tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("./tflite-1211/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_quant_file = tflite_models_dir/"tracknetv2_1211_fp16_new.tflite"
    tflite_model_quant_file.write_bytes(quant_and_pruned_tflite_model)

def quantization_uint8():
    model_weight = "/root/workspace/tracknet-finetune/tracknet-finetune-1227/tracknetv2_best_finetune_1227.h5"
    model = TrackNet3_CL(HEIGHT, WIDTH)
    model.load_weights(model_weight)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = False
    tflite_model_quant = converter.convert()

    tflite_models_dir = pathlib.Path("./tflite-1227-finetune")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir/"tracknetv2_1227_finetune_rgb.tflite"
    tflite_model_quant_file.write_bytes(tflite_model_quant)

    interpreter = tf.lite.Interpreter(model_path=tflite_models_dir/"tracknetv2_1227_finetune_rgb.tflite")
    interpreter.allocate_tensors()

    # Print model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

if __name__ == '__main__':
    quantization_uint8()
    # quantization_fp16()

    