import argparse
import os
import numpy as np
import tensorflow as tf
import pathlib
from glob import glob

HEIGHT = 288
WIDTH = 512

def representative_data_gen(data_dir, num_calibration_samples):
    # List all npz files in the directory
    data_files = glob(os.path.join(data_dir, '*.npz'))
    
    # Counter to keep track of how many samples have been used
    sample_count = 0
    
    for data_file in data_files:
        # Load the data file
        data = np.load(data_file)
        x_data = data['x_data']
        
        for i in range(x_data.shape[0]):
            if sample_count >= num_calibration_samples:
                return
            input_value = x_data[i:i+1, ...]  # Get a single sample, keep batch dimension
            # input_value = np.transpose(input_value, (0, 2, 3, 1))
            print(sample_count, input_value.shape)
            input_value = input_value.astype(np.float32)
            yield [input_value]
            sample_count += 1

def quantization_fp16(model_path, save_path, model_name):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16] 
    converter.experimental_new_converter = False
    quant_and_pruned_tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path(save_path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_quant_file = tflite_models_dir / f"{model_name}_fp16.tflite"
    tflite_model_quant_file.write_bytes(quant_and_pruned_tflite_model)

def quantization_uint8(model_path, data_dir, save_path, num_calibration_samples, model_name):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(data_dir, num_calibration_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = False
    tflite_model_quant = converter.convert()

    tflite_models_dir = pathlib.Path(save_path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir / f"{model_name}_uint8.tflite"
    tflite_model_quant_file.write_bytes(tflite_model_quant)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a TensorFlow model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the Keras model.")
    parser.add_argument('--data_dir', type=str, required=False, help="Directory containing representative dataset.")
    parser.add_argument('--save_path', type=str, required=True, help="Directory to save the quantized model.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the quantized model file.")
    parser.add_argument('--mode', type=str, choices=['fp16', 'uint8'], required=True, help="Quantization mode: 'fp16' or 'uint8'.")
    parser.add_argument('--num_calibration_samples', type=int, default=100, help="Number of samples for calibration (only for uint8).")

    args = parser.parse_args()

    if args.mode == 'fp16':
        quantization_fp16(args.model_path, args.save_path, args.model_name)
    elif args.mode == 'uint8':
        if not args.data_dir:
            raise ValueError("--data_dir is required for uint8 quantization.")
        quantization_uint8(args.model_path, args.data_dir, args.save_path, args.num_calibration_samples, args.model_name)
