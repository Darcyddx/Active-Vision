import tensorflow as tf

# 1. Load the Keras H5 model
model = tf.keras.models.load_model('tennis_rnn.h5')

# 2. Create the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. (Optional) Enable default optimization — here using dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Allow retaining some TensorFlow native ops (Select TF Ops), and disable TensorList downgrade attempt
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,    # Standard TFLite built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS       # Retain TF native ops
]
# Disable experimental TensorList ops downgrade (to avoid automatic conversion failure)
converter._experimental_lower_tensor_list_ops = False

# 5. Perform the conversion
try:
    tflite_model = converter.convert()
except Exception as e:
    # If there are other errors, adjust based on the prompt
    print("Conversion failed:", e)
    raise

# 6. Save the .tflite file
output_path = 'model_with_select_tf_ops.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f'Conversion completed, saved to {output_path}')
