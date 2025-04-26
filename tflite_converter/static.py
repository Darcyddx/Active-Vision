import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.models import Model

# —— 1. Rebuild the network architecture ——
time_steps  = 30   # Sequence length (from the model configuration)
feature_dim = 26   # Input feature dimensions
# Key modification: unroll=True
inputs = Input(shape=(time_steps, feature_dim), name="gru_input")
x = GRU(units=24, dropout=0.1, unroll=True, name="gru")(inputs)
x = Dropout(rate=0.2, name="dropout")(x)
x = Dense(units=8, activation="relu", name="dense")(x)
outputs = Dense(units=4, activation="softmax", name="dense_1")(x)

model = Model(inputs=inputs, outputs=outputs, name="tennis_rnn_unrolled")
model.summary()

# —— 2. Load the original weights ——
model.load_weights("tennis_rnn.h5")

# —— 3. Convert to TFLite ——
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# —— If quantization is needed, uncomment the line below ——
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open("tennis_rnn_unrolled.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion completed, output file: tennis_rnn_unrolled.tflite")
