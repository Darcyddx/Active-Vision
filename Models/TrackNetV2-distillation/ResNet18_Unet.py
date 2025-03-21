import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import HeNormal

class BasicBlock(keras.layers.Layer):
    def __init__(self, filters, strides=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(
            filters, 3, strides=strides, padding="same", kernel_initializer=HeNormal()
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(
            filters, 3, padding="same", kernel_initializer=HeNormal()
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation("relu")
        self.downsample = downsample

    def call(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += identity
        x = self.relu(x)
        return x

class UpBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        self.up = keras.Sequential([
            keras.layers.Conv2DTranspose(
                out_channels, 2, strides=2, 
                padding="same", kernel_initializer=HeNormal()
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu")
        ])
        
        # Adjust skip connection channels if needed
        if skip_channels != out_channels:
            self.skip_conv = keras.Sequential([
                keras.layers.Conv2D(
                    out_channels, 1, kernel_initializer=HeNormal()
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu")
            ])
        else:
            self.skip_conv = None
            
        self.conv_block = keras.Sequential([
            keras.layers.Conv2D(
                out_channels, 3, padding="same", kernel_initializer=HeNormal()
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(
                out_channels, 3, padding="same", kernel_initializer=HeNormal()
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu")
        ])

    def call(self, x, skip):
        x = self.up(x)
        if self.skip_conv:
            skip = self.skip_conv(skip)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv_block(x)
        return x

class ResNet18_UNet(keras.Model):
    def __init__(self, input_shape):
        super(ResNet18_UNet, self).__init__()
        
        # Initial convolution
        self.initial = keras.Sequential([
            keras.layers.Conv2D(
                64, 3, padding="same", 
                input_shape=input_shape, kernel_initializer=HeNormal()
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(
                64, 3, padding="same", kernel_initializer=HeNormal()
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu")
        ])

        # Encoder blocks
        self.encoder1 = self._make_stage(64, 64, 2, strides=2)   # 144x256
        self.encoder2 = self._make_stage(64, 128, 2, strides=2)  # 72x128
        self.encoder3 = self._make_stage(128, 256, 2, strides=2) # 36x64
        self.encoder4 = self._make_stage(256, 512, 2, strides=2) # 18x32

        # Decoder blocks
        self.decoder1 = UpBlock(512, 256, 256)  # 36x64
        self.decoder2 = UpBlock(256, 128, 128)  # 72x128
        self.decoder3 = UpBlock(128, 64, 64)    # 144x256
        self.decoder4 = UpBlock(64, 64, 64)     # 288x512

        # Final output
        self.final_conv = keras.layers.Conv2D(
            3, 3, padding="same", activation="sigmoid"
        )

    def _make_stage(self, in_channels, out_channels, blocks, strides=1):
        downsample = None
        if strides != 1 or in_channels != out_channels:
            downsample = keras.Sequential([
                keras.layers.Conv2D(
                    out_channels, 1, strides=strides, 
                    kernel_initializer=HeNormal()
                ),
                keras.layers.BatchNormalization()
            ])
            
        layers = [BasicBlock(out_channels, strides, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels))
            
        return keras.Sequential(layers)

    def call(self, inputs):
        # Encoder
        s0 = self.initial(inputs)  # 288x512x64
        s1 = self.encoder1(s0)     # 144x256x64
        s2 = self.encoder2(s1)     # 72x128x128
        s3 = self.encoder3(s2)     # 36x64x256
        s4 = self.encoder4(s3)     # 18x32x512

        # Decoder with skip connections
        d1 = self.decoder1(s4, s3)  # 36x64x256
        d2 = self.decoder2(d1, s2)  # 72x128x128
        d3 = self.decoder3(d2, s1)  # 144x256x64
        d4 = self.decoder4(d3, s0)  # 288x512x64
        
        return self.final_conv(d4)

if __name__ == "__main__":
    model = ResNet18_UNet(input_shape=(288, 512, 9))
    model.build((None, 288, 512, 9))
    model.summary()
    
    dummy_input = tf.random.normal((1, 288, 512, 9))
    dummy_output = model(dummy_input)
    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", dummy_output.shape)