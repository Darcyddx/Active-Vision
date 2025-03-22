from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D, Input, concatenate
from tensorflow.keras.activations import *


def TrackNet3_CL(input_height, input_width):  # input_height = 288, input_width = 512

    imgs_input = Input(shape=(input_height, input_width, 9))  # Channel-last input format
    # Layer1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same')(imgs_input)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x1 = BatchNormalization(axis=2)(x)

    # Layer3
    x = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # Layer4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x2 = BatchNormalization(axis=2)(x)

    # Layer6
    x = MaxPooling2D((2, 2), strides=(2, 2))(x2)

    # Layer7
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer8
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer9
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x3 = BatchNormalization(axis=2)(x)

    # Layer10
    x = MaxPooling2D((2, 2), strides=(2, 2))(x3)

    # Layer11
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer12
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer13
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer14
    x = concatenate([UpSampling2D((2, 2))(x), x3], axis=-1)

    # Layer15
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer16
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer17
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer18
    x = concatenate([UpSampling2D((2, 2))(x), x2], axis=-1)

    # Layer19
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer20
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer21
    x = concatenate([UpSampling2D((2, 2))(x), x1], axis=-1)

    # Layer22
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer23
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=2)(x)

    # Layer24
    x = Conv2D(3, (1, 1), kernel_initializer='random_uniform', padding='same')(x)
    x = Activation('sigmoid')(x)

    output = x
    model = Model(imgs_input, output)

    return model
