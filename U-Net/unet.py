from keras.layers import Input, MaxPooling2D, UpSampling2D, concatenate, Conv2D, BatchNormalization
from keras.models import Model

def get_unet(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv5_2 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool5)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5_2)
    conv5_2 = BatchNormalization()(conv5_2)

    up5_2 = concatenate([(UpSampling2D(size=(2, 2))(conv5_2)), conv5], axis=3)
    conv6_2 = Conv2D(256, (3, 3), activation="relu", padding="same")(up5_2)
    conv6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6_2)
    conv6_2 = BatchNormalization()(conv6_2)

    tmp = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6_2)
    up6 = concatenate([tmp, conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(3, (1, 1), padding="same", activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


