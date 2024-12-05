import tensorflow as tf 
import keras 
from keras.layers import Conv2DTranspose, Concatenate, Input, Conv2D, MaxPooling2D, Dropout

inputs = Input(shape=(128, 128, 3))

c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.3)(c3)
c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.5)(c4)
c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
u5 = Concatenate()([u5, c3])
u5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
u5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
u6 = Concatenate()([u6, c2])
u6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
u6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
u7 = Concatenate()([u7, c1])
u7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
u7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(u7)

frame1_unet = keras.models.Model(inputs=inputs, outputs=outputs, name="3frame_unet_model")

dot_img_file = 'tmp/frame1_unet.png'
keras.utils.plot_model(frame1_unet, to_file=dot_img_file, show_shapes=True)