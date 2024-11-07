# Yous data seemed to use simple blotches?
#
# Blotches where just black blotches were added to frames

# Import

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import os
import random

from keras import layers, models
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# Load Data

def data2Patches(image, patch_size=(256,256)):
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    patches = []

    for i in range(0, img_height - patch_height + 1, patch_height):
        for j in range(0, img_width - patch_width + 1, patch_width):
            patch = image[i:i + patch_height, j:j + patch_width]
            patches.append(patch)

    patches = np.array(patches)

    return patches

# C:\Users\brian\Desktop\MAIProject\MP4\training_data
# Load training data
def loadData(datapath, numfiles):
    print("Loading Data...")

    images = np.empty((0, 256, 256, 3))
    masks = np.empty((0, 256, 256, 3))

    for i in range(1, numfiles+1):
        #Load Respective Image and Masks
        imagepath = os.path.join(datapath, f"{i}/frame_1.npy")
        # maskpath = os.path.join(datapath, f"{i}/mask_1.npy")
        maskpath = os.path.join(datapath, f"{i}/blotch_1.npy")

        img = np.load(imagepath)
        mask = np.load(maskpath)

        images = np.concatenate((images, data2Patches(img)))
        masks = np.concatenate((masks, data2Patches(mask)))
    
    #images = np.array(images)
    #masks = np.array(masks)

    print(images.shape)
    print(masks.shape)

    print(f"{numfiles} Image(s) Loaded!")

    masks = np.where(masks == 0, 1 , 0)

    # Convert YData into 1 hot encoded
    #one_hot_array = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 2), dtype=np.uint8)
    #one_hot_array[..., 0] = np.where(masks[:,:,:,1] == 0, 1, 0)  # Class 0
    #one_hot_array[..., 1] = np.where(masks[:,:,:,1] != 0, 1, 0)  # Class 1

    return images, masks[:,:,:,0] #one_hot_array


XData, YData = loadData("C:/Users/brian/Desktop/MAIProject/MP4/training_data/", 20)

# Converted into FP64 somewhere????
if XData.dtype != np.uint8:
    XData = XData.astype(np.uint8)

if YData.dtype != np.uint8:
    YData = YData.astype(np.uint8)

#for randomExample in range(0, XData.shape[0]-1):
randomExample = random.randint(0, XData.shape[0] - 1)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(XData[randomExample])
axs[0].title.set_text('Input')
axs[1].imshow(YData[randomExample])
axs[1].title.set_text('Output 1')
#axs[2].imshow(YData[randomExample][:,:,1])
#axs[2].title.set_text('Output 2')
plt.subplots_adjust(top=1.1)
plt.show()


# CNN

def yousModel(input_shape):
    inputs = layers.Input(input_shape)

    #Encoder Layers
    c1 = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal', padding="same")(inputs)
    c2 = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal', padding="same")(c1)
    c3 = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal', padding="same")(c2)
    u1 = layers.concatenate([c1,c2,c3])
    c4 = layers.Conv2D(64, (1, 1), activation="relu", kernel_initializer='he_normal', padding="same")(u1)
    p1 = layers.MaxPooling2D((2, 2))(c4)
    
    #Decoder Layers
    u2 = layers.UpSampling2D((2, 2))(p1) # May need to be bilinear?
    u3 = layers.Conv2D(32, (2, 2), activation="relu", kernel_initializer='he_normal', padding="same")(u2)
    u4 = layers.Conv2D(32, (2, 2), activation="relu", kernel_initializer='he_normal', padding="same")(u3)
    # One-Hot Encoding
    #c5 = layers.Conv2D(2, (3, 3), activation="relu", kernel_initializer='he_normal', padding="same")(u4)
    #outputs = tf.keras.layers.Softmax(name='softmax')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)
    #outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)
    return models.Model(inputs=[inputs], outputs=[outputs])

def weighted_binary_cross_entropy(y_true, y_pred, class_weights):
    # Clip predictions to avoid log(0) issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Compute binary cross-entropy loss
    bce_loss = - (class_weights[0] * y_true * tf.math.log(y_pred) +
                  class_weights[1] * (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(bce_loss)


input_shape = (256, 256, 3)
model = yousModel(input_shape)

# loss="categorical_crossentropy" - Needs class weighting or something, dumb classifier predicts no blotch all the time
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.00001, weight_decay=0.001),
              loss=lambda y_true, y_pred: weighted_binary_cross_entropy(y_true, y_pred, class_weights),
              metrics=['accuracy', 'AUC'])
# model.summary()

X_train, X_test, Y_train, Y_test = train_test_split(XData, YData, test_size=.1, random_state=42)

# Calculate class weights based on the distribution of the classes
y_train_flat = Y_train.flatten()  # Flatten your target array if needed
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
class_weights = dict(zip(np.unique(y_train_flat), class_weights))

print(f"Class Weights: {class_weights}")

history = model.fit(X_train, Y_train, epochs=7, batch_size=4, validation_data=(X_test, Y_test))

predict_Y = model.predict(X_test)

for randomExample in range(0, X_test.shape[0]-1):
    #randomExample = random.randint(0, X_test.shape[0] - 1)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(X_test[randomExample])
    axs[0].title.set_text('Input')
    axs[1].imshow(Y_test[randomExample])
    axs[1].title.set_text('Output')
    axs[2].imshow(predict_Y[randomExample])
    axs[2].title.set_text('Predicted Output')
    plt.subplots_adjust(top=1.1)
    plt.show()