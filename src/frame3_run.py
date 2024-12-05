import tensorflow as tf 
import keras 
import os 
from dataset_func import dataset_func
from frame3_unet import frame3_unet

# Tensorboard Command:
# tensorboard --logdir=./results/frame3_unet/logs

EPOCHS = 500
LR = 1e-3 
BATCH_SIZE_PER_GPU = 32
BATCH_SIZE = int(BATCH_SIZE_PER_GPU*2)

dataFolder = 'E:/MAI_data/'

resultsFolder = './results/frame3_unet'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(resultsFolder, 'models/encoder_decoder{epoch:02d}.keras'))

loss = 'binary_crossentropy' # combined_ssim_l1_loss

dataset_train = dataset_func(os.path.join(dataFolder, 'train'), BATCH_SIZE, 2)
dataset_test = dataset_func(os.path.join(dataFolder, 'test'), BATCH_SIZE, 2)

model = frame3_unet
optimizer = keras.optimizers.Adam(LR)
model.compile(optimizer, loss=loss, metrics=['mse'])

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(resultsFolder, 'logs'))

model.fit(dataset_train, epochs=EPOCHS, callbacks=[tensorboard_callback, model_checkpoint_callback], validation_data=dataset_test)