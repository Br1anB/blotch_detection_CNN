import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

# load checkpoint
checkpoint_path = './results/frame3_unet/models/encoder_decoder105.keras'
model = tf.keras.models.load_model(checkpoint_path)

folder_path = 'E:/MAI_data/test/'

num_images = 100

for i in range(num_images):
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    if not npz_files:
        raise ValueError(f"No .npz files found in {folder_path}")

    # select random file
    selected_npz_file = random.choice(npz_files)

    file_path = os.path.join(folder_path, selected_npz_file)
    npz_data = np.load(file_path)

    prev_frame = npz_data['prev_deg']
    current_frame = npz_data['deg']
    next_frame = npz_data['nxt_deg']
                          
    image = np.concatenate((prev_frame, current_frame, next_frame), axis=-1)
    mask = npz_data['mask']

    # Normalise
    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    image_input = np.expand_dims(image, axis=0)


    predicted_mask = model.predict(image_input)

    # Threshold for binary implementation
    # predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(npz_data['deg'])
    plt.title('Input Image')
    plt.axis('off')


    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask.squeeze())  # .squeeze() to remove the batch dimension
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title('Ground Truth')
    plt.axis('off')

    plt.show()