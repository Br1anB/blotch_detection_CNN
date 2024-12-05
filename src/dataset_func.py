import tensorflow as tf 
import keras
from tqdm import tqdm
import numpy as np
import os

# Mode selects the input type in this case, mode=1 selects 1 single input frame, mode=2 selects 3 input frames
# TODO mode=3 will use difference, hence return 128x128x9 input and 128x128x1 input mask, diff2-1, frame 2, diff2-3
def dataset_func(folder_path, batch_size=32, mode=1):
    npy_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    npy_files = npy_files[:10000]  # use only ten files for initial debug
    
    match mode:
        case 1:
            print("Mode 1")

            inputs = np.ones((len(npy_files), 128, 128, 3), dtype=np.uint8)
            outputs = np.ones((len(npy_files), 128, 128, 1), dtype=np.uint8)

            for idx, _file in enumerate(tqdm(npy_files, desc=folder_path)):
                data = np.load(_file) # avoid loading file twice
                inputs[idx] = data['deg'] # np.load(_file)['deg']
                outputs[idx] = data['mask'][...,[0]] # np.load(_file)['mask'][...,[0]]
        case 2:
            print("Mode 2")

            inputs = np.ones((len(npy_files), 128, 128, 9), dtype=np.uint8)
            outputs = np.ones((len(npy_files), 128, 128, 1), dtype=np.uint8)

            for idx, _file in enumerate(tqdm(npy_files, desc=folder_path)):
                data = np.load(_file) # avoid loading file twice

                prev_frame = data['prev_deg']
                current_frame = data['deg']
                next_frame = data['nxt_deg']

                inputs[idx] = np.concatenate((prev_frame, current_frame, next_frame), axis=-1)
                outputs[idx] = data['mask'][...,[0]] # np.load(_file)['mask'][...,[0]]
        case _:
            print("Unknown mode, FAILURE")
            return None

    input_dataset = tf.data.Dataset.from_tensor_slices(inputs)
    output_dataset = tf.data.Dataset.from_tensor_slices(outputs)
    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    dataset = dataset.shuffle(buffer_size=len(npy_files))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, tf.cast(y, tf.float32)/255.0), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset