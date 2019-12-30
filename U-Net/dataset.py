import numpy as np
import cv2


def own_frames_load(video_path):
    vidcap = cv2.VideoCapture(video_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    count = 0
    out = np.ndarray((length, image.shape[0], image.shape[1], image.shape[2]), np.float32)

    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out[count] = image
        success, image = vidcap.read()
        count += 1

    return out


def get_dataset(video_path):
    frames = own_frames_load(video_path)
    dataset_size = len(frames)

    print(f'Dataset size: {dataset_size}')
    sample = frames[0]
    val = 0
    train = 0
    X = np.ndarray((dataset_size - dataset_size // 5, sample.shape[0], sample.shape[1], sample.shape[2] * 2),
                   np.float32)
    Y = np.ndarray((dataset_size - dataset_size // 5, sample.shape[0], sample.shape[1], sample.shape[2]), np.float32)
    X_val = np.ndarray((dataset_size // 5, sample.shape[0], sample.shape[1], sample.shape[2] * 2), np.float32)
    Y_val = np.ndarray((dataset_size // 5, sample.shape[0], sample.shape[1], sample.shape[2]), np.float32)
    X_all = np.ndarray((dataset_size, sample.shape[0], sample.shape[1], sample.shape[2] * 2),
                       np.float32)
    for i in range(1, dataset_size - 2):
        X1 = frames[i - 1] / 255.0
        X2 = frames[1 + 1] / 255.0
        X_all[i - 1] = np.concatenate((X1, X2), axis=2)

        if (i - 1) % 5 == 0:
            X_val[val] = np.concatenate((X1, X2), axis=2)
            Y_val[val] = frames[i] / 255.0
            val += 1
            continue

        Y[train] = frames[i] / 255.0
        X[train] = np.concatenate((X1, X2), axis=2)
        train += 1

    print(f'Train size: {X.shape}')
    print(f'Test size: {X_val.shape}')

    return X, Y, X_val, Y_val, X_all
