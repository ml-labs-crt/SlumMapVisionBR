# Script to train models with half of the image and test with the other half (both halves)

import numpy as np
import pandas as pd
import tensorflow as tf
import skimage.io
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
import sklearn.metrics
import matplotlib.image
from pathlib import Path
from datetime import datetime
import os

BACKBONE = "efficientnetb3"
CLASSES = ["slum"]
LR = 0.001
EPOCHS = 100
BATCH_SIZE = 8
WEIGHT_DECAY = 0.5
SIZE_TILE = 64
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_START_FROM_EPOCH = 20

os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm

sm.set_framework("tf.keras")
keras.backend.set_image_data_format("channels_last")

now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")

dir_results = Path("../results_"+dt_string)
dir_results.mkdir(parents=True, exist_ok=True)
dir_graphs = Path("../results_"+dt_string+"/graphs")
dir_graphs.mkdir(parents=True, exist_ok=True)
dir_models = Path("../results_"+dt_string+"/models")
dir_models.mkdir(parents=True, exist_ok=True)
dir_logs = Path("../results_"+dt_string+"/logs")
dir_logs.mkdir(parents=True, exist_ok=True)
dir_rasters = Path("../results_"+dt_string+"/rasters")
dir_rasters.mkdir(parents=True, exist_ok=True)


def all_metrics(tp, tn, fp, fn):
    """Calculates all metrics for a given confusion matrix.

    Args:
        tp: true positives
        tn: true negatives
        fp: false positives
        fn: false negatives

    Returns:
        tuple: kappa, precision, sensitivity, iou, iou_2, mean_iou, accuracy, f1score
    """
    kappa = (2 * (tp * tn - fn * fp)) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    iou = tp / (tp + fp + fn)
    iou_2 = tn / (tn + fp + fn)
    mean_iou = (iou + iou_2) / 2
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1score = 2 * tp / (2 * tp + fp + fn)
    print("Kappa: {}".format(round(kappa, 3)))
    print("Precision: {}".format(round(precision, 3)))
    print("Sensitivity: {}".format(round(sensitivity, 3)))
    print("IoU Class 1: {}".format(round(iou, 3)))
    print("IoU Class 0: {}".format(round(iou_2, 3)))
    print("mIoU: {}".format(round(mean_iou, 3)))
    print("Accuracy: {}".format(round(accuracy, 3)))
    print("F1Score: {}".format(round(f1score, 3)))
    return kappa, precision, sensitivity, iou, iou_2, mean_iou, accuracy, f1score


def reduce_image_for_irregular_tile_cutting(image_array, tile_height, tile_width):
    """Adjusts the number of rows and columns so the image can be cut into tiles of the desired dimensions.

    Args:
        image_array (numpy.ndarray): array with original imagery (img_height, img_width, channels).
        tile_height (int): height of each tile.
        tile_width (int): width of each tile.

    Returns:
        array: Image reduced
    """
    number_of_tiles_i = int(image_array.shape[0] / tile_height)
    new_number_rows = number_of_tiles_i * tile_height
    number_of_tiles_j = int(image_array.shape[1] / tile_width)
    new_number_columns = number_of_tiles_j * tile_width
    image_reduced = image_array[0:new_number_rows, 0:new_number_columns]
    return image_reduced


def reshape_split(image, kernel_size):
    """Reshapes an array into tiles the a desired kernel size.
    Developed by: Iosif Doundoulakis.
    Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7

    Args:
        image (array): Array to be resized in the format (img_height, img_width, channels).
        kernel_size (tuple): A tuple with (tile_height, tile_width) of the desired kernel.

    Returns:
        array: Reshaped array.
    """
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(
        img_height // tile_height,
        tile_height,
        img_width // tile_width,
        tile_width,
        channels,
    )
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


def tile_data(image, mask, size_tile = 64):      
    '''
    This function takes an image and a mask and returns the image and mask tiled
    according to the size_tile parameter. The image and mask are reduced to a size
    that is a multiple of size_tile. The image is tiled and the mask is tiled and
    expanded to have a fourth dimension.

    Parameters
    ----------
    image : numpy array
        The image to be tiled.
    mask : numpy array
        The mask to be tiled.
    size_tile : int, optional
        The size of the tiles. The default is 64.

    Returns
    -------
    im_tiled : numpy array
        The tiled image.
    mask_tiled : numpy array    
        The tiled and expanded mask.
    '''
    im_red = reduce_image_for_irregular_tile_cutting(
            image, size_tile, size_tile)
    mask_red = reduce_image_for_irregular_tile_cutting(
            mask, size_tile, size_tile)
    mask_red = np.expand_dims(mask_red, axis=-1)
    im_tiled = reshape_split(im_red, (size_tile, size_tile)).reshape(-1, size_tile, size_tile, 7)
    mask_tiled = reshape_split(mask_red, (size_tile, size_tile)).reshape(-1, size_tile, size_tile, 1)
    return im_tiled, mask_tiled.astype(np.float32)


def augment_slums(norm_train_val, mask_original_train_val, size_tile = 64, n_aug = 1000):
    '''
    This function takes the normalized training and validation data and the original
    masks and returns the augmented data and masks (only slums). The augmented data and masks are
    created by randomly selecting pixels labeled with 1 from the original masks and then 
    cropping a patch of the image around them. The default number of augmented data is 1000. 

    Parameters
    ----------
    norm_train_val : numpy array
        The normalized training and validation data.
    mask_original_train_val : numpy array
        The original masks.
    size_tile : int
        The size of the tiles.

    Returns
    -------
    norm_train_val_aug : numpy array
        The augmented normalized training and validation data.
    mask_original_train_val_aug : numpy array   
        The augmented original masks.
    '''
    image_channels = norm_train_val.shape[-1]
    mask_coordinates = np.argwhere(mask_original_train_val == 1)   
    mask_coordinates = mask_coordinates[(mask_coordinates[:,0]>size_tile//2) & (mask_coordinates[:,0]<mask_original_train_val.shape[0]-size_tile//2) & (mask_coordinates[:,1]>size_tile//2) & (mask_coordinates[:,1]<mask_original_train_val.shape[1]-size_tile//2)]
    n_samples = min(n_aug,len(mask_coordinates))
    random_sample = np.random.choice(mask_coordinates.shape[0], n_samples, replace=False)
    mask_coordinates = mask_coordinates[random_sample]
    data_aug = np.zeros((n_samples,size_tile,size_tile, image_channels))
    mask_aug = np.zeros((n_samples,size_tile,size_tile))
    for i in range(n_samples):
        data_aug[i] = norm_train_val[mask_coordinates[i,0]-size_tile//2:mask_coordinates[i,0]+size_tile//2,mask_coordinates[i,1]-size_tile//2:mask_coordinates[i,1]+size_tile//2,:]
        mask_aug[i] = mask_original_train_val[mask_coordinates[i,0]-size_tile//2:mask_coordinates[i,0]+size_tile//2,mask_coordinates[i,1]-size_tile//2:mask_coordinates[i,1]+size_tile//2]
        # plt.imshow(data_aug[1,:,:,0:3])
        # plt.imshow(mask_aug[1])
    mask_aug = np.expand_dims(mask_aug, axis=-1)
    return data_aug, mask_aug


def augment_data(norm_train_val, mask_original_train_val, size_tile = 64, n_samples = 1000, percentage_slum = 0.1):
    '''	
    This function takes the normalized data and the original
    masks (not tiles) and returns the augmented data and masks (tiled). 
    The default number of augmented data is 1000.
    The default percentage of slums is 0.1.

    Parameters
    ----------
    norm_train_val : numpy array
        The normalized data.
    mask_original_train_val : numpy array
        The original masks.
    size_tile : int
        The size of the tiles.
    n_samples : int
        The number of samples to be augmented.
    percentage_slum : float
        The percentage of slums to be augmented.

    Returns
    -------
    data_aug : numpy array
        The augmented data.
    mask_aug : numpy array
        The augmented masks.
    '''
    
    image_channels = norm_train_val.shape[-1]
    mask_coordinates_slum = np.argwhere(mask_original_train_val == 1)   
    mask_coordinates_slum = mask_coordinates_slum[(mask_coordinates_slum[:,0]>size_tile//2) & (mask_coordinates_slum[:,0]<mask_original_train_val.shape[0]-size_tile//2) & (mask_coordinates_slum[:,1]>size_tile//2) & (mask_coordinates_slum[:,1]<mask_original_train_val.shape[1]-size_tile//2)]
    if len(mask_coordinates_slum) < int(n_samples*percentage_slum):
        replace_bool = True
    else:
        replace_bool = False
    np.random.seed(0)
    random_sample = np.random.choice(mask_coordinates_slum.shape[0], int(n_samples*percentage_slum), replace=replace_bool)
    mask_coordinates_slum = mask_coordinates_slum[random_sample]
    data_aug_slum = np.zeros((int(n_samples*percentage_slum),size_tile,size_tile, image_channels))
    mask_aug_slum = np.zeros((int(n_samples*percentage_slum),size_tile,size_tile))
    for i in range(int(n_samples*percentage_slum)):
        data_aug_slum[i] = norm_train_val[mask_coordinates_slum[i,0]-size_tile//2:mask_coordinates_slum[i,0]+size_tile//2,mask_coordinates_slum[i,1]-size_tile//2:mask_coordinates_slum[i,1]+size_tile//2,:]
        mask_aug_slum[i] = mask_original_train_val[mask_coordinates_slum[i,0]-size_tile//2:mask_coordinates_slum[i,0]+size_tile//2,mask_coordinates_slum[i,1]-size_tile//2:mask_coordinates_slum[i,1]+size_tile//2]
        # plt.imshow(data_aug[1,:,:,0:3])
        # plt.imshow(mask_aug[1])
    mask_aug_slum = np.expand_dims(mask_aug_slum, axis=-1)

    mask_coordinates_nonslum = np.argwhere(mask_original_train_val == 0)   
    mask_coordinates_nonslum = mask_coordinates_nonslum[(mask_coordinates_nonslum[:,0]>size_tile//2) & (mask_coordinates_nonslum[:,0]<mask_original_train_val.shape[0]-size_tile//2) & (mask_coordinates_nonslum[:,1]>size_tile//2) & (mask_coordinates_nonslum[:,1]<mask_original_train_val.shape[1]-size_tile//2)]
    if len(mask_coordinates_nonslum) < int(n_samples*(1-percentage_slum)):
        replace_bool = True
    else:
        replace_bool = False
    np.random.seed(0)
    random_sample = np.random.choice(mask_coordinates_nonslum.shape[0], int(n_samples*(1-percentage_slum)), replace=replace_bool)
    mask_coordinates_nonslum = mask_coordinates_nonslum[random_sample]
    data_aug_nonslum = np.zeros((int(n_samples*(1-percentage_slum)),size_tile,size_tile, image_channels))
    mask_aug_nonslum = np.zeros((int(n_samples*(1-percentage_slum)),size_tile,size_tile))
    for i in range(int(n_samples*(1-percentage_slum))):
        data_aug_nonslum[i] = norm_train_val[mask_coordinates_nonslum[i,0]-size_tile//2:mask_coordinates_nonslum[i,0]+size_tile//2,mask_coordinates_nonslum[i,1]-size_tile//2:mask_coordinates_nonslum[i,1]+size_tile//2,:]
        mask_aug_nonslum[i] = mask_original_train_val[mask_coordinates_nonslum[i,0]-size_tile//2:mask_coordinates_nonslum[i,0]+size_tile//2,mask_coordinates_nonslum[i,1]-size_tile//2:mask_coordinates_nonslum[i,1]+size_tile//2]
    mask_aug_nonslum = np.expand_dims(mask_aug_nonslum, axis=-1)

    data_aug = np.concatenate((data_aug_slum, data_aug_nonslum), axis=0)
    mask_aug = np.concatenate((mask_aug_slum, mask_aug_nonslum), axis=0)

    return data_aug, mask_aug


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=0):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(
            mode="horizontal_and_vertical", seed=seed
        )
        self.augment_labels = tf.keras.layers.RandomFlip(
            mode="horizontal_and_vertical", seed=seed
        )

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def reshape_combine(tiled_array, image_shape):
    """Combines an array of tiles into a single image.

    Args:
        tiled_array (array): Array of tiles to be combined.
        image_shape (tuple): A tuple with (img_height, img_width, channels) of the desired output image.

    Returns:
        array: Combined array.
    """
    img_height, img_width, channels = image_shape
    tile_height, tile_width = tiled_array.shape[1], tiled_array.shape[2]

    num_tiles_height = img_height // tile_height
    num_tiles_width = img_width // tile_width

    reconstructed_img = tiled_array.reshape(
        num_tiles_height, num_tiles_width, tile_height, tile_width, channels
    )
    reconstructed_img = reconstructed_img.swapaxes(1, 2)
    reconstructed_img = reconstructed_img.reshape(img_height, img_width, channels)
    return reconstructed_img


PATH_RASTERS = "../BrazilianDataset/intermediary_rasters/"
PATH_MASKS = "../BrazilianDataset/masks/"

results_columns = [
    "location",
    "half",
    "kappa",
    "precision",
    "sensitivity",
    "IoU Class 1",
    "IoU Class 0",
    "mIoU",
    "accuracy",
    "f1score",
    "tn",
    "fp",
    "fn",
    "tp",
]

df_all_results = pd.DataFrame(columns=results_columns)

filtered_locations = pd.read_csv("info_capitals.csv")

filtered_locations = filtered_locations["location"].astype(str).values.tolist()

for i in range(0,len(filtered_locations)):

    LOCATION_ID = filtered_locations[i]

    image_original = skimage.io.imread(
        PATH_RASTERS + LOCATION_ID + "/" + LOCATION_ID + ".tif"
    )
    mask_original = skimage.io.imread(PATH_MASKS + "mask_" + LOCATION_ID + ".tif")

    image_channels = image_original.shape[-1]

    middle_image_y = int(image_original.shape[1] / 2)

    for half in range(2):
        if half == 0:
            image_original_test = image_original[:, 0:middle_image_y, :]
            mask_original_test = mask_original[:, 0:middle_image_y]
            image_original_train_val = image_original[:, middle_image_y:, :]
            mask_original_train_val = mask_original[:, middle_image_y:]
        if half == 1:
            image_original_test = image_original[:, middle_image_y:, :]
            mask_original_test = mask_original[:, middle_image_y:]
            image_original_train_val = image_original[:, 0:middle_image_y, :]
            mask_original_train_val = mask_original[:, 0:middle_image_y]

        layer = tf.keras.layers.Normalization(
        axis=-1)  # axis=-1 is the last dimension (channels)
        layer.adapt(image_original_train_val)

        norm_train_val = layer(image_original_train_val).numpy()
        norm_test = layer(image_original_test).numpy()

        # plt.figure(figsize=(15, 15))
        # plt.imshow(norm_train_val[:,:,0:3])
        # plt.imshow(mask_original_train_val)
        # plt.imshow(norm_test[:,:,0:3])
        # plt.imshow(mask_original_test)
        # norm_train_val.numpy()[:,:,0].mean()
        # norm_train_val.numpy()[:,:,0].std()
        # plt.imshow(mask_original_train_val)
        # plt.imshow(mask_original_test)

        im_tiled, mask_tiled = tile_data(norm_train_val, mask_original_train_val, SIZE_TILE)

        # plt.figure(figsize=(15, 15))
        # plt.imshow(im_tiled[0,:,:,0:3])
        # plt.imshow(mask_tiled[0,:,:])

        indices_slums = []
        for item in range(len(mask_tiled)):
            if np.sum(mask_tiled[item]) > 0:
                indices_slums.append(True)
            else:
                indices_slums.append(False)

        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            im_tiled,
            mask_tiled,
            test_size=0.5,
            random_state=0,
            shuffle=True,
            # stratify=indices_slums, # removed as some locations have few tiles non-slums
        )

        im_aug, mask_aug = augment_data(norm_train_val, mask_original_train_val, size_tile=SIZE_TILE, n_samples=100, percentage_slum = 0.10)

        # plt.figure(figsize=(15, 15))
        # plt.imshow(im_aug[0,:,:,0:3])
        # plt.imshow(mask_aug[0,:,:])

        X_train = np.concatenate((X_train, im_aug), axis=0)
        y_train = np.concatenate((y_train, mask_aug), axis=0)

        # plt.figure(figsize=(15, 15))
        # plt.imshow(X_train[2,:,:,0:3])
        # plt.imshow(y_train[2,:,:])

        train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        normalised_image_tiled_test, mask_tiled_test = tile_data(norm_test, mask_original_test, SIZE_TILE)

        train_batches = (
            train.cache()
            .shuffle(len(train))
            .batch(BATCH_SIZE)
            .map(Augment())
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        val_batches = (
            val.cache()
            .shuffle(len(val))
            .batch(BATCH_SIZE)
            .map(Augment())
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        TRAIN_LENGTH = len(train)
        VAL_LENGTH = len(val)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(dir_models) + "/model_cnn_weights_best_{}_half_{}.h5".format(LOCATION_ID, half),
                monitor="val_loss",
                save_weights_only=True,
                save_best_only=True,
                mode="min",
                verbose=2,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=50, cooldown=20, verbose=2
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=EARLY_STOPPING_PATIENCE,
                verbose=2,
                mode="auto",
                restore_best_weights=True,
                start_from_epoch=EARLY_STOPPING_START_FROM_EPOCH,
            ),
            tf.keras.callbacks.CSVLogger(
                str(dir_logs) + "/model_log_{}_half_{}.csv".format(LOCATION_ID, half),
                separator=",",
                append=False,
            ),
        ]

        n_classes = (
            1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
        )  # case for binary and multiclass segmentation
        activation = "sigmoid" if n_classes == 1 else "softmax"

        # Expand model to work with 7 bands

        N = normalised_image_tiled_test.shape[-1]

        base_model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

        inp = tf.keras.layers.Input(shape=(None, None, N))
        l1 = tf.keras.layers.Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
        out = base_model(l1)

        model = tf.keras.Model(inp, out, name=base_model.name)

        optim = keras.optimizers.AdamW(LR, weight_decay=WEIGHT_DECAY)

        dice_loss = sm.losses.DiceLoss()
        focal_loss = (
            sm.losses.BinaryFocalLoss()
            if n_classes == 1
            else sm.losses.CategoricalFocalLoss()
        )
        total_loss = dice_loss + (1 * focal_loss)

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        model.compile(optim, total_loss, metrics)

        # model.summary()

        STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
        VAL_STEPS = VAL_LENGTH // BATCH_SIZE if VAL_LENGTH // BATCH_SIZE > 0 else 1

        history = model.fit(
            train_batches,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VAL_STEPS,
            validation_data=val_batches,
            callbacks=callbacks,
            verbose=2,
        )

        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history["iou_score"])
        plt.plot(history.history["val_iou_score"])
        plt.title("Model IoU Score")
        plt.ylabel("IoU score")
        plt.xlabel("Epoch")
        plt.legend(["Training", "Validation"], loc="upper left")

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Training", "Validation"], loc="upper left")
        plt.savefig(
            str(dir_graphs) + "/model_plot_{}_half_{}.png".format(LOCATION_ID, half)
        )
        plt.show()

        y_pred = model.predict(normalised_image_tiled_test, verbose=1)
        y_pred_flat = y_pred.ravel()
        y_pred_mask_flat = np.array([1 if x > 0.5 else 0 for x in y_pred_flat])

        y_mask = mask_tiled_test.ravel()

        # np.unique(y_pred_mask_flat, return_counts=True)
        # np.unique(y_mask, return_counts=True)

        plt.figure(figsize=(15, 15))
        cm = sklearn.metrics.confusion_matrix(y_true=y_mask, y_pred=y_pred_mask_flat)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt="g", ax=ax, cbar=False)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix (Test Set)")
        plt.savefig(
            str(dir_graphs) + "/cm_cnn_{}_half_{}.png".format(LOCATION_ID, half)
        )
        plt.show()

        tn, fp, fn, tp = cm.ravel()

        print("Metrics test data:")
        metrics = all_metrics(tp=tp, tn=tn, fp=fp, fn=fn)

        results = [LOCATION_ID] + [half] + list(metrics) + [tn, fp, fn, tp]

        df_results = pd.DataFrame(results).transpose()
        df_results.columns = results_columns

        df_all_results = pd.concat([df_all_results, df_results], axis=0)

        df_all_results.to_csv(str(dir_results)+"/cnn_7_bands_{}.csv".format(dt_string))

        # Confusion matrix for the training data

        normalised_image_tiled_train, mask_tiled_2 = tile_data(norm_train_val, mask_original_train_val, SIZE_TILE)

        y_pred = model.predict(normalised_image_tiled_train, verbose=1)
        y_pred_flat = y_pred.ravel()
        y_pred_mask_flat = np.array([1 if x > 0.5 else 0 for x in y_pred_flat])

        y_mask = mask_tiled_2.ravel()

        # np.unique(y_pred_mask_flat, return_counts=True)
        # np.unique(y_mask, return_counts=True)

        plt.figure(figsize=(15, 15))
        cm = sklearn.metrics.confusion_matrix(y_true=y_mask, y_pred=y_pred_mask_flat)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt="g", ax=ax, cbar=False)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix Training/Validation Data")
        plt.savefig(
            str(dir_graphs) + "/cm_cnn_train_{}_half_{}.png".format(LOCATION_ID, half)
        )
        plt.show()

        tn, fp, fn, tp = cm.ravel()

        print("Metrics train data:")
        metrics = all_metrics(tp=tp, tn=tn, fp=fp, fn=fn)

        # Loading the model

        # BACKBONE = "efficientnetb3"
        # CLASSES = ["slum"]
        # LR = 0.001

        # n_classes = (
        #     1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
        # )  # case for binary and multiclass segmentation
        # activation = "sigmoid" if n_classes == 1 else "softmax"

        # N = normalised_image_tiled_test.shape[-1]

        # base_model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

        # inp = tf.keras.layers.Input(shape=(None, None, N))
        # l1 = tf.keras.layers.Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
        # out = base_model(l1)

        # model = tf.keras.Model(inp, out, name=base_model.name)

        # optim = keras.optimizers.Adam(LR)

        # dice_loss = sm.losses.DiceLoss()
        # focal_loss = (
        #     sm.losses.BinaryFocalLoss()
        #     if n_classes == 1
        #     else sm.losses.CategoricalFocalLoss()
        # )
        # total_loss = dice_loss + (1 * focal_loss)

        # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # model.compile(optim, total_loss, metrics)

        # model.load_weights(str(dir_models) + "/model_cnn_{}_half_{}.h5".format(LOCATION_ID,half))

        # Plotting the results

        if half == 0:
            y_pred = model.predict(normalised_image_tiled_test, verbose=1)
            reduced_shape = (image_original_test.shape[0]//SIZE_TILE*SIZE_TILE, image_original_test.shape[1]//SIZE_TILE*SIZE_TILE, 1)          
            y_pred_2D_array_reduced_half_0 = reshape_combine(
                y_pred.round(), reduced_shape
            )
            mask_reduced_half_0 = reshape_combine(mask_tiled_test, reduced_shape)
            pad_bottom = (
                image_original_test.shape[0] - y_pred_2D_array_reduced_half_0.shape[0]
            )
            pad_right = (
                image_original_test.shape[1] - y_pred_2D_array_reduced_half_0.shape[1]
            )
            y_pred_2D_array_half_0 = np.pad(
                y_pred_2D_array_reduced_half_0.squeeze(),
                ((0, pad_bottom), (0, pad_right)),
                "constant",
                constant_values=0,
            )
            mask_half_0 = np.pad(
                mask_reduced_half_0.squeeze(),
                ((0, pad_bottom), (0, pad_right)),
                "constant",
                constant_values=0,
            )

        if half == 1:
            y_pred = model.predict(normalised_image_tiled_test, verbose=1)
            reduced_shape = (image_original_test.shape[0]//SIZE_TILE*SIZE_TILE, image_original_test.shape[1]//SIZE_TILE*SIZE_TILE, 1)          
            y_pred_2D_array_reduced_half_1 = reshape_combine(
                y_pred.round(), reduced_shape
            )
            mask_reduced_half_1 = reshape_combine(mask_tiled_test, reduced_shape)
            pad_bottom = (
                image_original_test.shape[0] - y_pred_2D_array_reduced_half_0.shape[0]
            )
            pad_right = (
                image_original_test.shape[1] - y_pred_2D_array_reduced_half_0.shape[1]
            )
            y_pred_2D_array_half_1 = np.pad(
                y_pred_2D_array_reduced_half_1.squeeze(),
                ((0, pad_bottom), (0, pad_right)),
                "constant",
                constant_values=0,
            )
            mask_half_1 = np.pad(
                mask_reduced_half_1.squeeze(),
                ((0, pad_bottom), (0, pad_right)),
                "constant",
                constant_values=0,
            )

    y_pred_final = np.concatenate(
        (y_pred_2D_array_half_0, y_pred_2D_array_half_1), axis=1
    )
    mask_double_check = np.concatenate((mask_half_0, mask_half_1), axis=1)

    matplotlib.image.imsave(
        str(dir_rasters) + "/mask_{}.png".format(LOCATION_ID), 
        mask_double_check,
    )
    matplotlib.image.imsave(
        str(dir_rasters) + 
        "/pred_cnn_{}.png".format(LOCATION_ID), y_pred_final
    )

    print(
        "Finished location {}: {} out of {}".format(
            LOCATION_ID, i + 1, len(filtered_locations)
        )
    )