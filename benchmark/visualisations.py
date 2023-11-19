# This script is used to visualise the results.

import pandas as pd
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df_116 = pd.read_csv("info_locations.csv")
df_32 = pd.read_csv("info_locations_32.csv")
df_capitals = pd.read_csv("info_capitals.csv")

dir_visualisations = Path("../visualisations")
dir_visualisations.mkdir(parents=True, exist_ok=True)

assert dir_visualisations.exists()


def removing_outliers_image(image):
    """
    Removes outliers from an input image by clipping pixel values to the 2nd and 98th percentiles of each color channel.

    Args:
        image (numpy.ndarray): Input image as a numpy array of shape (height, width, channels).

    Returns:
        numpy.ndarray: Output image as a numpy array of shape (height, width, channels) with pixel values clipped to the 2nd and 98th percentiles of each color channel.
    """
    image_BGR = image[:, :, 0:3]
    image_RGB_original = image_BGR[:, :, ::-1].copy()

    for band in range(3):
        percentile_2 = np.percentile(image_RGB_original[..., band], 2, axis=(0, 1))
        percentile_98 = np.percentile(image_RGB_original[..., band], 98, axis=(0, 1))
        image_RGB_original[image_RGB_original[:, :, band] < percentile_2] = percentile_2
        image_RGB_original[
            image_RGB_original[:, :, band] > percentile_98
        ] = percentile_98

    image_RGB_float = image_RGB_original.astype(np.float32)
    for band in range(3):
        image_RGB_float[..., band] = (
            image_RGB_float[..., band] - np.min(image_RGB_float[..., band])
        ) / (np.max(image_RGB_float[..., band]) - np.min(image_RGB_float[..., band]))

    return image_RGB_float


PATH_RASTERS = "../BrazilianDataset/intermediary_rasters/"
PATH_MASKS = "../BrazilianDataset/masks/"
PATH_PREDICTIONS = "../predictions/"

# df df_116 except locations mentioned in df_32 and df_capitals

df = df_116[~df_116["location"].isin(df_32["location"])]
df = df[~df["location"].isin(df_capitals["location"])]
df = df.reset_index(drop=True)

for i in range(len(df)):
    LOCATION_ID = str(df.iloc[i]["location"])
    NM_MUNICIPALITY = str.title(df.iloc[i]["NM_MUNICIP"])
    NAME_UF = df.iloc[i]["Name_UF"]
    XSIZE = df.iloc[i]["xsize"]
    YSIZE = df.iloc[i]["ysize"]
    PERCENTAGE_SLUMS = round(df.iloc[i]["proportion_slums"] * 100, 1)

    image = skimage.io.imread(PATH_RASTERS + LOCATION_ID + "/" + LOCATION_ID + ".tif")
    mask = skimage.io.imread(PATH_MASKS + "mask_" + LOCATION_ID + ".tif")
    prediction = skimage.io.imread(
        PATH_PREDICTIONS + "pred_multispectral_" + LOCATION_ID + ".png"
    )

    image_RGB = removing_outliers_image(image)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image_RGB)
    axs[0].axis("off")
    axs[0].set_title("Imagery")
    axs[1].imshow(mask)
    axs[1].axis("off")
    axs[1].set_title("Ground truth")
    axs[2].imshow(prediction)
    axs[2].axis("off")
    axs[2].set_title("Prediction \n(MS Random Sampling)")
    plt.suptitle(
        "{}, {}\n Location ID: {}\n {} x {} pixels\n Slums {}%".format(
            NM_MUNICIPALITY, NAME_UF, LOCATION_ID, XSIZE, YSIZE, PERCENTAGE_SLUMS
        )
    )
    plt.tight_layout()
    plt.savefig(
        str(dir_visualisations)
        + "/"
        + "image_ground_truth_prediction_MS_Random_Sampling_{}.png".format(LOCATION_ID),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
    )

# df df_32 except locations mentioned in df_capitals

df = df_32[~df_32["location"].isin(df_capitals["location"])]
df = df.reset_index(drop=True)

for i in range(len(df)):
    LOCATION_ID = str(df.iloc[i]["location"])
    NM_MUNICIPALITY = str.title(df.iloc[i]["NM_MUNICIP"])
    NAME_UF = df.iloc[i]["Name_UF"]
    XSIZE = df.iloc[i]["xsize"]
    YSIZE = df.iloc[i]["ysize"]
    PERCENTAGE_SLUMS = round(df.iloc[i]["proportion_slums"] * 100, 1)

    image = skimage.io.imread(PATH_RASTERS + LOCATION_ID + "/" + LOCATION_ID + ".tif")
    mask = skimage.io.imread(PATH_MASKS + "mask_" + LOCATION_ID + ".tif")
    prediction = skimage.io.imread(
        PATH_PREDICTIONS + "pred_multispectral_" + LOCATION_ID + ".png"
    )
    prediction_multi_halves = skimage.io.imread(
        PATH_PREDICTIONS + "pred_multispectral_halves_" + LOCATION_ID + ".png"
    )

    image_RGB = removing_outliers_image(image)

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(image_RGB)
    axs[0].axis("off")
    axs[0].set_title("Imagery")
    axs[1].imshow(mask)
    axs[1].axis("off")
    axs[1].set_title("Ground truth")
    axs[2].imshow(prediction)
    axs[2].axis("off")
    axs[2].set_title("Prediction \n(MS Random Sampling)")
    axs[3].imshow(prediction_multi_halves)
    axs[3].axis("off")
    axs[3].set_title("Prediction \n(MS \nwith Halves)")
    plt.suptitle(
        "{}, {}\n Location ID: {}\n {} x {} pixels\n Slums {}%".format(
            NM_MUNICIPALITY, NAME_UF, LOCATION_ID, XSIZE, YSIZE, PERCENTAGE_SLUMS
        )
    )
    plt.tight_layout()
    plt.savefig(
        str(dir_visualisations)
        + "/"
        + "image_ground_truth_prediction_multi_random_halves_{}.png".format(
            LOCATION_ID
        ),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
    )

# df df_capitals

df = df_capitals
df = df.reset_index(drop=True)

for i in range(len(df)):
    LOCATION_ID = str(df.iloc[i]["location"])
    NM_MUNICIPALITY = str.title(df.iloc[i]["NM_MUNICIP"])
    NAME_UF = df.iloc[i]["Name_UF"]
    XSIZE = df.iloc[i]["xsize"]
    YSIZE = df.iloc[i]["ysize"]
    PERCENTAGE_SLUMS = round(df.iloc[i]["proportion_slums"] * 100, 1)

    image = skimage.io.imread(PATH_RASTERS + LOCATION_ID + "/" + LOCATION_ID + ".tif")
    mask = skimage.io.imread(PATH_MASKS + "mask_" + LOCATION_ID + ".tif")
    prediction = skimage.io.imread(
        PATH_PREDICTIONS + "pred_multispectral_" + LOCATION_ID + ".png"
    )
    prediction_multi_halves = skimage.io.imread(
        PATH_PREDICTIONS + "pred_multispectral_halves_" + LOCATION_ID + ".png"
    )
    prediction_cnn_halves = skimage.io.imread(
        PATH_PREDICTIONS + "pred_cnn_" + LOCATION_ID + ".png"
    )

    image_RGB = removing_outliers_image(image)

    fig, axs = plt.subplots(1, 5)
    axs[0].imshow(image_RGB)
    axs[0].axis("off")
    axs[0].set_title("Imagery")
    axs[1].imshow(mask)
    axs[1].axis("off")
    axs[1].set_title("Ground truth")
    axs[2].imshow(prediction)
    axs[2].axis("off")
    axs[2].set_title("Prediction \n(MS Random \nSampling)")
    axs[3].imshow(prediction_multi_halves)
    axs[3].axis("off")
    axs[3].set_title("Prediction \n(MS \nwith Halves)")
    axs[4].imshow(prediction_cnn_halves)
    axs[4].axis("off")
    axs[4].set_title("Prediction \n(CNN \nwith Halves)")
    plt.suptitle(
        "{}, {}\n Location ID: {}\n {} x {} pixels\n Slums {}%".format(
            NM_MUNICIPALITY, NAME_UF, LOCATION_ID, XSIZE, YSIZE, PERCENTAGE_SLUMS
        )
    )
    plt.tight_layout()
    plt.savefig(
        str(dir_visualisations)
        + "/"
        + "image_ground_truth_prediction_multi_random_cnn_{}.png".format(LOCATION_ID),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
    )
