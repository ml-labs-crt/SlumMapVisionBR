# This script trains a random forest model with multispectral data as features for each location.
# Data is split in half and the model is trained on one half and tested on the other half (both halves).

import numpy as np
import os
import pandas as pd
import skimage.io
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.image

# import seaborn as sns
import joblib

# import rasterio
from pathlib import Path
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")

# dir_raw = Path("../results/granular_multi_halves/")
# dir_raw.mkdir(parents=True, exist_ok=True)
dir_graphs = Path("../results_" + dt_string + "_multispectral_halves/graphs/")
dir_graphs.mkdir(parents=True, exist_ok=True)
dir_models = Path("../results_" + dt_string + "_multispectral_halves/models/")
dir_models.mkdir(parents=True, exist_ok=True)
dir_rasters = Path("../results_" + dt_string + "_multispectral_halves/rasters/")
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


results_columns = [
    "location",
    "RF_min_split",
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

PATH_RASTERS = "../BrazilianDataset/intermediary_rasters/"
PATH_MASKS = "../BrazilianDataset/masks/"

assert os.path.exists(PATH_RASTERS)
assert os.path.exists(PATH_MASKS)

df_locations = pd.read_csv("info_locations_32.csv")

locations = df_locations["location"].astype(str).values.tolist()

for i in range(len(locations)):
    LOCATION_ID = locations[i]

    image = skimage.io.imread(PATH_RASTERS + LOCATION_ID + "/" + LOCATION_ID + ".tif")
    mask = skimage.io.imread(PATH_MASKS + "mask_" + LOCATION_ID + ".tif")

    plt.imshow(mask)

    image_channels = image.shape[-1]

    middle_image_y = int(image.shape[1] / 2)

    column_names = []
    for channel in range(1, image_channels + 1):
        TEXT = "band_" + str(channel)
        column_names.append(TEXT)

    for half in range(2):
        if half == 0:
            image_test = image[:, 0:middle_image_y, :]
            mask_test = mask[:, 0:middle_image_y]
            image_train_val = image[:, middle_image_y:, :]
            mask_train_val = mask[:, middle_image_y:]
        else:
            image_test = image[:, middle_image_y:, :]
            mask_test = mask[:, middle_image_y:]
            image_train_val = image[:, 0:middle_image_y, :]
            mask_train_val = mask[:, 0:middle_image_y]

        image_wide_test = image_test.ravel(order="C").reshape(-1, image_channels)
        mask_wide_test = mask_test.ravel(order="C")

        image_wide = image_train_val.ravel(order="C").reshape(-1, image_channels)
        mask_wide = mask_train_val.ravel(order="C")

        X_train = pd.DataFrame(image_wide, columns=column_names)
        y_train = pd.DataFrame(mask_wide, columns=["y_true"])

        X_test = pd.DataFrame(image_wide_test, columns=column_names)
        y_test = pd.DataFrame(mask_wide_test, columns=["y_true"])

        param_grid_initial = {"min_samples_leaf": [1, 10, 50, 100, 500, 1000]}

        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=100, random_state=0, n_jobs=4, class_weight="balanced"
        )

        CV_rfc = sklearn.model_selection.GridSearchCV(
            estimator=clf,
            param_grid=param_grid_initial,
            cv=2,
            verbose=1,
            n_jobs=-1,
            scoring="jaccard",
            refit=True,
        )

        CV_rfc.fit(X_train, y_train.values.ravel())

        min_samples_leaf = CV_rfc.best_params_["min_samples_leaf"]

        # if min_samples_leaf <= 10:
        #     param_grid = {"min_samples_leaf": [1, 2, 3, 5, 10]}
        # elif min_samples_leaf <= 100:
        #     param_grid = {"min_samples_leaf": [25, 50, 100]}
        # elif min_samples_leaf <= 500:
        #     param_grid = {"min_samples_leaf": [200, 350, 500]}
        # else:
        #     param_grid = {"min_samples_leaf": [1000, 2000, 3000, 4000]}

        # clf = sklearn.ensemble.RandomForestClassifier(
        #     n_estimators=100, random_state=0, n_jobs=-1, class_weight="balanced"
        # )

        # CV_rfc = sklearn.model_selection.GridSearchCV(
        #     estimator=clf,
        #     param_grid=param_grid,
        #     cv=2,
        #     verbose=1,
        #     n_jobs=-1,
        #     scoring="jaccard",
        #     refit=True,
        # )

        # CV_rfc.fit(X_train, y_train.values.ravel())

        # min_samples_leaf = CV_rfc.best_params_["min_samples_leaf"]

        joblib.dump(
            CV_rfc,
            str(dir_models) + "/" + "RF_{}_half_{}.joblib".format(LOCATION_ID, half),
        )

        y_pred = CV_rfc.predict(X_test)

        # X_train = X_train.copy()
        # X_train["split"] = "train"
        # X_train["y_pred"] = np.NaN
        # df_train = pd.concat([X_train, y_train], axis=1)

        # X_test = X_test.copy()
        # X_test["split"] = "test"
        # X_test["y_pred"] = y_pred
        # df_val = pd.concat([X_test, y_test], axis=1)

        # df = pd.concat([df_train, df_val], axis=0)

        # df.to_csv(
        #     "../results/granular_multi_halves/multispectral_{}_half_{}.csv".format(
        #         LOCATION_ID, half
        #     )
        # )

        y_true = y_test.values.ravel()

        # Uncomment if reading from files
        # df = pd.read_csv("../results/granular/multispectral_{}.csv".format(LOCATION_ID))
        # df = df[df['split']=='test']
        # y_true = df['y_true'].values.ravel()
        # y_pred = df['y_pred'].values.ravel()

        cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        # ax = plt.subplot()
        # sns.heatmap(cm, annot=True, fmt="g", ax=ax, cbar=False)
        # ax.set_xlabel("Predicted labels")
        # ax.set_ylabel("True labels")
        # ax.set_title("Confusion Matrix")
        # plt.savefig(
        #     "../results/graphs/cm_multi_halves/cm_multispectral_{}_half_{}.png".format(
        #         LOCATION_ID, half
        #     )
        # )
        # plt.show()

        tn, fp, fn, tp = cm.ravel()

        metrics = all_metrics(tp=tp, tn=tn, fp=fp, fn=fn)

        results = (
            [LOCATION_ID]
            + [min_samples_leaf]
            + [half]
            + list(metrics)
            + [tn, fp, fn, tp]
        )

        df_results = pd.DataFrame(results).transpose()
        df_results.columns = results_columns

        df_all_results = pd.concat([df_all_results, df_results], axis=0)

        df_all_results.to_csv(
            "../results_" + dt_string + "_multispectral_halves_per_half.csv"
        )

        if half == 0:
            half_0_array = y_pred.reshape(mask_test.shape)
            half_0_array = np.where(half_0_array > 0.5, 1, 0)

            half_0_array_mask = y_true.reshape(mask_test.shape)

        else:
            half_1_array = y_pred.reshape(mask_test.shape)
            half_1_array = np.where(half_1_array > 0.5, 1, 0)

            half_1_array_mask = y_true.reshape(mask_test.shape)

    y_pred_2D_all_image = np.concatenate((half_0_array, half_1_array), axis=1)
    y_mask_double_check = np.concatenate((half_0_array_mask, half_1_array_mask), axis=1)

    # plt.imshow(y_pred_2D_all_image)
    # plt.imshow(y_mask_double_check)

    matplotlib.image.imsave(
        str(dir_rasters) + "/" + "pred_multispectral_halves_{}.png".format(LOCATION_ID),
        y_pred_2D_all_image,
    )
    matplotlib.image.imsave(
        str(dir_rasters) + "/" + "mask_multispectral_halves_{}.png".format(LOCATION_ID),
        y_mask_double_check,
    )

    #     df_mean_location = (
    #         pd.concat([df_all_results_half_0, df_all_results_half_1], axis=0)
    #         .mean()
    #         .to_frame()
    #         .transpose()
    #     )
    #     df_mean_location.columns = results_columns
    #     df_mean_location["location"] = LOCATION_ID

    #     df_mean_results = pd.concat([df_mean_results, df_mean_location], axis=0)
    #     df_mean_results.to_csv("../results/multispectral_halves_mean_results.csv")

    print(
        "Finished location {}: {} out of {}".format(LOCATION_ID, i + 1, len(locations))
    )

    # # Plotting results as a raster

    # # Populating array with the classification results
    # pred_array = np.zeros((image.shape[0],image.shape[1]))
    # pred_array[:,:middle_image_y] = y_pred.reshape((-1,middle_image_y))

    # mask_double_check = np.zeros((image.shape[0],image.shape[1]))
    # mask_double_check[:,:middle_image_y] = y_true.reshape((-1,middle_image_y))

    # # All raster including the test and train
    # image_total = image.ravel(order='C').reshape(-1, image_channels)
    # mask_total = mask.ravel(order='C')

    # X_total = pd.DataFrame(image_total, columns=column_names)
    # y_total = pd.DataFrame(mask_total, columns=['y_true'])

    # y_pred_total = CV_rfc.predict(X_total)

    # pred_array = y_pred_total.reshape((image.shape[0],image.shape[1]))
    # mask_double_check = y_total.values.reshape((image.shape[0],image.shape[1]))

    # # Generate raster with results

    # pred_raster = 'multi_3_{}_half_{}.tif'.format(LOCATION_ID, half)
    # path_new_raster = "../results/rasters/" + pred_raster

    # with rasterio.open((PATH_RASTERS+LOCATION_ID+"/"+LOCATION_ID+".tif"), 'r') as image_raster:
    #     bands_array = image_raster.read()

    # with rasterio.open(path_new_raster,'w',
    # driver='GTiff',
    # height=bands_array.shape[1],
    # width=bands_array.shape[2],
    # count=1,
    # dtype=rasterio.float32,
    # crs=image_raster.crs,
    # transform=image_raster.transform
    # ) as new_dataset:
    #     new_dataset.write(pred_array.astype(rasterio.float32), 1)

    # # Generate raster with mask

    # mask_raster = 'mask_double_check_{}.tif'.format(LOCATION_ID)
    # path_new_raster = "../results/rasters/" + mask_raster

    # with rasterio.open(path_new_raster,'w',
    # driver='GTiff',
    # height=bands_array.shape[1],
    # width=bands_array.shape[2],
    # count=1,
    # dtype=rasterio.float32,
    # crs=image_raster.crs,
    # transform=image_raster.transform
    # ) as new_dataset:
    #     new_dataset.write(mask_double_check.astype(rasterio.float32), 1)

# # Aggregated results

# df_only_metrics = df_all_results[['kappa','precision','sensitivity',
# 'IoU Class 1','IoU Class 0','mIoU','accuracy','f1score']]

# df_only_metrics = df_mean_results[
#     [
#         "kappa",
#         "precision",
#         "sensitivity",
#         "IoU Class 1",
#         "IoU Class 0",
#         "mIoU",
#         "accuracy",
#         "f1score",
#     ]
# ]

# df_only_metrics.mean()
# df_only_metrics.std()
# df_only_metrics.min()
# df_only_metrics.max()

# df_agg = pd.concat(
#     [
#         df_only_metrics.mean(),
#         df_only_metrics.std(),
#         df_only_metrics.min(),
#         df_only_metrics.max(),
#     ],
#     axis=1,
# )

# df_agg.columns = ["mean", "std", "min", "max"]

# df_agg.to_csv("../results/multispectral_3_halves_agg.csv")

# df = df_mean_results[["IoU Class 1", "IoU Class 0"]].astype(float)

# boxplot = df.boxplot(
#     column=["IoU Class 1", "IoU Class 0"],
#     grid=False,
#     fontsize=15,
#     rot=0,
#     figsize=(10, 6),
# )
# # boxplot.set_title('IoU for each class', fontsize=15)
# boxplot.figure.savefig("../results/boxplot_IoU_multi_3_halves.png")

# Testing ravel function

# image = np.random.rand(3,3,4)
# mask = np.array([0,1,1,1,0,1,0,1,0]).reshape(3,3)

# image_channels = image.shape[-1]

# image_wide = image.ravel(order='C').reshape(-1,image_channels)
# mask_wide = mask.ravel(order='C')

# Perfect!
