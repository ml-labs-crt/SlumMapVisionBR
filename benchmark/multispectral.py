# This script trains a random forest model with multispectral data (random sampling) as features for each location

import numpy as np
import os
import pandas as pd
import skimage.io
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.image
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")

# dir_raw = Path("../results_multispectral/granular/")
# dir_raw.mkdir(parents=True, exist_ok=True)
dir_graphs = Path("../results_" + dt_string + "_multispectral/graphs/")
dir_graphs.mkdir(parents=True, exist_ok=True)
dir_models = Path("../results_" + dt_string + "_multispectral/models/")
dir_models.mkdir(parents=True, exist_ok=True)
dir_rasters = Path("../results_" + dt_string + "_multispectral/rasters/")
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

results_columns = ['location','RF_min_split','kappa','precision',
'sensitivity','IoU Class 1','IoU Class 0','mIoU','accuracy', 'f1score', 'tn', 'fp', 'fn', 'tp']

df_all_results = pd.DataFrame(columns=results_columns)

PATH_RASTERS = "../BrazilianDataset/intermediary_rasters/"
PATH_MASKS = "../BrazilianDataset/masks/"

assert os.path.exists(PATH_RASTERS)
assert os.path.exists(PATH_MASKS)

df_locations = pd.read_csv("info_locations.csv")

locations = df_locations["location"].astype(str).values.tolist()

for i in range(0,len(locations)):

    LOCATION_ID = locations[i]

    image = skimage.io.imread(
        PATH_RASTERS + LOCATION_ID + "/" + LOCATION_ID + ".tif"
    )
    mask = skimage.io.imread(PATH_MASKS + "mask_" + LOCATION_ID + ".tif")

    # plt.imshow(mask)

    image_channels = image.shape[-1]

    image_wide = image.ravel(order='C').reshape(-1,image_channels)
    mask_wide = mask.ravel(order='C')

    column_names = []
    for channel in range(1,image_channels+1):
        text = "band_" + str(channel)
        column_names.append(text)

    X = pd.DataFrame(image_wide, columns = column_names)
    y = pd.DataFrame(mask_wide, columns = ['y_true'])

    X_train, X_val, \
    y_train, y_val = sklearn.model_selection\
            .train_test_split(X, y,\
            test_size=0.50, stratify=y,random_state=0)


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

    # Plot one tree of random forest

    # plt.figure(figsize=(20,20))
    # sklearn.tree.plot_tree(clf.estimators_[0], feature_names=column_names, filled=True)
    # plt.savefig(str(dir_graphs) + "/" + 'tree_multispectral_{}.png'
    # .format(LOCATION_ID))

    min_samples_leaf = CV_rfc.best_params_["min_samples_leaf"]

    joblib.dump(CV_rfc, str(dir_models) + "/" + "RF_{}.joblib"
    .format(LOCATION_ID))

    y_pred = CV_rfc.predict(X_val)
    y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])

    # X_train = X_train.copy()
    # X_train['split'] = 'train'
    # X_train['y_pred'] = np.NaN
    # df_train = pd.concat([X_train,y_train], axis=1)

    # X_val = X_val.copy()
    # X_val['split'] = 'test'
    # X_val['y_pred'] = y_pred
    # df_val = pd.concat([X_val,y_val], axis=1)

    # df = pd.concat([df_train,df_val], axis=0)

    # df.to_csv("../results/granular/multispectral_{}.csv"
    # .format(LOCATION_ID))

    y_true=y_val.values.ravel()

    # Uncomment if reading from files
    # df = pd.read_csv("../results/granular/multispectral_{}.csv".format(LOCATION_ID))
    # df = df[df['split']=='test']
    # y_true = df['y_true'].values.ravel()
    # y_pred = df['y_pred'].values.ravel()

    cm=sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, fmt='g', ax=ax);
    # ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    # ax.set_title('Confusion Matrix');
    # plt.savefig(str(dir_graphs) + "/" + 'cm_multispectral_{}.png'
    # .format(LOCATION_ID))
    # plt.show()

    tn, fp, fn, tp = cm.ravel()

    metrics = all_metrics(tp=tp, tn=tn, fp=fp, fn=fn)

    results = [LOCATION_ID] + [min_samples_leaf] + list(metrics) + [tn, fp, fn, tp]

    df_results = pd.DataFrame(results).transpose()
    df_results.columns = results_columns

    df_all_results = pd.concat([df_all_results,df_results], axis=0)

    df_all_results.to_csv("../results_" + dt_string + "_multispectral/multispectral.csv")

    # Testing model on all image
    
    X_all_image = pd.DataFrame(image_wide, columns = column_names)
    y_pred_all_image = CV_rfc.predict(X_all_image)

    y_pred_2D_all_image = y_pred_all_image.reshape(image.shape[0],image.shape[1])
    y_pred_2D_all_image = np.where(y_pred_2D_all_image > 0.5, 1, 0)

    y_mask_double_check = mask_wide.reshape(image.shape[0],image.shape[1])

    # plt.imshow(y_pred_2D_all_image, cmap='viridis')
    # np.unique(y_pred_2D_all_image)
    # plt.imshow(y_mask_double_check)

    matplotlib.image.imsave(str(dir_rasters) + "/" + "pred_multispectral_{}.png".format(LOCATION_ID), y_pred_2D_all_image)
    matplotlib.image.imsave(str(dir_rasters) + "/" + "mask_multispectral_{}.png".format(LOCATION_ID), y_mask_double_check)

    print('Finished location {}: {} out of {}'
    .format(LOCATION_ID,i+1,len(locations)))