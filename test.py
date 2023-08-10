import os
import json
import sys
import cv2
import keras
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.applications import EfficientNetB5, ResNet50, VGG16
from keras.layers import Input
from keras.models import Model

def normalization(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def Seed_fix(My_Seed = 72):
    tf.keras.utils.set_random_seed(My_Seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(My_Seed)
    random.seed(My_Seed)

def normalization_and_histogram_equalization(img):
    img = normalization(img)
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    return img

def evaluate_data(Y_test, Y_pred):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(Y_pred)):
        if Y_test[i] == 0 and Y_pred[i] == 0:
            TN = TN + 1
        elif Y_test[i] == 0 and Y_pred[i] == 1:
            FP = FP + 1
        elif Y_test[i] == 1 and Y_pred[i] == 0:
            FN = FN + 1
        elif Y_test[i] == 1 and Y_pred[i] == 1:
            TP = TP + 1

    Recall = recall_score(Y_test, Y_pred)
    Precision = precision_score(Y_test, Y_pred)
    Accuracy = accuracy_score(Y_test, Y_pred)
    F1_Score = f1_score(Y_test, Y_pred)
    return TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision

def get_weld_image(config, img_path):

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    image = normalization(img)
    img = cv2.resize(img, (config["target_size"][0], config["target_size"][1]))
    image = cv2.resize(image, (512, 512))
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = np.array(image, dtype=np.float32)

    mean_list = []
    for i in range(config["target_size"][0]):
        mean_list.append(np.mean(image[:,i]))

    image = np.gradient(np.squeeze(mean_list))
    y1, y2 = config["target_size"][0] - int(np.argmax(image)* config["target_size"][0] / config["target_size"][0]), config["target_size"][0] - int(np.argmin(image) * config["target_size"][0] / config["target_size"][0]) 

    weld = (y1, y2)

    if config["Preprocessing_methods"] == "normalization":
        img = normalization(img)

    elif config["Preprocessing_methods"] == "normalization and histogram equalization":
        img = normalization_and_histogram_equalization(img)

    result = img[weld[0] - 15 : weld[1] + 15]
    h, w = result.shape
    width,height = config["target_size"][0], config["target_size"][1]
    dst = 255 * np.ones((height, width), dtype=np.uint8)
    roi = result[0:h, 0:w]
    dst[int((config["target_size"][0]/2)-h/2):int((config["target_size"][1])/2+h/2), 0:config["target_size"][0]] = roi

    return dst

def load_test_data(config):
    df = pd.read_csv(config["test_path"])

    X_test = list()
    test_path = list()

    for num in tqdm(range(len(df))):
        img_path = df["path"][num]
        test_path.append(img_path)

        if config["Image_cropping_methods"] == "Weld cropping":
            img = get_weld_image(config, img_path)
        
        elif config["Image_cropping_methods"] == "Non cropping":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
            if config["Preprocessing_methods"] == "normalization":
                img = normalization(img)

            elif config["Preprocessing_methods"] == "normalization and histogram equalization":
                img = normalization_and_histogram_equalization(img)
        
        img = cv2.resize(img, dsize = (config["target_size"][0], config["target_size"][1]), interpolation = cv2.INTER_CUBIC)
        img = img/255
        X_test.append(img)
    
    X_test = np.array(X_test, dtype = np.float32)
    y_test = np.array(list(map(float, list(df["ground_truth"]))), dtype = np.float32)

    return X_test, y_test, test_path

def test_result(config, X_test, y_test, test_path):
    save_path = config["save_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"]
    model_path = save_path + "/train_result/save_weight.h5"

    model = tf.keras.models.load_model(model_path)

    img_path_list = []
    y_predict_score = []
    ground_truth = []

    for num in range(len(X_test)):
        y_prediction = model.predict(X_test[num].reshape(-1, 512, 512))
        y_predict_score.append(y_prediction)
        ground_truth.append(y_test[num])
        img_path_list.append(test_path[num])
    
    for num in range(len(y_predict_score)):
        y_predict_score[num] = y_predict_score[num][0][0]
    
    model_df = pd.DataFrame({"path" : img_path_list,
                             "y_predict_score" : y_predict_score,
                             "ground_truth" : ground_truth})
    
    model_df.to_csv(save_path + "/test_result/test_result.csv")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    total_Accuracy = []
    total_F1_Score = []
    total_Recall = []
    total_Precision = []
    total_inspection_persent = []
    total_hit_ratio = []
    total_TN = []
    total_FP = []
    total_FN = []
    total_TP = []
    total_minus_inspection_present = []

    for threshold in thresholds:
        y_pred = []
        for i in range(len(model_df)):
            if model_df["y_predict_score"][i] <= threshold:
                y_pred.append(0)
            else:
                y_pred.append(1)
        y_test = list(model_df["ground_truth"])
        inspection_df = model_df.loc[threshold < model_df["y_predict_score"]]
        temp_df = model_df.drop(inspection_df.index, axis = 0)
        inspection_len = len(inspection_df)
        try:
            inspection_persent = (inspection_len/len(temp_df))
        except:
            inspection_persent = 0
        try:
            minus_inspection_persent = 1 - (inspection_len/len(temp_df))
        except:
            minus_inspection_persent = 1
        TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision = evaluate_data(y_test, y_pred)
        hit_ratio = TP/(FN + TP)
        total_Accuracy.append(Accuracy)
        total_F1_Score.append(F1_Score)
        total_Recall.append(Recall)
        total_Precision.append(Precision)
        total_inspection_persent.append(inspection_persent)
        total_hit_ratio.append(hit_ratio)
        total_TN.append(TN)
        total_FP.append(FP)
        total_FN.append(FN)
        total_TP.append(TP)
        total_minus_inspection_present.append(minus_inspection_persent)

    df = pd.DataFrame({"threshold" : thresholds,
                        "TN" : total_TN,
                        "FP" : total_FP,
                        "FN" : total_FN,
                        "TP" : total_TP,
                        "hit_ratio" : total_hit_ratio,
                        "재검률" : total_inspection_persent,
                        "Accuracy" : total_Accuracy,
                        "F1_Score" : total_F1_Score,
                        "Recall" : total_Recall,
                        "precision" : total_Precision})
    
    df.to_csv(save_path + "/test_result/hit_evaluation_result.csv", index = False)

    x = list(df["threshold"])
    y1 = list(df["hit_ratio"])
    y2 = total_minus_inspection_present

    fig, ax1 = plt.subplots()
    plt.title(config["model_name"])
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('hit_ratio')
    line1 = ax1.plot(x, y1, color = 'red', alpha = 0.5, label = "hit_ratio(%)", marker = "o")

    ax2 = ax1.twinx()
    ax2.set_ylabel('1 - inspection')
    line2 = ax2.plot(x, y2, color = 'blue', alpha = 0.5, label = "1 - inspection", marker = "o")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')

    plt.savefig(save_path + "/test_result/test_result_image.jpg")
        
    threshold_path = config["threshold_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"] + "/best_threshold.json"
    
    with open(threshold_path, "r") as file:
        data = json.load(file)
    
    threshold = 0.036674563

    y_pred = []
    for i in range(len(model_df)):
        if model_df["y_predict_score"][i] <= threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)
    
    TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision = evaluate_data(y_test, y_pred)

    result_json = {"TN" : TN,
                   "FP" : FP,
                   "FN" : FN,
                   "tp" : TP,
                   "Accuracy" : Accuracy,
                   "F1_Score" : F1_Score,
                   "Recall" : Recall,
                   "Precision" : Precision}
    
    save_json_path = save_path + "/test_result/result_using_AUROC_threshold.json"
    with open(save_json_path, "w") as f:
        json.dump(result_json, f)
    
if __name__ == "__main__":
    
    with open(sys.argv[1], "w") as f:
        cfg = json.load(f)

    image_cropping_methods = ["Weld cropping", "Non cropping"]
    preprocessing_methods = ["normalization", "normalization and histogram equalization"]
    imbalance_methods = ["Focal loss", "Virtual flaw", "None"]

    Seed_fix(My_Seed = cfg["SEED"])

    for crop_method in image_cropping_methods:

        cfg["Image_cropping_methods"] = crop_method

        for preprocess_method in preprocessing_methods:

            cfg["Preprocessing_methods"] = preprocess_method

            for imbalance_method in imbalance_methods:

                cfg["Imbalance_methods"] = imbalance_method

                X_test, y_test, test_path = load_test_data(cfg)
                test_result(cfg, X_test, y_test, test_path)
                
                tf.keras.backend.clear_session()
