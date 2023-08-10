import os
import sys
import cv2
import keras
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.applications import EfficientNetB5, ResNet50, VGG16
from keras.layers import Input
from keras.models import Model

def make_plot(history, save_path):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.legend()
    plt.xlabel('Epoch'); plt.ylabel('loss')
    plt.savefig(save_path + "/train_result/loss_graph.jpg")
    plt.close()

def Seed_fix(My_Seed = 72):
    tf.keras.utils.set_random_seed(My_Seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(My_Seed)
    random.seed(My_Seed)

def normalization(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def normalization_and_histogram_equalization(img):
    img = normalization(img)
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    return img

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

def load_train_data(config):
    
    if config["Imbalance_methods"] == "None" or config["Imbalance_methods"] == "Focal loss":
        
        fold_num_list = list(range(1, 6, 1))
        fold_num_list.remove(config["fold_num"])
        print("test fold : ", config["fold_num"])

        train_df1 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[0]) + ".txt")
        train_df2 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[1]) + ".txt")
        train_df3 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[2]) + ".txt")
        train_df4 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[3]) + ".txt")

        df1 = pd.concat([train_df1, train_df2, train_df3, train_df4], ignore_index = True)
        df2 = pd.read_csv(config["train_path"] + "fold_" + str(config["fold_num"]) + ".txt")

    elif config["Imbalance_methods"] == "Virtual flaw":

        fold_num_list = list(range(1, 6, 1))
        fold_num_list.remove(config["fold_num"])
        print("test fold : ", config["fold_num"])
        
        train_df1 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[0]) + ".txt")
        vf_df1 = pd.read_csv(config["virtual_flaw_path"] + "fold_" + str(fold_num_list[0]) + ".txt")
        train_df1 = pd.concat([train_df1, vf_df1], ignore_index = True)
        train_df2 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[1]) + ".txt")
        vf_df2 = pd.read_csv(config["virtual_flaw_path"] + "fold_" + str(fold_num_list[1]) + ".txt")
        train_df2 = pd.concat([train_df2, vf_df2], ignore_index = True)
        train_df3 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[2]) + ".txt")
        vf_df3 = pd.read_csv(config["virtual_flaw_path"] + "fold_" + str(fold_num_list[2]) + ".txt")
        train_df3 = pd.concat([train_df3, vf_df3], ignore_index = True)
        train_df4 = pd.read_csv(config["train_path"] + "fold_" + str(fold_num_list[3]) + ".txt")
        vf_df4 = pd.read_csv(config["virtual_flaw_path"] + "fold_" + str(fold_num_list[3]) + ".txt")
        train_df4 = pd.concat([train_df4, vf_df4], ignore_index = True)

        df1 = pd.concat([train_df1, train_df2, train_df3, train_df4], ignore_index = True)
        df2 = pd.read_csv(config["train_path"] + "fold_" + str(config["fold_num"]) + ".txt")

    X_train = list()

    for num in tqdm(range(len(df1))):
        img_path = df1["path"][num]

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
        X_train.append(img)
    
    X_train = np.array(X_train, dtype = np.float32)
    y_train = np.array(list(map(float, list(df1["ground_truth"]))), dtype = np.float32)

    X_test = list()
    test_path = list()

    for num in tqdm(range(len(df2))):
        img_path = df2["path"][num]
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
    y_test = np.array(list(map(float, list(df2["ground_truth"]))), dtype = np.float32)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, 
                                                            random_state = config["SEED"], stratify = y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, test_path

def setting_folder(config):
    
    try:
        save_path = os.path.join(config["save_path"], config["Image_cropping_methods"])
        os.makedirs(save_path)
        save_path = os.path.join(save_path, config["Preprocessing_methods"])
        os.makedirs(save_path)
        save_path = os.path.join(save_path, config["Imbalance_methods"])
        os.makedirs(save_path)
    
    except:
        save_path = os.path.join(config["save_path"], config["Image_cropping_methods"])
        save_path = os.path.join(save_path, config["Preprocessing_methods"])
        save_path = os.path.join(save_path, config["Imbalance_methods"])

    save_path = os.path.join(save_path, ("fold_" + str(config["fold_num"])))
    os.makedirs(save_path)
    os.makedirs(save_path + "/train_result")
    os.makedirs(save_path + "/test_result")

def training_model(config, X_train, y_train, X_valid, y_valid):

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        save_path = config["save_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"] + ("/fold_" + str(config["fold_num"]))
        
        if config["model_name"] == "CNN":
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(config["target_size"][0], config["target_size"][1], config["target_size"][2])),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        elif config["model_name"] == "VGG16":
            input_shape = Input(shape=(config["target_size"][0], config["target_size"][1], config["target_size"][2]))
            model = VGG16(weights = None, include_top = False, input_tensor = input_shape)
            layer_dict = dict([(layer.name, layer) for layer in model.layers])
            x = layer_dict['block5_pool'].output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
            model = Model(inputs = model.input, outputs = x)

        elif config["model_name"] == "ResNet50":
            input_shape = Input(shape=(config["target_size"][0], config["target_size"][1], config["target_size"][2]))
            model = ResNet50(weights = None, include_top = False, input_tensor = input_shape)
            layer_dict = dict([(layer.name, layer) for layer in model.layers])
            x = layer_dict['conv5_block3_out'].output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(512, activation = "relu")(x)
            x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
            model = Model(inputs = model.input, outputs = x)

        class CustomCallback(Callback):
            def on_train_begin(self, logs = None):
                raw_data = {'epoch' : [],
                            'train_loss' : [],
                            'train_accuracy' : [],
                            'validation_loss' : [],
                            'validation_accuracy': [],
                            }
                df = pd.DataFrame(raw_data)
                df.to_csv(save_path + "/train_result/train_log.csv", index = False)

            def on_epoch_end(self, epoch, logs=None):
                df = pd.read_csv(save_path + "/train_result/train_log.csv")
                df.loc[-1]=[epoch, logs["loss"], logs["binary_accuracy"], logs["val_loss"], logs["val_binary_accuracy"]]
                df.to_csv(save_path + "/train_result/train_log.csv", index = False)

        filename = (save_path + "/train_result/save_weight.h5")
        checkpoint = ModelCheckpoint(filename,
                                    monitor = 'val_loss',
                                    verbose = 1,
                                    save_best_only = True,
                                    mode = 'auto')

        earlystopping = EarlyStopping(monitor = 'val_loss', 
                                    patience = 15,
                                    )

        if config["Imbalance_methods"] == "Virtual flaw" or "None":
            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = config["learning_rate"], weight_decay = config["weight_decay"]),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = tf.keras.metrics.BinaryAccuracy()
                )
        
        elif config["Imbalance_methods"] == "Focal loss":
            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = config["learning_rate"], weight_decay = config["weight_decay"]),
                loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing = True, alpha = 0.9),
                metrics = tf.keras.metrics.BinaryAccuracy()
                )

        history = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = config["epoch"], 
                            batch_size = config["batch_size"], callbacks = [checkpoint, earlystopping, CustomCallback()])
        
        make_plot(history, save_path)
    
def training_result(config):

    result_dir = {
                "training_loss_평균" : list(),
                "training_loss_표준편차" : list(),
                "training_acc_평균" : list(),
                "training_acc_표준편차" : list(),
                "validation_loss_평균" : list(),
                "validation_loss_표준편차" : list(),
                "validation_acc_평균" : list(),
                "validation_acc_표준편차" : list()
                }
    
    training_loss = list()
    training_acc = list()
    valid_loss = list()
    valid_acc = list()

    for fold_num in range(1, 6, 1):
        
        save_path = config["save_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"]
        log_path = save_path + ("/fold_" + str(fold_num)) + "/train_result/train_log.csv"

        df = pd.read_csv(log_path)
        min_loss = min(df["validation_loss"])
        wanted_index = list(df[df["validation_loss"] == min_loss].index)
        result = df.iloc[wanted_index]
        training_loss.append(list(result["train_loss"]))
        training_acc.append(list(result["train_accuracy"]))
        valid_loss.append(list(result["validation_loss"]))
        valid_acc.append(list(result["validation_accuracy"]))
    
    result_dir["training_loss_평균"].append(np.mean(training_loss))
    result_dir["training_loss_표준편차"].append(np.std(training_loss))
    result_dir["training_acc_평균"].append(np.mean(training_acc))
    result_dir["training_acc_표준편차"].append(np.std(training_acc))
    result_dir["validation_loss_평균"].append(np.mean(valid_loss))
    result_dir["validation_loss_표준편차"].append(np.std(valid_loss))
    result_dir["validation_acc_평균"].append(np.mean(valid_acc))
    result_dir["validation_acc_표준편차"].append(np.std(valid_acc))

    df1 = pd.DataFrame(result_dir)
    df1.to_csv(save_path + "/cross_validation_result.csv", index = False)

def fold_test(config, X_test, y_test, path_list):
    save_path = config["save_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"] + ("/fold_" + str(config["fold_num"]))
    model_path = save_path + "/train_result/save_weight.h5"

    model = tf.keras.models.load_model(model_path)

    img_path_list = []
    y_predict_score = []
    ground_truth = []

    for num in range(len(X_test)):
        y_prediction = model.predict(X_test[num].reshape(-1, 512, 512))
        y_predict_score.append(y_prediction)
        ground_truth.append(y_test[num])
        img_path_list.append(path_list[num])
    
    for num in range(len(y_predict_score)):
        y_predict_score[num] = y_predict_score[num][0][0]
    
    model_df = pd.DataFrame({"path" : img_path_list,
                             "y_predict_score" : y_predict_score,
                             "ground_truth" : ground_truth})
    
    model_df.to_csv(save_path + "/test_result/test_result.csv")

def compute_roc_curve(config):
    fig1 = plt.figure(figsize=[12,12])
    ax1 = fig1.add_subplot(111, aspect = 'equal')

    tprs = []
    aucs = []

    mean_fpr = np.linspace(0,1,100)
    save_path = config["save_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"]

    for fold_num in range(1, 6, 1):
        csv_path = save_path + "/fold_" + str(fold_num) + "/test_result/test_result.csv"
        df = pd.read_csv(csv_path)
        y_pred = list(df["y_predict_score"])
        y_test = list(df["ground_truth"])

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 2, alpha = 0.3, label = "ROC fold %d (area = %0.2f)" % (fold_num, roc_auc))
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, "k--", label = "Mean ROC (area = %0.2f)" % mean_auc, lw = 2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc = "lower right")
    plt.savefig(save_path + "/roc_result.png")

    best_auc = 0.0
    best_threshold = 0.0
    best_fpr = 1.0

    for tpr, fpr, threshold in zip(mean_tpr, mean_fpr, thresholds):
        if tpr > best_auc or (tpr == best_auc and fpr < best_fpr):
            best_auc = tpr
            best_fpr = fpr
            best_threshold = threshold
    
    threshold_json = {"best_threshold" : best_threshold}
    json_path = save_path + "/best_threshold.json"
    with open(json_path, "w") as f:
        json.dump(threshold_json, f)


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

                for fold_num in range(1, 6, 1):

                    cfg["fold_num"] = fold_num
                    setting_folder(cfg)
                    X_train, y_train, X_valid, y_valid, X_test, y_test, test_path = load_train_data(cfg)
                    training_model(cfg, X_train, y_train, X_valid, y_valid)
                    fold_test(cfg, X_test, y_test, test_path)
                
                training_result(cfg)
                compute_roc_curve(cfg)
                
                tf.keras.backend.clear_session()