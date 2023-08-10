import os
import json
import sys
import cv2
import keras
import random
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
        
        df1 = pd.read_csv(config["train_path"])
        df2 = pd.read_csv(config["validation_path"])

    elif config["Imbalance_methods"] == "Virtual flaw":
        
        train_df1 = pd.read_csv(config["virtual_flaw_path"] + "fold_1.txt")
        train_df2 = pd.read_csv(config["virtual_flaw_path"] + "fold_2.txt")
        train_df3 = pd.read_csv(config["virtual_flaw_path"] + "fold_3.txt")
        train_df4 = pd.read_csv(config["virtual_flaw_path"] + "fold_4.txt")
        train_df5 = pd.read_csv(config["virtual_flaw_path"] + "fold_5.txt")

        df = pd.read_csv(config["train_path"])
        df1 = pd.concat([df, train_df1, train_df2, train_df3, train_df4, train_df5], ignore_index = True)
        df2 = pd.read_csv(config["validation_path"])

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

    X_valid = list()

    for num in tqdm(range(len(df2))):
        img_path = df2["path"][num]

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
        X_valid.append(img)    
    
    X_valid = np.array(X_valid, dtype = np.float32)
    y_valid = np.array(list(map(float, list(df2["ground_truth"]))), dtype = np.float32)

    X_train = tf.expand_dims(X_train, -1)
    y_train = tf.expand_dims(y_train, -1)
    X_valid = tf.expand_dims(X_valid, -1)
    y_valid = tf.expand_dims(y_valid, -1)

    return X_train, y_train, X_valid, y_valid

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

    #save_path = os.path.join(save_path, ("fold_" + str(config["fold_num"])))
    os.makedirs(save_path + "/train_result")
    os.makedirs(save_path + "/test_result")

def training_model(config, X_train, y_train, X_valid, y_valid):

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        save_path = config["save_path"] + "/" + config["Image_cropping_methods"] + "/" + config["Preprocessing_methods"] + "/" + config["Imbalance_methods"]
        
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

                setting_folder(cfg)
                X_train, y_train, X_valid, y_valid = load_train_data(cfg)
                training_model(cfg, X_train, y_train, X_valid, y_valid)
                
                tf.keras.backend.clear_session()
