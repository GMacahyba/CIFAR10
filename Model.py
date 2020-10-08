import os
import numpy as np
import cv2, random
import argparse
from random import shuffle
from keras import models
from keras import layers
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import json

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir',
                        help = 'Directory which contains the data')
    parser.add_argument('output_dir',
                        help = 'Directory in which the training report will be stored')
    parser.add_argument('json_file',
                        help = 'JSON File path from which model hyperparameters will be retrieved')
    
    arguments = parser.parse_args()

    print("Data Directory:   {}".format(arguments.data_dir))
    print("Output Directory: {}".format(arguments.output_dir))
    print("JSON File;        {}".format(arguments.json_file))

    return arguments



#define one hot encoder function
def label_one_hot_encoder(img):
    img = img.split("_")[0]
    if img == "airplane":
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    elif img == "automobile":
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ]
    elif img == "bird":
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]
    elif img == "cat":
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    elif img == "deer":
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ]
    elif img == "dog":
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ]
    elif img == "frog":
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]
    elif img == "horse":
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ]
    elif img == "ship":
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]
    elif img == "truck":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]


#define function to process the data
def process_data(DATA_FOLDER, IMG_SIZE):
    data_df = []
    for img in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img/255.0
        filename = path.split('/')[-1]
        data_df.append([np.array(img), np.array(label_one_hot_encoder(filename))])
    shuffle(data_df)
    return data_df


def train_test_val_split(params, path):
    img_df = process_data(path, params['img_size'])

    #define train, test and validation df's
    train = img_df[0 : int(params['sample_size']*(1 - params['test_size'] - params['val_size']))]
    test = img_df[int(params['sample_size']*(1 - params['test_size'] - params['val_size'])) + 1: int(params['sample_size']*(1-params['val_size'])) + 1]
    val = img_df[int(params['sample_size']*(1-params['val_size'])) + 1 : -1]

    # #define X_train, y_train
    X_train = np.array([i[0] for i in train]).reshape(-1, params['img_size'], params['img_size'], 3)
    y_train = np.array([i[1] for i in train])

    # #define X_test, y_test
    X_test = np.array([i[0] for i in test]).reshape(-1, params['img_size'], params['img_size'], 3)
    y_test = np.array([i[1] for i in test])

    # #define X_val, y_val
    X_val = np.array([i[0] for i in val]).reshape(-1, params['img_size'], params['img_size'], 3)
    y_val = np.array([i[1] for i in val])

    return X_train, y_train, X_test, y_test, X_val, y_val 

def build_model(params):

    inputs = tf.keras.Input(shape = (params['img_size'], params['img_size'], params['input_channels']))


    conv_layers = []
    for conv_layer in params['network']['conv_layers']:
        if len(conv_layers) == 0:
            conv_layers.append(tf.keras.layers.Conv2D(filters = conv_layer['num_filters'], kernel_size = eval(conv_layer['kernel_size']), padding = conv_layer['padding'], activation = conv_layer['activation'])(inputs))
        else:
            conv_layers.append(tf.keras.layers.Conv2D(filters = conv_layer['num_filters'], kernel_size = eval(conv_layer['kernel_size']), padding = conv_layer['padding'], activation = conv_layer['activation'])(conv_layers[-1]))
            if (conv_layer['max_pool']) != 0:
                conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size = eval(conv_layer['max_pool']))(conv_layers[-1]))
            else:
                pass
            if conv_layer['batch_norm'] == 1:
                conv_layers.append(tf.keras.layers.BatchNormalization()(conv_layers[-1]))
            else:
                pass
    

    flat = tf.keras.layers.Flatten()(conv_layers[-1])

    dense_layers = []
    for dense_layer in params['network']['dense_layers']:
        if len(dense_layers) == 0:
            dense_layers.append(tf.keras.layers.Dense(dense_layer['size'], activation = dense_layer['activation'])(flat))
        else:
            dense_layers.append(tf.keras.layers.Dense(dense_layer['size'], activation = dense_layer['activation'])(dense_layers[-1]))
    
    drop = tf.keras.layers.Dropout(params['network']['dropout_rate'])(dense_layers[-1])

    output_layer = tf.keras.layers.Dense(params['num_classes'], activation = params['network']['output_activation'])(drop)

    model = tf.keras.Model(inputs = inputs, outputs = output_layer)
    
    model.compile(optimizer = params['network']['optimizer'], loss = params['network']['loss'], metrics = params['network']['metrics'])

    return model


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(X_test, y_test)
    
    return score[0], score[1]


if __name__ == '__main__':
    args = argument_parser()

    with open(args.json_file, 'r') as f:
        params = json.load(f)

    X, y, X_test, y_test, X_val, y_val = train_test_val_split(params, args.data_dir)
    model = build_model(params)


    reduce_lr = ReduceLROnPlateau(monitor = params['network']['reduce_lr']['monitor'], factor = params['network']['reduce_lr']['factor'],
                              patience = params['network']['reduce_lr']['patience'], min_lr = params['network']['reduce_lr']['min_lr'])
    train_model = model.fit(X, y, batch_size = params['batch_size'], epochs = params['epochs'], validation_data = (X_val, y_val), callbacks = [reduce_lr])

    test_loss, test_acc = evaluate_model(train_model, X_test, y_test)

    #initialize the dictionaries
    model_name = 'Convolutional Neural Network'

    dssplit_dict = {
    
        "Training": float(params['sample_size'] * (1 - params['val_size'] - params['test_size'])), 
        "Validation": float(params['val_size']), 
        "Test": float(params['val_size'])
    
    }
    metric_dict = {
    
        "Test Categorical CrossEntropy Loss" : float(test_loss), 
        "Test Accuracy" : float(test_acc) 
    
    }

    final_dict = {"Model Type" : model_name, 
        "Hyperparameters" : params, 
        "DataSetSplit" : dssplit_dict, 
        "Metrics" : metric_dict 
    
    }


    with open('Report_Part_2.json', 'w') as jsonfile:
            json.dump(final_dict, jsonfile, indent = 4, separators = (',', ':'))




