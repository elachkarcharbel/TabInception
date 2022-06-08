from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda, concatenate, BatchNormalization, AveragePooling2D
from tensorflow.keras import backend as K
from DataGenerator_224 import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *
from functools import partial
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

import glob
AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "X": tf.io.FixedLenFeature([50176], tf.float32),
            "Y": tf.io.FixedLenFeature([126], tf.float32),
        }
        if labeled
        else {"X": tf.io.FixedLenFeature([50176], tf.float32),}
    )
    example = tf.io.parse_example(example, tfrecord_format)
    input = example["X"]
    input = tf.cast(input, tf.float32)
 #   input = tf.reshape(input, [9,192])
 #   input = tf.transpose(input)
    input = tf.reshape(input, [224,224,1])

    if labeled:
        label = tf.cast(example["Y"], tf.float32)
        label = tf.reshape(label, [6,21])
        return input, label
    return input



def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed                                                                           
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files                                                                                                   
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order                                                                                 
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False                                                                
    return dataset


def get_dataset(filenames, batch,labeled=True):
    dataset = load_dataset(filenames, labeled=labeled).repeat()
    dataset = dataset.shuffle(batch)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch)
    return dataset


def get_dataset_validation(filenames, batch,labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch)
    return dataset



class TabCNN:
    
    def __init__(self, 
                 batch_size=128, 
                 epochs=16,
                 con_win_size = 9,
                 spec_repr="c",
                 data_path="../data/spec_repr/",
                 id_file="id.csv",
                 save_path="../model/saved/"):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        
        self.load_IDs()
        
        self.save_folder = self.save_path + self.spec_repr + "EfficientNetB0_tfrec_224x224_" + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["acc"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]
        
        if self.spec_repr == "c":
            self.input_shape = (224, 224, 1)
#        elif self.spec_repr == "m":
#            self.input_shape = (128, self.con_win_size, 1)
#        elif self.spec_repr == "cm":
#            self.input_shape = (320, self.con_win_size, 1)
#        elif self.spec_repr == "s":
#            self.input_shape = (1025, self.con_win_size, 1)
            
        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    
    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        
    
    def partition_data(self, data_split):
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["validation"] = []
        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["validation"].append(ID)
            else:
                self.partition["training"].append(ID)
                
        file_list_train=glob.glob("../data/tfrecords_224x224/0[0-2|4]*.tfrecords")
        print("Training files",len(file_list_train))
        self.training_generator=get_dataset(file_list_train, self.batch_size)
        self.nb_files_train=len(file_list_train)
        
        file_list_val=glob.glob("../data/tfrecords_224x224/05*.tfrecords")
        print("Validation files",len(file_list_val))
        self.validation_generator=get_dataset_validation(file_list_val, len(file_list_val))
        
        self.split_folder = self.save_folder + str(self.data_split) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)


          
    def log_model(self):
        with open(self.log_file,'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    
    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)
    
    
    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss
    
    
    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
           
    
    def build_model(self):
        
        model_input = Input(shape=self.input_shape)
        x = EfficientNetB0(include_top=True, weights=None, classes = self.num_classes * self.num_strings)(model_input)
        logits = Dense(units=(self.num_classes * self.num_strings))(x)
        logits = Reshape((self.num_strings, self.num_classes))(logits)
        model_output = Activation(self.softmax_by_string)(logits)

        model = Model(inputs=model_input, outputs = model_output)

        model.compile(loss=self.catcross_by_string,
                      optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=[self.avg_acc])
        
        self.model = model


    
    def train(self):
        self.model.fit(self.training_generator,
                    validation_data=None,
                    epochs=self.epochs,
                    steps_per_epoch = self.nb_files_train//self.batch_size,
                    verbose=1,
                    use_multiprocessing=True,
                    workers=9)
        
    
    def save_weights(self):
        self.model.save_weights(self.split_folder + "weights.h5")
        
    
    def test(self):
        self.validation_array = self.validation_generator.unbatch()
        self.X_test = np.array(list(self.validation_array.map(lambda x, y: x)))
        self.y_gt = np.array(list(self.validation_array.map(lambda x, y: y)))
        print("X_test shape:", self.X_test.shape)
        print("y_gt shape:", self.y_gt.shape)
        #self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test)
        print("y_pred shape:", self.y_pred.shape)
    
    def save_predictions(self):
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        self.metrics["acc"].append(float(self.model.evaluate(self.X_test, self.y_gt, batch_size=self.batch_size)[1]))
    
    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv") 
        
##################################
########### EXPERIMENT ###########
##################################

if __name__ == '__main__':
    tabcnn = TabCNN()
    tabcnn.build_model()
    tabcnn.log_model()

    for fold in range(1):
        print("\nfold " + str(fold))
        tabcnn.partition_data(fold)
        print("building model...")
        tabcnn.build_model()  
        print("training...")
        tabcnn.train()
        tabcnn.save_weights()
        print("testing...")
        tabcnn.test()
        tabcnn.save_predictions()
        print("evaluation...")
        tabcnn.evaluate()
    print("saving results...")
    tabcnn.save_results_csv()
