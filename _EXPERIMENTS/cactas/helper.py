import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
import random
import nrrd
import pickle


class Helper:
    
    def load_data(DATAPATH='/raid/mpsych/CACTAS/DATA/ESUS'):
        directory = DATAPATH
        min_filesize_bytes = 15728640 #15MB
        
        images = []
        labels=[]

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) >= min_filesize_bytes:
                file = file_path.split("/")[6] 
                images.append(file)
            else:
                if filename.endswith(".b.seg.nrrd"):
                    file = file_path.split("/")[6]
                    labels.append(file)

        return images, labels

    def split_patients(images, labels):
        numbers = []
        for filename in images:
            number = filename.split(".")[0]
            numbers.append(number)
            
        label_dict = {}
        for filename in labels:
            num = filename.split(".")[0]
            label_dict[num] = filename

        data=[]
        for filename in images:
            data.append(tuple((filename,label_dict[filename.split(".")[0]])))
        
        random.shuffle(numbers)
        split_ratio = 0.8
        split_index = int(len(numbers) * split_ratio)
        train_numbers = numbers[:split_index]
        test_numbers = numbers[split_index:]
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # Sort the data based on the order of train_numbers and test_numbers
        data.sort(key=lambda x: (x[0].split(".")[0] in test_numbers, random.random()))

        for image, label in data:
            if image.split(".")[0] in train_numbers:
                X_train.append(image)
                y_train.append(label)
            elif image.split(".")[0] in test_numbers:
                X_test.append(image)
                y_test.append(label)

        return X_train, y_train, X_test, y_test
    

    def normalization(DATAPATH,X_train, y_train, X_test, y_test):
        norm_X_train = []
        norm_y_train = []
        norm_X_test = []
        norm_y_test = []
        
        for file in X_train:
            data, header = nrrd.read(DATAPATH + "/" +file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_train.append(normalized_data)
            
        for file in y_train:
            data, header = nrrd.read(DATAPATH + "/" + file)
            norm_y_train.append(data)

        for file in X_test:
            data, header = nrrd.read(DATAPATH + "/" + file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_test.append(normalized_data)
            
        for file in y_test:
            data, header = nrrd.read(DATAPATH + "/" + file)
            norm_y_test.append(data)

        
        return norm_X_train, norm_y_train, norm_X_test, norm_y_test
    
    def map_and_key(y_train):
        slice_to_patient_mapping = {}

        current_slice_index = 0
        remove_list = []

        for patient_id in range(len(y_train)):
            for slice_index in range(y_train[patient_id].shape[2]):
                if np.max(y_train[patient_id][:, :, slice_index]) != 0:
                    slice_to_patient_mapping[current_slice_index] = patient_id
                    current_slice_index += 1
        
        unique_items = set(slice_to_patient_mapping.values())

        unique_items_list = list(unique_items)
        a = len(unique_items_list)
        b = round(0.8 * a)
        
        target_value = b - 1

        last_key = None

        for key, value in reversed(slice_to_patient_mapping.items()):
            if value == target_value:
                last_key = key
                break
        
        return slice_to_patient_mapping, last_key
    
    def map_and_key_fulldata(y_train):
        slice_to_patient_mapping = {}

        current_slice_index = 0
        remove_list = []

        for patient_id in range(len(y_train)):
            for slice_index in range(y_train[patient_id].shape[2]):
                slice_to_patient_mapping[current_slice_index] = patient_id
                current_slice_index += 1

        unique_items = set(slice_to_patient_mapping.values())

        unique_items_list = list(unique_items)
        a = len(unique_items_list)
        b = round(0.8 * a)

        target_value = b - 1

        last_key = None
        

        for key, value in reversed(slice_to_patient_mapping.items()):
            if value == target_value:
                last_key = key
                break
                
        return slice_to_patient_mapping, last_key
    
    def extract_slices(X_train, y_train, X_test, y_test):
        slices = []
        for i in range(len(X_train)):
            for z in range(X_train[i].shape[2]):
                slice_2d = X_train[i][:, :, z]
                slices.append(slice_2d)
        X_train_array = np.array(slices)
        
        slices1 = []
        for i in range(len(y_train)):
            for z in range(y_train[i].shape[2]):
                slice_2d = y_train[i][:, :, z]
                slices1.append(slice_2d)
                
        new_slices = []
        for i in range(len(slices1)):
            # create a new array where all elements other than zero are replaced by True
            slices = np.where(slices1[i] != 0, True, False)
            new_slices.append(slices)
        y_train_array = np.array(new_slices)
        
        slices2 = []
        for i in range(len(X_test)):
            for z in range(X_test[i].shape[2]):
                slice_2d = X_test[i][:, :, z]
                slices2.append(slice_2d)
        X_test_array = np.array(slices2)  
        
        slices3 = []
        for i in range(len(y_test)):
            for z in range(y_test[i].shape[2]):
                slice_2d = y_test[i][:, :, z]
                slices3.append(slice_2d)  
        
        new_slice = []
        for i in range(len(slices3)):
            # create a new array where all elements other than zero are replaced by True
            slices = np.where(slices3[i] != 0, True, False)
            new_slice.append(slices)
        y_test_array = np.array(new_slice)
        
        #X_test_array = X_test_array.astype(np.float64)
        #y_test_array = y_test_array.astype(np.float64)

        X_train_array = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1],X_train_array.shape[2], 1)
        y_train_array = y_train_array.reshape(y_train_array.shape[0], y_train_array.shape[1],y_train_array.shape[2], 1)
        X_test_array = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1],X_test_array.shape[2], 1)
        y_test_array = y_test_array.reshape(y_test_array.shape[0], y_test_array.shape[1],y_test_array.shape[2], 1)
        
        #y_train_array = y_train.astype(bool)
        #y_test_array = y_train.astype(bool)

        print(X_train_array.shape, y_train_array.shape, X_test_array.shape, y_test_array.shape)
        
        return X_train_array, y_train_array, X_test_array, y_test_array
    
    def filter_slices(X_train, y_train):
        remove_list1 = []
        for i,d in enumerate(y_train):
            if d.max() == 0:
                remove_list1.append(i)

        images = []
        for index, element in enumerate(X_train):
            if index not in remove_list1:
                images.append(element)
        labels = []
        for index, element in enumerate(y_train):
            if index not in remove_list1:
                labels.append(element)
                
        X_train = np.array(images)
        y_train = np.array(labels)
        
        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        #X_test = X_test.astype(np.float64)
        #y_test = y_test.astype(np.float64)
        
        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print(X_train.shape, y_train.shape)
        
        return X_train, y_train
    
    
    def augment(X_train, y_train):
        from keras_unet.utils import get_augmented
        
        train_gen = get_augmented(
            X_train, y_train, batch_size=2,
            data_gen_args = dict(
                rotation_range=5.,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=40,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))

        return train_gen
    
    def augment_1(X_train, y_train):
        from keras_unet.utils import get_augmented

        train_gen = get_augmented(
            X_train, y_train, batch_size=2,
            data_gen_args = dict(
                rotation_range=45.,  # Increase rotation range
                width_shift_range=0.1,  # Increase shift range
                height_shift_range=0.1,  # Increase shift range
                shear_range=80,  # Increase shear range
                zoom_range=[0.2, 0.5],  # Increase zoom range
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))    
        return train_gen
    
    def augment_2(X_train, y_train):
        from keras_unet.utils import get_augmented

        train_gen = get_augmented(
            X_train, y_train, batch_size=2,
            data_gen_args = dict(
                rotation_range=90.,  
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))    
        return train_gen
    
    def create_unet(input_shape):
        from keras_unet.models import custom_unet
        model = custom_unet(
            input_shape=input_shape,
            use_batch_norm=False,
            num_classes=1,
            filters=64,
            dropout=0.2, 
            dropout_change_per_layer=0.0,
            num_layers=4,
            output_activation='sigmoid')

        from keras_unet.metrics import iou, iou_thresholded
        from tensorflow.keras.optimizers import Adam
        
        model.compile(optimizer = Adam(learning_rate=0.001),
              loss='binary_crossentropy', 
              metrics=[iou, iou_thresholded])

        return model

    def train_unet(X_train, y_train, X_val, y_val, model, epochs=200):
        batch_size = 32
        history = model.fit(#train_gen,
                            X_train,
                            y_train,
                            batch_size = batch_size,
                            #steps_per_epoch = len(X_train) // 2,
                            epochs=epochs,
                            validation_data=(X_val, y_val))
        
        #weights = model.save_weights('unet_weights_v1.h5')
        
        #with open('training_history_5.pkl', 'wb') as file:
        #    pickle.dump(history.history, file)
        
        return model, history
    
    def visualize_graph(history):
        from keras_unet.utils import plot_segm_history
        
        vis = plot_segm_history(history)

    
    def prediction(X_val, model):
        y_pred = model.predict(X_val)
        
        return y_pred
    
    def visualize_result(X_val, y_val, y_pred):
        from keras_unet.utils import plot_imgs

        plot_imgs(org_imgs=X_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=20)
    
    def evaluate(X_val, y_val, model):
        loss, iou, iou_thresholded = model.evaluate(X_val, y_val)
        
    
    
    
    
    
    
    
    
    
    
    