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
import mahotas as mh
from keras import backend as K
import hashlib

class LOO:

    def order(images, labels, masks):

        numbers = [filename.split(".")[0] for filename in images]
        label_dict = {filename.split(".")[0]: filename for filename in labels}
        mask_dict = {filename.split(".")[0]: filename for filename in masks}

        data = [(filename, label_dict[filename.split(".")[0]], mask_dict[filename.split(".")[0]]) for filename in images]

        X_train = []
        y_train = []
        m_train = []

        #data.sort(key=lambda x: (x[0].split(".")[0] in numbers, random.random()))
        data.sort(key=lambda x: x[0].split(".")[0] in numbers)

        for image, label, mask in data:
            if image.split(".")[0] in numbers:
                X_train.append(image)
                y_train.append(label)
                m_train.append(mask)

        return X_train, y_train, m_train
    
    def stable_shuffle_key(s):
        return hashlib.md5(s.encode()).hexdigest()
    
    def orders(images, labels, masks):
        data = []
        test_data = []

        # Map image folders to corresponding mask folders
        mask_folder_map = {
            'ESUS': 'CA_ESUS3',
            'CEA': 'CA_CEA5',
            'CAS': 'CA_CAS5'
        }

        # Build a dictionary to match labels and masks with corresponding images
        label_dict = {}
        mask_dict = {}

        for label in labels:
            folder, filename = label.split("/")
            number = filename.split(".")[0]
            label_dict[(folder, number)] = label

        for mask in masks:
            folder, filename = mask.split("/")
            number = filename.split(".")[0]
            mask_folder = mask_folder_map.get(folder, folder)
            mask_dict[(mask_folder, number)] = mask

        for image in images:
            folder, filename = image.split("/")
            number = filename.split(".")[0]
            key = (folder, number)
            mask_key = (mask_folder_map.get(folder, folder), number)

            label = label_dict.get(key, None)
            mask = mask_dict.get(mask_key, None)

            data.append((image, label, mask))

            # Debugging output to ensure matching
            if mask is None:
                print(f"Warning: No mask found for {image}. Key used: {mask_key}")

        data = sorted(data, key=lambda x: LOO.stable_shuffle_key(x[0]))
        
        images, labels, masks = zip(*data)
        
        return images, labels, masks

        
    
    def normalization(DATAPATH, CAPATH, images, labels, masks):
        img = []
        label = []
        mask = []
        
        # read and normalize data
        for file in images:
            data, header = nrrd.read(DATAPATH + "/" +file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            img.append(normalized_data)

        for file in labels:
            data, header = nrrd.read(DATAPATH + "/" + file)
            label.append(data)

        for file in masks:
            data, header = nrrd.read(CAPATH + "/" + file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            mask.append(normalized_data)

        slices_per_patient = [im.shape[2] for im in img]
        
        return img, label, mask, slices_per_patient
    
    def data_normalization(DATAPATH, CAPATHS, images, labels, masks):
        img = []
        label = []
        mask = []
        
        # read and normalize data
        for file in images:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(DATAPATH, folder, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            img.append(normalized_data)

        for file in labels:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(DATAPATH, folder, filename))
            label.append(data)

        for file in masks:
            if not file or "/" not in file:
                print(f"Invalid file path: {file}. Creating an empty mask.")
                empty_mask = np.zeros((512, 512, 58))
                mask.append(empty_mask)
            else:
                folder, filename = file.split("/", 1)
                capath = CAPATHS[folder]
                data, header = nrrd.read(os.path.join(capath, filename))
                normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
                mask.append(normalized_data)
        
        slices_per_patient = [im.shape[2] for im in img]
        
        return img, label, mask, slices_per_patient
    
    
    def extract_slices(img, label, mask):
        # Image
        slices = []
        for i in range(len(img)):
            for z in range(img[i].shape[2]):
                slice_2d = img[i][:, :, z]
                slices.append(slice_2d)
        X_train_array = np.array(slices)

        # Label
        slices1 = []
        for i in range(len(label)):
            for z in range(label[i].shape[2]):
                slice_2d = label[i][:, :, z]
                slices1.append(slice_2d)

        new_slices = []
        for i in range(len(slices1)):
            slices = np.where(slices1[i] != 0, True, False)
            new_slices.append(slices)
        y_train_array = np.array(new_slices)

        # Mask
        slices_mtrain=[]
        for i in range(len(mask)):
            for z in range(mask[i].shape[2]):
                slice_2d = mask[i][:, :, z]
                dilated = mh.dilate(slice_2d.astype(np.bool_))
                for _ in range(9):
                    dilated = mh.dilate(dilated)
                slices_mtrain.append(dilated)
        m_train_array = np.array(slices_mtrain)
        
        X_train_array = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1],X_train_array.shape[2], 1)
        y_train_array = y_train_array.reshape(y_train_array.shape[0], y_train_array.shape[1],y_train_array.shape[2], 1)
        m_train_array = m_train_array.reshape(m_train_array.shape[0], m_train_array.shape[1],m_train_array.shape[2], 1)
        
        X_train_array = X_train_array.astype(np.float32)
        m_train_array = m_train_array.astype(np.float32)
        
        print(X_train_array.shape, y_train_array.shape, m_train_array.shape)
        
        return X_train_array, y_train_array, m_train_array
    
    
    def mask_image(X_train_array, m_train_array):
        # Mask Image
        train_masks=[]
        for i in range(len(m_train_array)):
            binary = (m_train_array[i] > 0).astype(np.uint8)
            train_masks.append(binary)
            
        train_images=[]
        for i in range(len(X_train_array)):
            train_image = X_train_array[i] * train_masks[i]
            train_images.append(train_image)
        images_array = np.array(train_images) 
        
        images_array = images_array.reshape(images_array.shape[0],images_array.shape[1],images_array.shape[2], 1)
        print(images_array.shape)
        
        return images_array
    
    def split_set(train_index, test_index, images_array, y_train_array, cumulative_slices):
        
        print(train_index, test_index)
    
        train_slices = []
        y_train_slices = []
        test_slices = []
        y_test_slices = []

        for i in train_index:
            start_idx = cumulative_slices[i]
            end_idx = cumulative_slices[i + 1]
            train_slices.extend(images_array[start_idx:end_idx])
            y_train_slices.extend(y_train_array[start_idx:end_idx])

        for i in test_index:
            start_idx = cumulative_slices[i]
            end_idx = cumulative_slices[i + 1]
            test_slices.extend(images_array[start_idx:end_idx])
            y_test_slices.extend(y_train_array[start_idx:end_idx])

        print(len(train_slices), len(y_train_slices))
        print(len(test_slices), len(y_test_slices))



        num_train_patients = len(train_index)
        train_patients_idx, val_patients_idx = train_test_split(
            np.arange(num_train_patients), test_size=0.1 #, shuffle=False
        )

        print(train_patients_idx,val_patients_idx, test_index)


        # get validation set from training set
        val_slices = []
        y_val_slices = []

        for i in val_patients_idx:
            start_idx = cumulative_slices[train_index[i]]
            end_idx = cumulative_slices[train_index[i] + 1]
            val_slices.extend(images_array[start_idx:end_idx])
            y_val_slices.extend(y_train_array[start_idx:end_idx])



        # update training set w/o validation set
        filtered_train_slices = []
        filtered_y_train_slices = []

        for i in train_patients_idx:
            start_idx = cumulative_slices[train_index[i]]
            end_idx = cumulative_slices[train_index[i] + 1]
            filtered_train_slices.extend(images_array[start_idx:end_idx])
            filtered_y_train_slices.extend(y_train_array[start_idx:end_idx])



        X_train_array = np.array(filtered_train_slices)
        y_train_array = np.array(filtered_y_train_slices)
        X_val_array = np.array(val_slices)
        y_val_array = np.array(y_val_slices)
        X_test_array = np.array(test_slices)
        y_test_array = np.array(y_test_slices)

        print(X_train_array.shape, y_train_array.shape)
        print(X_val_array.shape, y_val_array.shape)
        print(X_test_array.shape, y_test_array.shape)

    
        return X_train_array, y_train_array, X_val_array, y_val_array, X_test_array, y_test_array






















    
