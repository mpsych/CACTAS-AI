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
import mahotas as mh
from skimage.filters import threshold_otsu
import tensorflow as tf
import keras.backend as K


class Helper:
    
    def load_data(DATAPATH='/raid/mpsych/CACTAS/DATA/ESUS'):
        directory = DATAPATH
        min_filesize_bytes = 11000000 #11MB
        
        images = []
        labels=[]

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) >= min_filesize_bytes:
                file = file_path.split("/")[6] 
                images.append(file)
            else:
                if filename.endswith(".seg.nrrd"):
                    file = file_path.split("/")[6]
                    labels.append(file)

        return images, labels
    
    
    def load_datas(base_dir, folders=['ESUS', 'CEA', 'CAS'], min_filesize_bytes=11000000):
        images = []
        labels = []

        for folder in folders:
            directory = os.path.join(base_dir, folder)
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    if os.path.getsize(file_path) >= min_filesize_bytes:
                        images.append(f"{folder}/{filename}")
                    else:
                        labels.append(f"{folder}/{filename}")

        return images, labels
    
    
    def load_seg_data(DATAPATH='/raid/mpsych/CACTAS/DATA/CA'):
        seg=[]
        directory = DATAPATH
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file = file_path.split("/")[6] 
                seg.append(file)
        return seg
    
    
    def load_seg_datas(base_dir='/raid/mpsych/CACTAS/DATA', folders=['CA_ESUS3', 'CA_CEA5', 'CA_CAS5']):
        seg = []

        for folder in folders:
            directory = os.path.join(base_dir, folder)
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    seg.append(f"{folder}/{filename}")

        return seg

    
    
    def load_separate_data(DATAPATH='/raid/mpsych/CACTAS/DATA/CLEANED/', 
                           types=('images', 'labels', 'masks'), subsets=('train', 'test')):
        data = {}
        for data_type in types:
            for subset in subsets:
                folder_path = os.path.join(DATAPATH, data_type, subset)
                files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])

                data_list = [np.load(file) for file in files]

                data_list = [data.reshape(data.shape[0], data.shape[1], data.shape[2], 1) 
                             if data.ndim == 3 else data for data in data_list]

                key_name = f"{data_type}_{subset}"
                data[key_name] = data_list

        return data
    
    

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
    
    
    
    def split_patient_masks(images, labels, masks, test_files=['CEA/30.img.nrrd'], split_ratio=0.8):
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
            # Map the folder to its corresponding mask folder
            mask_folder = mask_folder_map.get(folder, folder)
            mask_dict[(mask_folder, number)] = mask

        for image in images:
            folder, filename = image.split("/")
            number = filename.split(".")[0]
            key = (folder, number)
            mask_key = (mask_folder_map.get(folder, folder), number)  # Use the mapped mask folder

            label = label_dict.get(key, None)
            mask = mask_dict.get(mask_key, None)

            if image in test_files:
                test_data.append((image, label, mask))
            else:
                data.append((image, label, mask))

            # Debugging output to ensure matching
            if mask is None:
                print(f"Warning: No mask found for {image}. Key used: {mask_key}")

        # Shuffle the data
        random.shuffle(data)

        # Split the data into training and testing sets
        split_index = int(len(data) * split_ratio)

        train_data = data[:split_index]
        test_data.extend(data[split_index:])

        # Separate images, labels, and masks for train and test
        X_train, y_train, m_train = zip(*train_data) if train_data else ([], [], [])
        X_test, y_test, m_test = zip(*test_data) if test_data else ([], [], [])

        return list(X_train), list(y_train), list(m_train), list(X_test), list(y_test), list(m_test)


    
    
    def split_patients_masks(images, labels, masks):
        numbers = [filename.split(".")[0] for filename in images]
        label_dict = {filename.split(".")[0]: filename for filename in labels}
        mask_dict = {filename.split(".")[0]: filename for filename in masks}

        data = [(filename, label_dict[filename.split(".")[0]], mask_dict[filename.split(".")[0]]) for filename in images]

        random.shuffle(numbers)
        split_ratio = 0.8
        split_index = int(len(numbers) * split_ratio)
        train_numbers = numbers[:split_index]
        test_numbers = numbers[split_index:]

        X_train = []
        y_train = []
        m_train = []
        X_test = []
        y_test = []
        m_test = []

        # Sort the data based on the order of train_numbers and test_numbers
        data.sort(key=lambda x: (x[0].split(".")[0] in test_numbers, random.random()))

        for image, label, mask in data:
            if image.split(".")[0] in train_numbers:
                X_train.append(image)
                y_train.append(label)
                m_train.append(mask)
            elif image.split(".")[0] in test_numbers:
                X_test.append(image)
                y_test.append(label)
                m_test.append(mask)

        return X_train, y_train, m_train, X_test, y_test, m_test


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
    
    
    def normalization2(DATAPATH, CAPATH, X_train, y_train, m_train, X_test, y_test, m_test):
        norm_X_train = []
        norm_y_train = []
        norm_X_test = []
        norm_y_test = []
        read_m_train = []
        read_m_test = []
        
        for file in X_train:
            data, header = nrrd.read(DATAPATH + "/" +file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_train.append(normalized_data)
            
        for file in y_train:
            data, header = nrrd.read(DATAPATH + "/" + file)
            norm_y_train.append(data)
            
        for file in m_train:
            data, header = nrrd.read(CAPATH + "/" + file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            read_m_train.append(normalized_data)

        for file in X_test:
            data, header = nrrd.read(DATAPATH + "/" + file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_test.append(normalized_data)
            
        for file in y_test:
            data, header = nrrd.read(DATAPATH + "/" + file)
            norm_y_test.append(data)
            
        for file in m_test:
            data, header = nrrd.read(CAPATH + "/" + file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            read_m_test.append(normalized_data)

        
        return norm_X_train, norm_y_train, read_m_train, norm_X_test, norm_y_test, read_m_test
    
    
    
    def normalization3(base_dir, capath_map, X_train, y_train, m_train, X_test, y_test, m_test):
        norm_X_train = []
        norm_y_train = []
        norm_X_test = []
        norm_y_test = []
        read_m_train = []
        read_m_test = []

        for file in X_train:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_train.append(normalized_data)

        for file in y_train:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            norm_y_train.append(data)

        for file in m_train:
            folder, filename = file.split("/", 1)
            capath = capath_map[folder]
            data, header = nrrd.read(os.path.join(capath, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            read_m_train.append(normalized_data)

        for file in X_test:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_test.append(normalized_data)

        for file in y_test:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            norm_y_test.append(data)

        for file in m_test:
            folder, filename = file.split("/", 1)
            capath = capath_map[folder]
            data, header = nrrd.read(os.path.join(capath, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            read_m_test.append(normalized_data)

        return norm_X_train, norm_y_train, read_m_train, norm_X_test, norm_y_test, read_m_test
    
    
    
    def normalize_images_and_labels(base_dir, X_train, y_train, X_test, y_test):
        norm_X_train = []
        norm_y_train = []
        norm_X_test = []
        norm_y_test = []

        for file in X_train:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_train.append(normalized_data)

        for file in y_train:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            norm_y_train.append(data)

        for file in X_test:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_test.append(normalized_data)

        for file in y_test:
            folder, filename = file.split("/", 1)
            data, header = nrrd.read(os.path.join(base_dir, folder, filename))
            norm_y_test.append(data)

        return norm_X_train, norm_y_train, norm_X_test, norm_y_test
    
    
    def normalize_masks(capath_map, m_train, m_test):
        read_m_train = []
        read_m_test = []

        for file in m_train:
            folder, filename = file.split("/", 1)
            capath = capath_map[folder]
            data, header = nrrd.read(os.path.join(capath, filename))
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            read_m_train.append(normalized_data)

        for file in m_test:
            if not file or "/" not in file:
                print(f"Invalid file path: {file}. Creating an empty mask.")
                empty_mask = np.zeros((512, 512, 58))
                read_m_test.append(empty_mask)
            else:
                folder, filename = file.split("/", 1)
                capath = capath_map[folder]
                data, header = nrrd.read(os.path.join(capath, filename))
                normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
                read_m_test.append(normalized_data)

        return read_m_train, read_m_test


    
    
    def normalize_data(data):
        normalized_data = []
        for array in data:
            array = array.astype(np.float32)
            min_val = array.min()
            max_val = array.max()

            array = (array - min_val) / (max_val - min_val)
            
            normalized_data.append(array)
        return normalized_data

    
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
        
        print(last_key)
        
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
        
        print(last_key)
        
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
        
        #y_train_array = y_train_array.astype(np.float64)
        #y_test_array = y_test_array.astype(np.float64)


        X_train_array = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1],X_train_array.shape[2], 1)
        y_train_array = y_train_array.reshape(y_train_array.shape[0], y_train_array.shape[1],y_train_array.shape[2], 1)
        X_test_array = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1],X_test_array.shape[2], 1)
        y_test_array = y_test_array.reshape(y_test_array.shape[0], y_test_array.shape[1],y_test_array.shape[2], 1)

        print(X_train_array.shape, y_train_array.shape, X_test_array.shape, y_test_array.shape)
        
        return X_train_array, y_train_array, X_test_array, y_test_array
    
    
    def extract_slices2(X_train, y_train, m_train, X_test, y_test, m_test):
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
        
        slices_mtrain=[]
        for i in range(len(m_train)):
            for z in range(m_train[i].shape[2]):
                slice_2d = m_train[i][:, :, z]
                slices_mtrain.append(slice_2d)
        m_train_array = np.array(slices_mtrain)
        
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
        
        slices_mtest=[]
        for i in range(len(m_test)):
            for z in range(m_test[i].shape[2]):
                slice_2d = m_test[i][:, :, z]
                slices_mtest.append(slice_2d)
        m_test_array = np.array(slices_mtest)
        
        #X_test_array = X_test_array.astype(np.float64)
        #y_test_array = y_test_array.astype(np.float64)

        X_train_array = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1],X_train_array.shape[2], 1)
        y_train_array = y_train_array.reshape(y_train_array.shape[0], y_train_array.shape[1],y_train_array.shape[2], 1)
        X_test_array = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1],X_test_array.shape[2], 1)
        y_test_array = y_test_array.reshape(y_test_array.shape[0], y_test_array.shape[1],y_test_array.shape[2], 1)
        
        m_train_array = m_train_array.reshape(m_train_array.shape[0], m_train_array.shape[1],m_train_array.shape[2], 1)
        m_test_array = m_test_array.reshape(m_test_array.shape[0], m_test_array.shape[1],m_test_array.shape[2], 1)
        
        #y_train_array = y_train.astype(np.float64)
        #y_test_array = y_train.astype(np.float64)

        print(X_train_array.shape, y_train_array.shape, m_train_array.shape, X_test_array.shape, y_test_array.shape, 
              m_test_array.shape)
        
        return X_train_array, y_train_array, m_train_array, X_test_array, y_test_array, m_test_array
    
    
    
    
    def extract_masks_slices(m_train, m_test):
        slices_mtrain=[]
        for i in range(len(m_train)):
            for z in range(m_train[i].shape[2]):
                slice_2d = m_train[i][:, :, z]
                dilated = mh.dilate(slice_2d.astype(np.bool_))
                for _ in range(9):
                    dilated = mh.dilate(dilated)
                slices_mtrain.append(dilated)
        m_train_array = np.array(slices_mtrain)
        
        slices_mtest=[]
        for i in range(len(m_test)):
            for z in range(m_test[i].shape[2]):
                slice_2d = m_test[i][:, :, z]
                dilated = mh.dilate(slice_2d.astype(np.bool_))
                for _ in range(9):
                    dilated = mh.dilate(dilated)
                slices_mtest.append(dilated)
        m_test_array = np.array(slices_mtest)
        
        m_train_array = m_train_array.astype(np.float64)
        m_test_array = m_test_array.astype(np.float64)
        
        m_train_array = m_train_array.reshape(m_train_array.shape[0], m_train_array.shape[1],m_train_array.shape[2], 1)
        m_test_array = m_test_array.reshape(m_test_array.shape[0], m_test_array.shape[1],m_test_array.shape[2], 1)
        
        
        print(m_train_array.shape, m_test_array.shape)
        
        return m_train_array, m_test_array
    
    
    def extract_CAmasks_slices(m_train, m_test):
        slices1 = []
        for i in range(len(m_train)):
            for z in range(m_train[i].shape[2]):
                slice_2d = m_train[i][:, :, z]
                slices1.append(slice_2d)
                
        new_slices = []
        for i in range(len(slices1)):
            slices = np.where(slices1[i] != 0, True, False)
            new_slices.append(slices)
        m_train_array = np.array(new_slices)
        
        
        slices3 = []
        for i in range(len(m_test)):
            for z in range(m_test[i].shape[2]):
                slice_2d = m_test[i][:, :, z]
                slices3.append(slice_2d)  
        
        new_slice = []
        for i in range(len(slices3)):
            slices = np.where(slices3[i] != 0, True, False)
            new_slice.append(slices)
        m_test_array = np.array(new_slice)

        m_train_array = m_train_array.reshape(m_train_array.shape[0], m_train_array.shape[1],m_train_array.shape[2], 1)
        m_test_array = m_test_array.reshape(m_test_array.shape[0], m_test_array.shape[1],m_test_array.shape[2], 1)

        print(m_train_array.shape, m_test_array.shape)
        
        return m_train_array, m_test_array
       
    
    def filter_slices(X_train, y_train, m_train):
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
        
        train_images = []
        for index, element in enumerate(m_train):
            if index not in remove_list1:
                train_images.append(element)
               
                
        X_train = np.array(images)
        y_train = np.array(labels)
        m_train = np.array(train_images)
        
        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        m_train = m_train.astype(np.float64)
        #X_test = X_test.astype(np.float64)
        #y_test = y_test.astype(np.float64)
        
        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print(X_train.shape, y_train.shape, m_train.shape)
        
        return X_train, y_train, m_train
    
    
    def masked_image(X_train, m_train, X_test, m_test):
        
        train_masks=[]
        for i in range(len(m_train)):
            binary = (m_train[i] > 0).astype(np.uint8)
            train_masks.append(binary)
            
        train_images=[]
        for i in range(len(X_train)):
            train_image = X_train[i] * train_masks[i]
            train_images.append(train_image)
        train_images_array = np.array(train_images) 
        
        train_images_array = train_images_array.astype(np.float32)
        train_images_array = train_images_array.reshape(train_images_array.shape[0],
                                                        train_images_array.shape[1],train_images_array.shape[2], 1)
        
        
        test_masks=[]
        for i in range(len(m_test)):
            binary = (m_test[i] > 0).astype(np.uint8)
            test_masks.append(binary)
            
        test_images=[]
        for i in range(len(X_test)):
            if i < 58:
                test_image = X_test[i]
            else:
                test_image = X_test[i] * test_masks[i]
            test_images.append(test_image)

        test_images_array = np.array(test_images)
        
        test_images_array = test_images_array.astype(np.float32)
        test_images_array = test_images_array.reshape(test_images_array.shape[0],
                                                      test_images_array.shape[1],test_images_array.shape[2], 1)
        
        print(train_images_array.shape, test_images_array.shape)
        
        return train_images_array, test_images_array
    
    
    
    
    def augment(X_train, y_train):
        from keras_unet.utils import get_augmented
        
        train_gen = get_augmented(
            X_train, y_train, batch_size=16,
            data_gen_args = dict(
                rotation_range=2.,
                width_shift_range=0.02,
                height_shift_range=0.02,
                shear_range=20,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))

        return train_gen
    
    def augment_1(X_train, y_train):
        from keras_unet.utils import get_augmented

        train_gen = get_augmented(
            X_train, y_train, batch_size=16,
            data_gen_args = dict(
                rotation_range=45.,  
                width_shift_range=0.1,  
                height_shift_range=0.1, 
                shear_range=80, 
                zoom_range=[0.2, 0.5], 
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))    
        return train_gen
    
    def augment_2(X_train, y_train):
        from keras_unet.utils import get_augmented

        train_gen = get_augmented(
            X_train, y_train, batch_size=16,
            data_gen_args = dict(
                rotation_range=90.,  
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))    
        return train_gen
    
    from sklearn.metrics import f1_score
   
    def sensitivity(y_true, y_pred):
        s = K.sum(y_true, axis=(1,2,3))
        y_true_c = s / (s + K.epsilon())
        s_ = K.sum(y_pred, axis=(1,2,3))
        y_pred_c = s_ / (s_ + K.epsilon())

        true_positives = K.sum(K.round(K.clip(y_true_c * y_pred_c, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_c, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(y_true, y_pred):
        s = K.sum(y_true, axis=(1,2,3))
        y_true_c = s / (s + K.epsilon())
        s_ = K.sum(y_pred, axis=(1,2,3))
        y_pred_c = s_ / (s_ + K.epsilon())

        true_negatives = K.sum(K.round(K.clip((1-y_true_c) * (1-y_pred_c), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true_c, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    def precision(y_true, y_pred):
        s = K.sum(y_true, axis=(1,2,3))
        y_true_c = s / (s + K.epsilon())
        s_ = K.sum(y_pred, axis=(1,2,3))
        y_pred_c = s_ / (s_ + K.epsilon())

        true_positives = K.sum(K.round(K.clip(y_true_c * y_pred_c, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_c, 0, 1)))
        
        return true_positives / (predicted_positives + K.epsilon())

    def f1(y_true, y_pred):
        prec = Helper.precision(y_true, y_pred).numpy()
        rec = Helper.sensitivity(y_true, y_pred).numpy()

        return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

    
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    
    def create_unet(input_shape):
        
        from keras_unet.models import custom_unet
        from keras_unet.metrics import iou, iou_thresholded
        from tensorflow.keras.optimizers import Adam
        
        model = custom_unet(
            input_shape=input_shape,
            use_batch_norm=True,
            num_classes=1,
            filters=64,
            dropout=0.2, 
            dropout_change_per_layer=0.0,
            num_layers=4,
            output_activation='sigmoid')


        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[iou, iou_thresholded])
        
        return model
    
    def create_swinUNet(input_shape):
        from keras_unet_collection import models, losses
        import torch

        model = models.swin_unet_2d(input_shape, filter_num_begin=64,
                               n_labels=1, depth=4, stack_num_down=4, stack_num_up=4,
                               patch_size=(4, 4), num_heads=[4, 8, 16, 16],
                               window_size=[4, 2, 2, 2], num_mlp=512, 
                               output_activation='Sigmoid', shift_window=True, name='swin_unet')
        
        from tensorflow.keras import optimizers
        from keras_unet.metrics import iou, iou_thresholded, dice_coef, F1Score
        from keras_unet_collection import losses
        
        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate = 0.001),
              metrics=[accuracy, iou, iou_thresholded])
        
        return model

    def train_unet(X_train, y_train, X_val, y_val, model, epochs=200):     
        batch_size = 16
        history = model.fit(X_train,
                            y_train,
                            batch_size=batch_size,
                            steps_per_epoch=len(X_train) // batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val))
        
        #weights = model.save_weights('unet_weights_v1.h5')
        
        #with open('training_history_5.pkl', 'wb') as file:
        #    pickle.dump(history.history, file)
        
        return model, history
    
    def train_swinUNet(train_gen, X_train, y_train, X_val, y_val, model, epochs=200):
        batch_size = 16
        history = model.fit(train_gen,
                            #X_train,
                            #y_train,
                            batch_size=batch_size,
                            steps_per_epoch=len(X_train) // batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val))
        
        return model, history
    
    def visualize_graph(history, save_path1='plot1.png', save_path2='plot2.png'):
        from keras_unet.utils import plot_segm_history
        
        vis = plot_segm_history(history)
        #import os
        #import matplotlib.pyplot as plt
        
        #vis = plot_segm_history(history)

        #save_dir = '/home/jiehyun.kim001/CACTAS/_EXPERIMENTS/Output/'
        #os.makedirs(save_dir, exist_ok=True)

        # Call plot_segm_history once, assuming it creates two plots.
        #plot_segm_history(history)

        # Get the current figures
        #fig1 = plt.figure(1)  # First figure
        #full_save_path1 = os.path.join(save_dir, save_path1)
        #fig1.savefig(full_save_path1, format='png')  # Save the first plot
        #plt.close(fig1)  # Close the first figure

        #fig2 = plt.figure(2)  # Second figure
        #full_save_path2 = os.path.join(save_dir, save_path2)
        #fig2.savefig(full_save_path2, format='png')  # Save the second plot
        #plt.close(fig2)  # Close the second figure

    
    def prediction(X_val, model):
        y_pred = model.predict(X_val)
             
        return y_pred
    
    def visualize_result(X_val, y_val, y_pred):
        from keras_unet.utils import plot_imgs

        plot_imgs(org_imgs=X_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=3)
        
    def visualize_result_limit(X_val, y_val, y_pred, start_index=100, end_index=110, save_path='prediction.png'):
        from keras_unet.utils import plot_imgs
        
        save_dir = '/home/jiehyun.kim001/CACTAS/_EXPERIMENTS/Output/'
        os.makedirs(save_dir, exist_ok=True)
        full_save_path = os.path.join(save_dir, save_path)

        plot_imgs(
            org_imgs=X_val[start_index:end_index],
            mask_imgs=y_val[start_index:end_index],
            pred_imgs=y_pred[start_index:end_index],
            nm_img_to_plot=end_index - start_index
        )
        
        # Save the figure to a file
        plt.savefig(full_save_path)
        plt.close()
    
    def evaluate(X_val, y_val, model):
        loss, iou, iou_thresholded = model.evaluate(X_val, y_val)
        
        #sens = Helper.sensitivity(X_val, y_val)
        #spec = Helper.specificity(X_val, y_val) 
        #prec = Helper.precision(X_val, y_val) 
        #f1 = Helper.f1(X_val, y_val)
        
        
        return loss, iou, iou_thresholded
    
    def threshold(y_pred,t_val = 0.5):
        a = y_pred
        a_binary = np.zeros(a.shape, dtype=np.bool_)
        a_binary[a > t_val] = True
        
        return a_binary

    
    
    
    
    
    
    
    
    
    
    