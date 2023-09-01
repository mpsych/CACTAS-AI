import os
import mahotas as mh
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


class Helper:
    
    def load(DATAPATH='/raid/mpsych/CACTAS/DATA/ESUS'):
        directory = DATAPATH

        data = []
        for filename in os.listdir(directory):
            if filename.endswith(".nrrd"):
                data.append(filename)
                
        return data

    def split(data):
        number_groups = {}

        for filename in data:
            number = filename.split(".")[0]
            if number not in number_groups:
                number_groups[number] = []
            number_groups[number].append(filename)
            
        #print(number_groups)

        random_numbers = list(number_groups.keys())

        split_ratio = 0.8
        split_index = int(len(random_numbers) * split_ratio)
        train_numbers = random_numbers[:split_index]
        test_numbers = random_numbers[split_index:]
        
        random.shuffle(train_numbers)
        random.shuffle(test_numbers)
        
        #print(train_numbers)
        #print(test_numbers)
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        train = []
        test = []

        for number in train_numbers:
            train.extend(number_groups[number])
        #print(train)
        
        for data in train:
            if 'img' in data:
                X_train.append(data)
            elif 'seg' in data:
                y_train.append(data)

        for number in test_numbers:
            test.extend(number_groups[number])
        #print(test)
        
        for data in test:
            if 'img' in data:
                X_test.append(data)
            elif 'seg' in data:
                y_test.append(data)

        #print(X_train)
        #print(y_train)
        #print(X_test)
        #print(y_test)

        return X_train, y_train, X_test, y_test
    
    def normalization(X_train, y_train, X_test, y_test):
        norm_X_train = []
        norm_y_train = []
        norm_X_test = []
        norm_y_test = []
        
        for file in X_train:
            data, header = nrrd.read('/raid/mpsych/CACTAS/DATA/ESUS/'+file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_train.append(normalized_data)
            
        for file in y_train:
            data, header = nrrd.read('/raid/mpsych/CACTAS/DATA/ESUS/'+file)
            norm_y_train.append(data)

        for file in X_test:
            data, header = nrrd.read('/raid/mpsych/CACTAS/DATA/ESUS/'+file)
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            norm_X_test.append(normalized_data)
            
        for file in y_test:
            data, header = nrrd.read('/raid/mpsych/CACTAS/DATA/ESUS/'+file)
            norm_y_test.append(data)

        
        return norm_X_train, norm_y_train, norm_X_test, norm_y_test
    
    
    
    
    
    
    
    