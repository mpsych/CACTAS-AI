from src.setupDataDir import *
from src.runModel import *
import ast
import os
import numpy

scan_folder = '../../SCAN_DATA/'

makeFolders()
makeTrainingPatches(scan_folder)
makeTestingPatches(scan_folder)

print('\nData preprocessed\n ')

print('Training Model')
trainlist = np.array(os.listdir("./Data/Training/gt"))
scanlist = [trainlist[volume].split('_')[0] + '_' + trainlist[volume].split('_')[1] + '_' for volume in range(0, len(trainlist))]
scanlist = np.unique(scanlist)

for scan in scanlist:
    print('Leaving ' + scan + ' out')
    LOO = trainlist[~np.char.startswith(trainlist, scan)]

    train_model(LOO)
    test_model(scan)
    tf.keras.backend.clear_session(free_memory=True)

