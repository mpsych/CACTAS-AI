import os
import nrrd
import numpy as np
import cv2
import random
import json

def makeFolders():
    path = r'./Data' 
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = r'./Data/Checkpoints' 
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = r'./Data/Training' 
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = r'./Data/Training/inputs' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Training/gt' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Testing' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Testing/inputs' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Testing/gt' 
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = r'./Data/Testing/Patch_Coords' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Results' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Results/Predictions' 
    if not os.path.exists(path):
        os.makedirs(path)

    path = r'./Data/Results/Metrics' 
    if not os.path.exists(path):
        os.makedirs(path)

def makeTrainingPatches(scan_folder):
    def random_warp(img, plane):
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        if plane == 'xy':
            theta = random.randint(0, 360)
        else:
            theta = 0
        rotation = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    
        if plane == 'xy': 
            scale_x = np.round(random.uniform(0.75, 1.25), 2)
            scale_y = np.round(random.uniform(0.75, 1.25), 2)
        else:
            scale_x = 1
            scale_y = 1
        scale = np.array([[scale_x, 0, 0],
                        [0, scale_y, 0],
                        [0, 0, 1]])
        scale[0,2] = -((scale[0,0] * w/2) - w/2 )
        scale[1,2] = -((scale[1,1] * h/2) - h/2 )
    
        
        if plane == 'xy':
            shear_x = np.round(random.uniform(-0.25, 0.25), 2)
            shear_y = np.round(random.uniform(-0.25, 0.25), 2)
        else:
            shear_x = np.round(random.uniform(-0.15, 0.15), 2)
            shear_y = np.round(random.uniform(-0.15, 0.15), 2)
    
        shear = np.array([[1, shear_x, 0],
                        [shear_y, 1, 0],
                        [0, 0, 1]])
        shear[0,2] = -shear[0,1] * w/2
        shear[1,2] = -shear[1,0] * h/2
    
        affine = np.matmul(np.matmul(rotation, scale), shear)
        return affine[:2]

    datasets = ['CAS', 'CEA', 'ESUS']
    for dataset in datasets:
        print('processing: ' + dataset)
        datalist = os.listdir(scan_folder + 'CA_' + dataset)
        datalist = [int(s.split('.')[0]) for s in datalist]
        print(datalist)

        for number in datalist:
    
            print(dataset, number)
            for augmentation in range(0, 3):
                ca, _ = nrrd.read(scan_folder + 'CA_' + dataset + '/' + str(number) + ".ca.seg.nrrd") # carotid artery mask
                dilation_size = 3
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))
                for i in range(0, ca.shape[2]):
                    ca[:,:,i] = cv2.dilate(ca[:,:,i], element)
                ca_copy = ca.copy()
                full_ct, _ =  nrrd.read(scan_folder + 'SCANS_' + dataset + '/' + str(number) + ".img.nrrd") # full ct scan
                masked = ca * full_ct
                masked = (masked-np.min(masked)) / (np.max(masked)-np.min(masked))
                gt, _ =  nrrd.read(scan_folder + 'SCANS_' + dataset + '/' + str(number) + ".seg.nrrd") # plaque gt
                gt[gt>1] = 1 # make all plaque segments 1 instead of a new number for each one
                
                if augmentation > 0:
                # skip augmentation if theres no plaque in patch
                    if np.sum(gt) == 0:
                        continue
        
                if augmentation > 0:
                        
                    # warp on x y plane
                    xy_transform = random_warp(gt[:,:,0], 'xy')
                    w, h = gt[:,:,0].shape[1], gt[:,:,0].shape[0]
                    
                    for i in range(0, gt.shape[2]):
                        gt[:,:,i] = cv2.warpAffine(gt[:,:,i], xy_transform, (w, h))
                        masked[:,:,i] = cv2.warpAffine(masked[:,:,i], xy_transform, (w, h))
                        ca[:,:,i] = cv2.warpAffine(ca[:,:,i], xy_transform, (w, h))
                    
                    # warp on x z plane
                    xz_transform = random_warp(gt[:,0,:], 'xz')
                    w, h = gt[:,0,:].shape[1], gt[:,0,:].shape[0]
                    
                    for i in range(0, gt.shape[1]):
                        gt[:,i,:] = cv2.warpAffine(gt[:,i,:], xz_transform, (w, h))
                        masked[:,i,:] = cv2.warpAffine(masked[:,i,:], xz_transform, (w, h))
                        ca[:,i,:] = cv2.warpAffine(ca[:,i,:], xz_transform, (w, h))
                
            
                current_volume = 1
                for i in range(0, ca.shape[2]):
                    if np.sum(ca[:,:,i]) > 0:
                        
                        contours, _ = cv2.findContours(ca[:,:,i], mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                        for c in contours:
                            try:
                                # compute the center of the contour
                                M = cv2.moments(c)
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
        
                                volume = masked[int(cY-32):int(cY+32), int(cX-32):int(cX+32), i:i+64]
                                volume_gt = gt[int(cY-32):int(cY+32), int(cX-32):int(cX+32), i:i+64]
        
                                if np.shape(volume) == (64, 64, 64):
                                    nrrd.write("./Data/Training/inputs/"+dataset+'_'+str(number)+"_volume_"+str(current_volume)+ "_aug_"+str(augmentation)+".nrrd", volume)
                                    nrrd.write("./Data/Training/gt/"+dataset+'_'+str(number)+"_volume_"+str(current_volume)+"_aug_"+str(augmentation)+".nrrd", volume_gt)
            
                                ca[int(cY-32):int(cY+32), int(cX-32):int(cX+32), i:i+64] = 0
                                current_volume += 1
                            except:
                                pass

def makeTestingPatches(scan_folder):

    datasets = ['CAS', 'CEA', 'ESUS']
    for dataset in datasets:
        print('processing: ' + dataset)
        datalist = os.listdir(scan_folder + 'CA_' + dataset)
        datalist = [int(s.split('.')[0]) for s in datalist]
        print(datalist)

        for number in datalist:
        
            ca, _ = nrrd.read(scan_folder + 'CA_' + dataset + '/' + str(number) + ".ca.seg.nrrd")
            dilation_size = 3
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))
            for i in range(0, ca.shape[2]):
                ca[:,:,i] = cv2.dilate(ca[:,:,i], element)
            ca_copy = ca.copy()
            full_ct, _ =  nrrd.read(scan_folder + 'SCANS_' + dataset + '/' + str(number) + ".img.nrrd")
            masked = ca * full_ct
            masked = (masked-np.min(masked)) / (np.max(masked)-np.min(masked))
            gt, _ =  nrrd.read(scan_folder + 'SCANS_' + dataset + '/' + str(number) + ".seg.nrrd")
            gt[gt>1] = 1 # make all plaque segments 1 instead of a new number for each one
        
            patch_coords = [gt.shape]
        
            current_volume = 1
            for i in range(0, ca.shape[2]):
                if np.sum(ca[:,:,i]) > 0:
                    
                    contours, _ = cv2.findContours(ca[:,:,i], mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        try:
                            # compute the center of the contour
                            M = cv2.moments(c)
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            volume = masked[int(cY-32):int(cY+32), int(cX-32):int(cX+32), i:i+64]
                            if np.shape(volume) == (64, 64, 64):
                                nrrd.write("./Data/Testing/inputs/"+dataset+'_'+str(number)+"_volume_"+str(current_volume)+".nrrd", volume)
        
                            ca[int(cY-32):int(cY+32), int(cX-32):int(cX+32), i:i+64] = 0
        
                            patch_coords.append([current_volume, int(cY-32), int(cY+32), int(cX-32), int(cX+32), i, i+64])
                            
                            current_volume += 1
                        except:
                            pass
            savepath = "./Data/Testing/Patch_Coords/"+dataset+'_' + str(number)+"_patch_coords.json"
            with open(savepath, 'w') as f:
                json.dump(patch_coords, f, indent=2) 

    
        for number in datalist:
            print(scan_folder + 'SCANS_' + dataset + '/' + str(number) + ".seg.nrrd")
            gt, _ =  nrrd.read(scan_folder + 'SCANS_' + dataset + '/' + str(number) + ".seg.nrrd")
            gt[gt>1] = 1 # make all plaque segments 1 instead of a new number for each one
            nrrd.write("./Data/Testing/gt/"+dataset+'_' +str(number)+".nrrd", gt)
        