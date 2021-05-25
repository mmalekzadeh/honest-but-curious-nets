import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from .data import utk_face
from . import constants 
from torchvision.datasets import CelebA
from torchvision import transforms

def get_dataset(args):
    """
    Datasets    
    """ 
    if constants.DATASET == "celeba":
        
        save_dir = args.root_dir+"/data/temp/CelebA_npy_"+str(constants.IMAGE_SIZE)+"/"
        if constants.DATA_PREPROCESSING == 1:
            print("preprocess and save data.... ")
            root = args.root_dir+"/data"
            transform=transforms.Compose([
                transforms.Resize(size=(constants.IMAGE_SIZE,constants.IMAGE_SIZE)),                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])                
            ## For the first time 'download' should be set to True
            dataset_train = CelebA(root=root, split='train', target_type='attr',transform=transform, download=False) 
            dataset_valid = CelebA(root=root, split='valid', target_type='attr',transform=transform) 
            dataset_test  = CelebA(root=root, split='test', target_type='attr',transform=transform) 

            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = iter(data_loader).next()
            data_loader = DataLoader(dataset_valid, batch_size=len(dataset_valid))    
            valid_images, valid_labels = iter(data_loader).next()
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = iter(data_loader).next()
                        
            ## There are (almost) balanced attributes
            # attributes_dict = {'Attractive': 2, 'Black_Hair': 8, 'Blond_Hair': 9, 'Brown_Hair': 11,
            #         'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20,
            #         'Mouth_Slightly_Open': 21, 'Smiling': 31, 'Wavy_Hair': 33, 'Wearing_Lipstick': 36}
            # attributes_cols = list(attributes_dict.values())
            ####        
            ## Choose only those with Hair attributes
            print(len(train_images), len(train_labels))
            print(len(valid_images), len(valid_labels))
            print(len(test_images), len(test_labels))
            select = ((train_labels[:,8]==1) & (train_labels[:,9]==0) & (train_labels[:,11]==0) | \
                      (train_labels[:,8]==0) & (train_labels[:,9]==1) & (train_labels[:,11]==0) | \
                      (train_labels[:,8]==0) & (train_labels[:,9]==0) & (train_labels[:,11]==1))
            train_images = train_images[select]
            train_labels = train_labels[select]
            select = ((valid_labels[:,8]==1) & (valid_labels[:,9]==0) & (valid_labels[:,11]==0) | \
                      (valid_labels[:,8]==0) & (valid_labels[:,9]==1) & (valid_labels[:,11]==0) | \
                      (valid_labels[:,8]==0) & (valid_labels[:,9]==0) & (valid_labels[:,11]==1))
            valid_images = valid_images[select]
            valid_labels = valid_labels[select]
            select = ((test_labels[:,8]==1) & (test_labels[:,9]==0) & (test_labels[:,11]==0) | \
                      (test_labels[:,8]==0) & (test_labels[:,9]==1) & (test_labels[:,11]==0) | \
                      (test_labels[:,8]==0) & (test_labels[:,9]==0) & (test_labels[:,11]==1))
            test_images = test_images[select]
            test_labels = test_labels[select]                    
            ## Saving
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(constants.DATASET),
                                              train_images=train_images, train_labels=train_labels,
                                              valid_images=valid_images, valid_labels=valid_labels, 
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(valid_images), len(valid_labels))
            print(len(test_images), len(test_labels))                                              
            print("Completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(constants.DATASET)+".npz")
        train_images = np.array(dataset_npz['train_images'])
        train_labels = np.array(dataset_npz['train_labels']).astype(int)
        valid_images = np.array(dataset_npz['valid_images'])
        valid_labels = np.array(dataset_npz['valid_labels']).astype(int)
        test_images  = np.array(dataset_npz['test_images'])
        test_labels  = np.array(dataset_npz['test_labels']).astype(int)
        
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)

        return dataset_train, dataset_valid, dataset_test        

    #####################################   
    elif constants.DATASET == "utk_face":
        """
        Remeber to do remove these three images that are not properly labelled:
        39_1_20170116174525125.jpg.chip.jpg 
        61_1_20170109150557335.jpg.chip.jpg 
        61_1_20170109142408075.jpg.chip.jpg 
        """
        save_dir = args.root_dir+"/data/temp/UTKFace_npy_"+str(constants.IMAGE_SIZE)+"/"
        if constants.DATA_PREPROCESSING == 1:
            print("preprocess and save data.... ")
            dataset = utk_face.UTKFace(constants.IMAGE_SIZE)
            data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            images, labels_dict = iter(data_loader).next()
            labels = np.zeros((len(images),3)).astype(int)
            
            ## Combining Labels
            labels[:,0] = labels_dict['age'].numpy()
            labels[:,1] = labels_dict['gender'].numpy()
            labels[:,2] = labels_dict['race'].numpy()
            images = images.numpy()
            
            ## Shuffeling
            indices = np.random.RandomState(seed=constants.RANDOM_SEED).permutation(len(images))
            images = images[indices]
            labels = labels[indices]
            
            ## Saving
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            train_images = images[:constants.TRAIN_SHARE]
            train_labels = labels[:constants.TRAIN_SHARE]
            test_images = images[constants.TRAIN_SHARE:]
            test_labels = labels[constants.TRAIN_SHARE:]
            dataset_npz = np.savez_compressed(save_dir+str(constants.DATASET),
                                              train_images=train_images, train_labels=train_labels,
                                              test_images=test_images, test_labels=test_labels)
            print("Completed!")

        ## Loading        
        dataset_npz = np.load(save_dir+str(constants.DATASET)+".npz")
        train_images = np.array(dataset_npz['train_images'])
        train_labels = np.array(dataset_npz['train_labels'])
        test_images  = np.array(dataset_npz['test_images'])
        test_labels  = np.array(dataset_npz['test_labels'])
        
        dataset_train = (train_images, train_labels)
        dataset_test = (test_images, test_labels)

        return dataset_train, dataset_test

##############################################
def prepare_labels(train_labels, test_labels, valid_labels=None):
    if constants.DATASET == "celeba":         
        train_labels[:,0] = np.where(train_labels[:,8] == 1, 0, (np.where(train_labels[:,9] == 1, 1, 2)))
        train_labels = train_labels[:,[constants.HONEST,constants.CURIOUS]]
        valid_labels[:,0] = np.where(valid_labels[:,8] == 1, 0, (np.where(valid_labels[:,9] == 1, 1, 2)))
        valid_labels = valid_labels[:,[constants.HONEST,constants.CURIOUS]]
        test_labels[:,0] = np.where(test_labels[:,8] == 1, 0, (np.where(test_labels[:,9] == 1, 1, 2)))
        test_labels = test_labels[:,[constants.HONEST,constants.CURIOUS]]
        return train_labels, valid_labels, test_labels   
    #####            
    elif constants.DATASET == "utk_face":
        ####  Labels: age, gender, race        
        if  (constants.HONEST==0 and constants.K_Y == 2) or (constants.CURIOUS==0 and constants.K_S == 2):            
            train_labels[:, 0] = np.where(train_labels[:, 0] > 30, 1, 0)
            test_labels[:,  0] = np.where(test_labels[:,  0] > 30, 1, 0)                                                     
        elif (constants.HONEST==0 and constants.K_Y == 3) or (constants.CURIOUS==0 and constants.K_S == 3):
            train_labels[:, 0] = np.where(train_labels[:, 0] < 21, 2,
                                (np.where(train_labels[:, 0] < 36, 0, 1)))
            test_labels[:,  0] = np.where(test_labels[:,  0] < 21, 2,
                                (np.where(test_labels[:,  0] < 36, 0, 1)))                                                    
        elif (constants.HONEST==0 and constants.K_Y == 4) or (constants.CURIOUS==0 and constants.K_S == 4):
            train_labels[:, 0] = np.where(train_labels[:, 0] < 22, 0,
                                (np.where(train_labels[:, 0] < 30, 1, 
                                (np.where(train_labels[:, 0] < 46, 2, 3)))))
            test_labels[:,  0] = np.where(test_labels[:,  0] < 22, 0,
                                (np.where(test_labels[:,  0] < 30, 1, 
                                (np.where(test_labels[:,  0] < 46, 2, 3)))))                        
        elif (constants.HONEST==0 and constants.K_Y == 5) or (constants.CURIOUS==0 and constants.K_S == 5):
            train_labels[:, 0] = np.where(train_labels[:, 0] < 20, 0,
                                (np.where(train_labels[:, 0] < 27, 1, 
                                (np.where(train_labels[:, 0] < 35, 2, 
                                (np.where(train_labels[:, 0] < 50, 3, 4)))))))
            test_labels[:,  0] = np.where(test_labels[:,  0] < 20, 0,
                                (np.where(test_labels[:,  0] < 27, 1, 
                                (np.where(test_labels[:,  0] < 35, 2, 
                                (np.where(test_labels[:,  0] < 50, 3, 4)))))))
        else:
            if not (constants.HONEST==1 or constants.CURIOUS==1):
                raise ValueError("K_Y and K_S must be 2,3,4, or 5.")
        
            
        if  (constants.HONEST==2 and constants.K_Y == 2) or (constants.CURIOUS==2 and constants.K_S == 2):                                         
            train_labels[:, 2] = np.where(train_labels[:, 2] == 0, 0, 1)
            test_labels[:,  2] = np.where(test_labels[:,  2] == 0, 0, 1) 
        elif (constants.HONEST==2 and constants.K_Y == 3) or (constants.CURIOUS==2 and constants.K_S == 3):                                         
            train_labels[:, 2] = np.where(train_labels[:, 2] == 0, 0,
                                (np.where(train_labels[:, 2] == 2, 2, 1)))
            test_labels[:,  2] = np.where(test_labels[:,  2] == 0, 0,
                                (np.where(test_labels[:,  2] == 2, 2, 1))) 
        elif (constants.HONEST==2 and constants.K_Y == 4) or (constants.CURIOUS==2 and constants.K_S == 4):                                         
            train_labels[:, 2] = np.where(train_labels[:, 2] == 0, 0,
                                (np.where(train_labels[:, 2] == 1, 1, 
                                (np.where(train_labels[:, 2] == 3, 3, 2)))))
            test_labels[:,  2] = np.where(test_labels[:,  2] == 0, 0,
                                (np.where(test_labels[:,  2] == 1, 1, 
                                (np.where(test_labels[:,  2] == 3, 3, 2)))))        
        elif (constants.HONEST==2 and constants.K_Y == 5) or (constants.CURIOUS==2 and constants.K_S == 5):
            pass                                         
        else:
            if not (constants.HONEST==1 or constants.CURIOUS==1):
                raise ValueError("K_Y must be 2,3,4, or 5.")

        return train_labels, test_labels
