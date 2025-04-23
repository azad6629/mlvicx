import os
import pandas as pd
import numpy as np
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset
import torchvision.transforms as T
from .augmentations import VICRegAUgmentations

from .constants import *

class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
        
        
class NIHImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                  data_pct=0.01, imsize=224, task = NIH_TASKS):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(NIH_CXR_DATA_DIR):
            raise RuntimeError(f"{NIH_CXR_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(NIH_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(NIH_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        #get path
        self.df[NIH_PATH_COL] = self.df[NIH_PATH_COL].apply(lambda x: os.path.join(
                                    NIH_CXR_DATA_DIR, "/".join(x.split("/")[:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        self.path = self.df[NIH_PATH_COL].values
        self.labels = self.df.loc[:, task].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            timage = self.transform(image)
        
        # get labels
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)

        return timage, y

    def __len__(self):
        return len(self.df)    

    
class ChestX_ray14(BaseImageDataset):
    def __init__(self, split="train", transform=None, 
                 data_pct=0.01, imsize=224, task=NIH_TASKS):
        super().__init__(split=split, transform=transform)
        
        # Set path variables based on constants like in NIHImageDataset
        if not os.path.exists(CXR14_DATA_DIR):
            raise RuntimeError(f"{CXR14_DATA_DIR} does not exist!")
        
        self.split = split
        self.transform = transform
        self.imsize = imsize
        
        # Get module directory to find text files
        module_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine which dataset file to use based on split
        if split == "train":
            pathDatasetFile = os.path.join(module_dir, "Xray14_train_official.txt")
        elif split == "valid":
            pathDatasetFile = os.path.join(module_dir, "Xray14_val_official.txt")
        elif split == "test":
            pathDatasetFile = os.path.join(module_dir, "Xray14_test_official.txt")
        else:
            raise NotImplementedError(f"split {split} is not implemented!")
        
        # Check if file exists
        if not os.path.exists(pathDatasetFile):
            raise FileNotFoundError(f"Dataset file not found: {pathDatasetFile}")
        
        print(f"Loading dataset from: {pathDatasetFile}")
        
        # Read data from text file
        self.img_list = []
        self.labels = []
        
        with open(pathDatasetFile, "r") as fileDescriptor:
            for line in fileDescriptor:
                if line:
                    lineItems = line.split()
                    if len(lineItems) > 0:
                        # Use CXR14_DATA_DIR from constants
                        imagePath = os.path.join(CXR14_DATA_DIR, lineItems[0])
                        self.img_list.append(imagePath)
                        label = [float(l) for l in lineItems[1:]]
                        self.labels.append(label)
        
        self.labels = np.asarray(self.labels, dtype=np.float32)
        
        print(f"Loaded {len(self.img_list)} images")
        
        # Sample data if needed
        if data_pct != 1 and self.split == "train":
            print(f"Sampling {data_pct*100}% of the data...")
            indices = np.random.RandomState(42).choice(
                len(self.img_list), 
                int(len(self.img_list) * data_pct), 
                replace=False
            )
            self.img_list = [self.img_list[i] for i in indices]
            self.labels = self.labels[indices]
            print(f"After sampling: {len(self.img_list)} images")

    def __getitem__(self, index):
        # Get image
        img_path = self.img_list[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            timage = self.transform(image)
        
        # get labels
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)

        return timage, y
        
    def __len__(self):
        return len(self.img_list)    
    

class DataLoader():
    def __init__(self, config=None, model_name=None):
        self.config = config
        if model_name is not None:
            self.model_name      = model_name.lower() 
        
        self.tmode           = config['tmode']
        self.val_bs          = config['data']['down_vbs']        
        self.data_workers    = config['data']['data_workers']
        self.data_pct        = config['data']['data_pct']/100.0 
        self.imsize          = config['data']['resize_size']
            
        if config['data']['dataset'] == 'NIH14': 
            config['data']['task'] = NIH_TASKS
            self.task = NIH_TASKS
        if config['data']['dataset'] == 'Chex14': 
            config['data']['task'] = Chex14_TASKS
            self.task = Chex14_TASKS
                           
        if self.tmode == 'pre':
            self.train_bs = config['data']['pre_bs']
            self.train_transform = VICRegAUgmentations(self.config)
        if self.tmode == 'down':
            self.train_bs        = config['data']['down_tbs']
            self.train_transform = T.Compose([T.Resize((224, 224)), 
                                              T.RandomHorizontalFlip(),
                                              T.transforms.RandomGrayscale(p=0.2), 
                                              T.ToTensor(),           
                                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        
            
        self.valid_augmentations = T.Compose([T.Resize((224, 224)), 
                                              T.RandomHorizontalFlip(),
                                              T.ToTensor(),           
                                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])  

    
    def GetNihDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = NIHImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                                      imsize = self.imsize, 
                                      task = self.task
                                    
                               )
        valid_set = NIHImageDataset(split="valid",
                                 transform =valid_transform,
                                 imsize = self.imsize, 
                                 task = self.task
                               )
        
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
                
        return train_loader, valid_loader , valid_loader
    
    
    def GetChex14Dataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = ChestX_ray14(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                                      imsize = self.imsize, 
                                      task = self.task
                                    
                               )
        valid_set = ChestX_ray14(split="valid",
                                 transform =valid_transform,
                                 imsize = self.imsize, 
                                 task = self.task
                               )
        
        test_set = ChestX_ray14(split="test",
                                 transform =valid_transform,
                                 imsize = self.imsize, 
                                 task = self.task
                               )
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=1,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=False)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for test')
                
        return train_loader, valid_loader , test_loader        
        