import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

import models
from data import DataLoader
import yaml
import os
import numpy as np
import importlib
from optimizer import LARS
from utils import log, AverageMeter, collect_params
import wandb
import time
import math


class MLVICXTrainer():
    def __init__(self, model_name,config):
        self.model_name    = model_name
        self.config        = config
        self.gpu           = config['gpu']
        self.total_epochs  = config['optimizer']['total_epochs']
        self.warmup_epochs = config['optimizer']['warmup_epochs']
        
        batch_size = config['data']['pre_bs']
        data_ins   = DataLoader(config,self.model_name)
        dataset    = config['data']['dataset']
        
        if dataset == 'NIH14':
            self.train_loader, self.valid_loader, _ = data_ins.GetNihDataset()
        elif dataset == 'Chex14':
            self.train_loader, self.valid_loader, self.test_loader = data_ins.GetChex14Dataset()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
                
        num_examples = len(self.train_loader)*batch_size
        self.warmup_steps  = self.warmup_epochs * num_examples//batch_size    
        self.total_steps   = self.total_epochs * num_examples //batch_size

        self.resume_path = self.config['checkpoint']['resume_path']
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        save_path   = os.path.join('./ckpt',self.model_name.lower())
        self.method_name = f"{config['model']['backbone']['type']}_{dataset}_{batch_size}_{self.total_epochs}"
        self.config['checkpoint']['ckpt_path'] = os.path.join(save_path,self.method_name)        
        os.makedirs(config['checkpoint']['ckpt_path'], exist_ok=True)
        self.logger = log(path=config['checkpoint']['ckpt_path'], file=f"{self.method_name}.logs")
        
        """log tools in the running phase"""
        self.steps = 0
        self.total_training_time = 0
        self.log_step   = self.config['checkpoint']['log_step']
        self.save_epoch = self.config['checkpoint']['save_epoch']
   
        self.construct_model()
    
    @staticmethod
    def exclude_bias_and_norm(p):
        return p.ndim == 1

    
    def construct_model(self):
        self.logger.info('Training data Info:')
        self.logger.info(self.config)
        self.logger.info(f"dataset is {self.config['data']['dataset']}")        
        self.logger.info("init model!")
        
        model_module = importlib.import_module('models.' + self.model_name.lower())
        ModelClass   = getattr(model_module, self.model_name)
        self.model   = ModelClass(self.config).to(self.device)
        self.logger.info(self.model)
        
        self.logger.info("get optimizer!")
        weight_decay = float(self.config['optimizer']['weight_decay'])
        
        self.optimizer = LARS(self.model.parameters(),lr=0,weight_decay=weight_decay,
                         weight_decay_filter=self.exclude_bias_and_norm,lars_adaptation_filter=self.exclude_bias_and_norm,)

        
    def resume_model(self, resume):
        if resume:
            checkpoint_path = os.path.join(self.config['checkpoint']['ckpt_path'], f"{self.method_name}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.start_epoch = checkpoint['epoch']
                self.steps = checkpoint['steps']
                self.model.load_state_dict(checkpoint['model'], strict=True)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"--> Loaded checkpoint '{checkpoint_path}' (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            self.logger.info("--> No loaded checkpoint!")
        
    # save snapshots
    def save_checkpoint(self, epoch):
        if epoch % self.save_epoch == 0:
            model_state = {'config': self.config,
                           'epoch': epoch,
                           'steps': self.steps,
                           'model': self.model.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                     }
            online_state = {'online': self.model.encoder.state_dict()}
            SAVE_PATH1 = os.path.join(self.config['checkpoint']['ckpt_path'], f'{self.method_name}.pth')
            SAVE_PATH2 = os.path.join(self.config['checkpoint']['ckpt_path'], f'{self.method_name}_{epoch}.pth')
            torch.save(model_state, SAVE_PATH1)
            torch.save(online_state, SAVE_PATH2)
            
    def adjust_learning_rate(self, step):
        base_lr = self.config['optimizer']['base_lr'] * self.config['data']['pre_bs'] / 256
        if step < self.warmup_steps:
            lr = base_lr * step / self.warmup_steps
        else:
            step -= self.warmup_steps
#             self.total_steps -= self.warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        

    def recursive_to_device(self, inp, device):
        if isinstance(inp, list):
            return [self.recursive_to_device(item, device) for item in inp]
        elif isinstance(inp, torch.Tensor):
            return inp.to(device)
        else:
            return inp

    def train_epoch(self, epoch):
        loss_meter = AverageMeter()
        self.model.train()
        epoch_start_time = time.time() 
        for idx, (img,lbl) in enumerate(self.train_loader):
            self.adjust_learning_rate(self.steps)
            self.steps += 1
            img = self.recursive_to_device(img, self.device)
            loss = self.model(img)    
            self.optimizer.zero_grad()
            loss.backward()
            
            # Ensure gradient tensors are contiguous and have matching shapes
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.contiguous()
                    if param.grad.shape != param.shape:
                        param.grad = param.grad.reshape(param.shape)
                        
            self.optimizer.step()
            loss_meter.update(loss.item(), img[0].size(0))

            # Print log info
            if self.steps % self.log_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
#                 wandb.log({"Epoch": epoch, "Step": self.steps, "lr": lr, "Loss": loss_meter.val})
                self.logger.info(f'Epoch: [{epoch}][{idx}/{len(self.train_loader)}]\t'
                                f'Step {self.steps}\t'
                                f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t'
                                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t')
                
        epoch_end_time = time.time()  # End time of current epoch
        epoch_training_time = (epoch_end_time - epoch_start_time)/60
        self.total_training_time += epoch_training_time
        self.logger.info(f"Epoch {epoch} training time: {epoch_training_time:.2f} minutes")
        if epoch == self.total_epochs +1:
            self.total_training_time_hours = self.total_training_time/3600  
            self.logger.info(f"Total training time: {self.total_training_time_hours:.2f} hours")
