import os
import time
import random
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models

from utils import log, computeAUROC
from barbar import Bar
from data import DataLoader
from data.constants import NIH_TASKS, Chex14_TASKS


def parse_args():
    parser = argparse.ArgumentParser(description='MLVICX Downstream Evaluation.')

    # Mode settings
    parser.add_argument('-tmode', default='down', choices=['pre', 'down'])
    parser.add_argument('-mode', default='ssl', choices=['ssl', 'sl'])
    parser.add_argument('-dmode', default='lp', choices=['lp', 'lf'])

    # Model settings
    parser.add_argument('-init', default='random', choices=['random', 'imagenet'])
    parser.add_argument('-model', default='mlvicx')
    parser.add_argument('-arch', default='resnet18')

    # Training hyperparameters
    parser.add_argument('-bs', default=128, type=int)
    parser.add_argument('-lr_min', default=0.000001, type=float, help='minimum learning rate')
    parser.add_argument('-wd', default=0.0, type=float)
    parser.add_argument('-epochs', default=300, type=int)
    parser.add_argument('-patience', default=10, type=int)

    # Dataset settings
    parser.add_argument('-data_per', default=1.0, type=float)
    parser.add_argument('-dataset', default='NIH14', choices=['NIH14', 'Chex14'])
    parser.add_argument('-pre_dataset', default='NIH14')
    parser.add_argument('-evaltask', default='NIH_TASKS', choices=['NIH_TASKS', 'Chex14_TASKS'])

    # Pretraining params
    parser.add_argument('-pre_bs', default=64, type=int)
    parser.add_argument('-pre_ep', default=300, type=int)
    parser.add_argument('-eval_epoch', default=None, type=int)
    parser.add_argument('-metric', default='auc', choices=['auc', 'f1'])

    # Misc
    parser.add_argument('-resume', default=False, action='store_true')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-gpu', default=0, type=int)
    parser.add_argument('-ver', default=None, type=str)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Classifier(nn.Module):
    def __init__(self, base_encoder, output_dim, proxy_weight=None, use_sigmoid=True):
        super(Classifier, self).__init__()
        self.model = base_encoder        
        self.n_inputs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        if proxy_weight:
            try:
                self._load_pretrained_weights(proxy_weight)
                print(f'Pre-trained weights loaded from {proxy_weight}')
            except Exception as e:
                print(f"Error loading weights: {str(e)}")
        
        if use_sigmoid:
            self.linear = nn.Sequential(
                nn.Linear(self.n_inputs, output_dim),
                nn.Sigmoid()
            )
        else:
            self.linear = nn.Linear(self.n_inputs, output_dim)

    def _load_pretrained_weights(self, proxy_weight):
        state_dict = {}
        length = len(self.model.state_dict())
        # Use weights_only=True to avoid security risks with pickle
        checkpoint = torch.load(proxy_weight, map_location='cpu', weights_only=True)
        
        if 'online_network' in checkpoint:
            checkpoint = checkpoint['model']['online_network']
            prefix = 'encoder.'
            for name, param in self.model.state_dict().items():
                if name in checkpoint:
                    state_dict[name] = checkpoint[name]
                elif prefix + name in checkpoint:
                    state_dict[name] = checkpoint[prefix + name]
        elif 'online' in checkpoint:
            for name, param in zip(self.model.state_dict(), list(checkpoint['online'].values())[:length]):
                state_dict[name] = param
        else:
            for name, param in zip(self.model.state_dict(), list(checkpoint.values())[:length]):
                state_dict[name] = param
                
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def count_parameters(model):
    """Count trainable parameters in millions"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def train(model, loader, criterion, optimizer, scheduler, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    for idx, (img, target) in enumerate(Bar(loader)):
        target = target.to(device)
        img = img.to(device)
        
        # Forward pass
        output = model(img)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"_Current LR: {current_lr:.6f}")
    return total_loss / len(loader)


def valid(model, loader, device, num_classes):
    """Validate model performance"""
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    
    with torch.no_grad():
        for idx, (img, target) in enumerate(Bar(loader)):
            target = target.to(device)
            img = img.to(device)
            
            output = model(img)
            
            outGT = torch.cat((outGT, target), 0)
            outPRED = torch.cat((outPRED, output), 0)
    
    # Calculate metrics
    metric_individual = computeAUROC(outGT, outPRED)
    metric_mean = np.array(metric_individual).mean()
    
    return metric_individual, metric_mean


def test(model, loader, device, num_classes):
    """Test model and return predictions and ground truth"""
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    
    with torch.no_grad():
        for idx, (img, target) in enumerate(Bar(loader)):
            target = target.to(device)
            img = img.to(device)
            
            output = model(img)
            
            outGT = torch.cat((outGT, target), 0)
            outPRED = torch.cat((outPRED, output), 0)
    
    # Calculate metrics
    metric_individual = computeAUROC(outGT, outPRED)
    metric_mean = np.array(metric_individual).mean()
    
    return metric_individual, metric_mean



def main():
    # Parse arguments
#     args = parser.parse_args()
    args = parse_args()
    print('Mode:', args.mode)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config_file_path = "./configs/resnet18/mlvicx.yaml"
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config based on args
    config['mode'] = args.mode
    config['tmode'] = args.tmode
    config['downstream_mode'] = args.dmode
    config['data']['data_pct'] = args.data_per
    config['data']['task'] = args.evaltask
    config['data']['dataset'] = args.dataset
    
    # Set up device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Determine class names and tasks based on dataset
    if args.dataset == 'NIH14':
        class_names = NIH_TASKS
        config['data']['task'] = class_names
        data_ins = DataLoader(config)
        train_loader, valid_loader, test_loader = data_ins.GetNihDataset()
        num_classes = len(class_names)
    elif args.dataset == 'Chex14':
        class_names = Chex14_TASKS
        config['data']['task'] = class_names
        data_ins = DataLoader(config)
        train_loader, valid_loader, test_loader = data_ins.GetChex14Dataset()
        num_classes = len(class_names)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Set up model based on training mode
    if args.mode == 'sl':
        # Supervised learning mode
        proxy_weight = None
        args.model = 'supervised'
        
        if args.init == 'random':
            base_encoder = models.__dict__[args.arch](weights=None)
        elif args.init == 'imagenet':
            if args.arch == 'resnet18':
                base_encoder = models.__dict__[args.arch](weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif args.arch == 'resnet50':
                base_encoder = models.__dict__[args.arch](weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                base_encoder = models.__dict__[args.arch](weights=models.ResNet18_Weights.IMAGENET1K_V1)
                
        save_path = os.path.join('./ckpt', 'supervised', args.init, args.dataset)
        proxy_dir = 'supervised'
        
    elif args.mode == 'ssl':
        # Self-supervised learning mode
        base_encoder = models.__dict__[args.arch](weights=None)
        
        # Set up paths for pretrained model
        method_name = f'{args.model.lower()}{args.ver if args.ver else ""}'
        save_path = os.path.join('./ckpt', method_name)
        
        # Use command line arguments for pretraining settings
        pre_bs = args.pre_bs
        pre_ep = args.pre_ep
        
        model_dir_pattern = f'{args.arch}_{args.pre_dataset}_{pre_bs}_{pre_ep}'
        
        # Find the directory containing the pretrained model
        try:
            proxy_dir = next((item for item in os.listdir(save_path) 
                             if model_dir_pattern in item), None)
            
            if proxy_dir is None:
                raise FileNotFoundError(f"No pretrained model directory matching pattern: {model_dir_pattern}")
                
            print('Found pretrained model directory:', proxy_dir)
            
            # Determine path to pretrained weights
            if args.eval_epoch is not None:
                proxy_weight = os.path.join(save_path, proxy_dir, f'{proxy_dir}_{args.eval_epoch}.pth')
            else:
                proxy_weight = os.path.join(save_path, proxy_dir, f'{proxy_dir}.pth')
                
        except Exception as e:
            print(f"Error finding pretrained model: {str(e)}")
            print("Continuing with randomly initialized model...")
            proxy_dir = model_dir_pattern
            proxy_weight = None
    else:
        raise ValueError("Invalid mode. Select either 'sl' or 'ssl'")
    
    # Setup checkpoint directory
    ckpt_path = os.path.join(save_path, proxy_dir, f"{args.evaltask}_downstream", 
                            str(args.data_per), args.dmode)
    config['checkpoint']['ckpt_path'] = ckpt_path
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Setup method name for saving
    method_name = f"{proxy_dir}_{args.data_per}_{args.dmode}"
    if args.eval_epoch is not None:
        method_name += f"_{args.eval_epoch}"
    
    print('Checkpoints will be saved at:', ckpt_path)
    
    # Initialize logger
    logger = log(path=ckpt_path, file=f"{method_name}.logs")
    
    # Create classifier model
    criterion = torch.nn.BCELoss()
    classifier = Classifier(base_encoder, num_classes, proxy_weight, use_sigmoid=True)
    
    if args.dmode == 'lp':  # Linear probing - freeze backbone
        lr = 0.003
        for param in classifier.model.parameters():
            param.requires_grad = False
            
    if args.dmode == 'lf':  #finetune
        lr = 0.0003
    
    # Count trainable parameters
    num_params = count_parameters(classifier)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
                        classifier.parameters(),
                        lr=lr,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=args.wd  
                    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=10,    # Restart every 10 epochs
                    T_mult=2,  # Double the restart interval after each restart
                    eta_min=args.lr_min
                )
    # Move model to device
    classifier = classifier.to(device)
    
    # Log model information
    logger.info("Initializing model!")
    logger.info(f"Configuration: {config}")
    logger.info(f"Total trainable parameters: {num_params:.2f}M")
    logger.info(f"Model architecture: {classifier}")
    
    # Training loop with early stopping
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss = train(classifier, train_loader, criterion, optimizer, scheduler, device)
        
        # Validate
        metric_individual, metric_mean = valid(classifier, valid_loader, device, num_classes)
        
        # Log progress
        metric_name = 'AUC' if args.metric == 'auc' else 'F1'
        logger.info(f'Epoch: [{epoch}]\t'
                   f'Train Loss: {train_loss:.5f}\t'
                   f'Valid {metric_name}: {metric_mean:.4f}\t')
        
        # Save model state
        model_state = {
            'config': config,
            'epoch': epoch,
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'model': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # Check for improvement
        if metric_mean > best_metric:
            logger.info(f'{metric_name} increased ({best_metric:.4f} --> {metric_mean:.4f}). Saving model...')
            torch.save(model_state, os.path.join(ckpt_path, f'{method_name}.pth'))
            best_metric = metric_mean
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= args.patience:
            logger.info(f"No improvement for {args.patience} epochs. Early stopping...")
            break
    
    # Log best performance
    logger.info(f"Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    
    # Log per-class metrics if available
    if args.metric == 'auc' and metric_individual is not None:
        for i in range(len(metric_individual)):
            logger.info(f'{class_names[i]}: {metric_individual[i]:.4f}')
    
    # Test phase
    logger.info("\nTesting best model...")
    checkpoint_path = os.path.join(ckpt_path, f'{method_name}.pth')
    

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
    classifier.load_state_dict(checkpoint['model'], strict=True)
    classifier = classifier.to(device)

    metric_individual, metric_mean = test(classifier, test_loader, device, num_classes)

    # Log test results
    logger.info(f'\nTest Results:\n'
               f'AUC: {metric_mean:.4f}')

    # Log per-class metrics
    for i in range(num_classes):
        logger.info(f'{class_names[i]}: {metric_individual[i]:.4f}')


if __name__ == '__main__':
    main()