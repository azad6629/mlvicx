from trainer import  MLVICXTrainer
import time
import argparse
import os
import yaml
import random
import numpy as np
import torch


def read_config(arch,model_name):
    config_file = model_name.lower() + ".yaml"
    config_file_path = os.path.join(f'./configs',arch,config_file)
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


parser = argparse.ArgumentParser(description='MLVICX Pre-Training.')
parser.add_argument('-mode', default = 'ssl',help='ssl or sl')
parser.add_argument('-tmode', default = 'pre',help='down = downstream transformations or pre=pre-training transformations')
parser.add_argument('-init', default = 'imagenet',help='random or imagenet')
parser.add_argument('-model', default = 'mlvicx', help='choice of SSL model, choices=model_names')
parser.add_argument('-arch', default = 'resnet18', help='model architecture.')
parser.add_argument('-bs', default=256, type=int, help='batch size.')
parser.add_argument('-epoch', default=300, help='total epoches.')
parser.add_argument('-dataset', default='Chex14', choices=['NIH14', 'Chex14'],help='Dataset to use')
parser.add_argument('-resume', default=False, action='store_true', help='To resume the pre-training.')
parser.add_argument('-seed', default = 42, help='seed for initializing training. ')
parser.add_argument('-gpu', default=0,help='GPU id to use.')


def main():   
    trainers = {'mlvicx': MLVICXTrainer}
    args = parser.parse_args()      
    config = read_config(args.arch, args.model)
    
    config['gpu']= args.gpu
    config['model_name']= args.model
    config['model']['backbone']['type'] = args.arch
    config['data']['dataset']= args.dataset
    config['data']['pre_bs']= args.bs
    config['optimizer']['total_epochs']= args.epoch
    config['mode'] = args.mode
    config['tmode'] = args.tmode
    if args.init == 'random':
        config['model']['backbone']['pretrained']= False
    elif args.init == 'imagenet':
        config['model']['backbone']['pretrained']= True
        
    resume = args.resume   
    
    print('resume ',args.resume)
    
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    if args.model in trainers:
        trainer_class = trainers[args.model]
        trainer = trainer_class(args.model, config)    
        trainer.resume_model(resume)
        start_epoch = trainer.start_epoch
        total_training_time = 0
        for epoch in range(start_epoch + 1, trainer.total_epochs + 1):
            trainer.save_checkpoint(epoch)
            start_time = time.time()  
            trainer.train_epoch(epoch)
            trainer.save_checkpoint(epoch)
            end_time = time.time()  
            epoch_training_time = end_time - start_time
            total_training_time += epoch_training_time

        total_training_time_hours = total_training_time / 3600  
        print(f"Total training time: {total_training_time_hours:.2f} hours")
    else:
        print(f"No trainer found for model: {args.model}")

if __name__ == '__main__':
    main()




