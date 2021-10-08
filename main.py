import numpy as np 
from document_reader import *
import os
import sys 
import torch 
from model import *
from exp import * 
from data import * 
import json 
from datetime import datetime 

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

### Read parameters ###
if len(sys.argv) > 1:
    gpu_num, rst_file_name = sys.argv[1][4:], sys.argv[2]
    
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
cuda = torch.device('cuda')

params = {
            'time': dt_string,
            'gpu_num': gpu_num,
            'task': 'subevent',
            'dataset': 'HiEve', # HiEve or IC
            'cons_no': "0429_6_cons", # 0429_6_cons or 0429_5_cons
            'add_loss': 1, 
            'lambda_annoH': 1.0,
            'lambda_annoS': 1.0,
            'lambda_cons': 1.0,
            'MLP_size': 4096,
            'roberta_hidden_size': 1024,
            'finetune': 1,
            'epochs': 10,
            'batch_size': 3, # 3 or 2
            'downsample': 1.0,
            'learning_rate': 0.00000001,
            'debugging': 0,
         }

if params['cons_no'] in ['0429_3_cons', '0429_4_cons', '0429_5_cons', '0429_6_cons']:
    params['fold'] = 4
else:
    params['fold'] = 5

with open("config/" + rst_file_name.replace("rst", "json"), 'w') as config_file:
    json.dump(params, config_file)
    
model_params_dir = "./model_params/"
HiEve_best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt")
IC_best_PATH = model_params_dir + "IC_best/" + rst_file_name.replace(".rst", ".pt")
model_name = rst_file_name.replace(".rst", "")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# MEM: membership relations; TEMP: temporal relations
train_dataloader, valid_dataloader_TEMP, test_dataloader_TEMP, valid_dataloader_MEM, test_dataloader_MEM, num_classes = data(params['dataset'], params['debugging'], params['downsample'], params['batch_size'])

model = roberta_mlp_cons(num_classes, params, cuda)
model.to(cuda)
model.zero_grad()
print("# of parameters:", count_parameters(model))
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())
model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters
total_steps = len(train_dataloader) * params['epochs']
print("Total steps: [number of batches] x [number of epochs] =", total_steps)

if params['dataset'] == 'HiEve':
    best_PATH = HiEve_best_PATH
else:
    best_PATH = IC_best_PATH
    
mem_exp = exp(cuda, model, params['epochs'], params['learning_rate'], train_dataloader, None, None, valid_dataloader_MEM, test_dataloader_MEM, params['finetune'], params['dataset'], None, best_PATH, None, model_name)
T_F1, H_F1, IC_F1 = mem_exp.train()
mem_exp.evaluate(eval_data = params['dataset'], test = True)
