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

### Read command line parameters ###
if len(sys.argv) > 1:
    input_file, f_out = sys.argv[1], sys.argv[2]
    
gpu_num = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
cuda = torch.device('cuda')
rst_file_name = '1236.rst'
f_out = "output/" + f_out

params = {
            'time': dt_string,
            'gpu_num': gpu_num,
            'task': 'subevent',
            'dataset': 'IC',
            'cons_no': "0429_5_cons",
            'add_loss': 1, 
            'lambda_annoH': 1.0, ####
            'lambda_annoS': 1.0, ####
            'lambda_cons': 1.0,  ####
            'MLP_size': 4096, 
            'roberta_hidden_size': 1024,
            'finetune': 1,
            'epochs': 10,
            'batch_size': 2, 
            'downsample': 1.0,
            'learning_rate': 0.00000001,
            'debugging': 0
         }

if params['cons_no'] in ['0429_3_cons', '0429_4_cons', '0429_5_cons', '0429_6_cons']:
    params['fold'] = 4
else:
    params['fold'] = 5
    
model_params_dir = "./model_params/"
HiEve_best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt")
IC_best_PATH = model_params_dir + "IC_best/" + rst_file_name.replace(".rst", ".pt")
model_name = rst_file_name.replace(".rst", "")
num_classes = 4

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Processing input data...")
with open(input_file) as f:
    input_list = json.load(f)
test_set = []
for an_instance in input_list:
    x_sent, x_position = subword_id_getter_space_split(an_instance["sent_1"], an_instance["e1_start_char"])
    y_sent, y_position = subword_id_getter_space_split(an_instance["sent_2"], an_instance["e2_start_char"])
    x_subword_len = len(tokenizer.encode(an_instance['e1_mention'])) - 2
    y_subword_len = len(tokenizer.encode(an_instance['e2_mention'])) - 2
    if an_instance["sent_1"] == an_instance["sent_2"]:
        xy_sent = padding(x_sent, max_sent_len = 193)
        to_append = xy_sent, x_position, x_position+x_subword_len, y_position, y_position+y_subword_len, \
                    xy_sent, x_position, x_position+x_subword_len, y_position, y_position+y_subword_len, \
                    xy_sent, x_position, x_position+x_subword_len, y_position, y_position+y_subword_len, \
                    0, 0, 0, \
                    0, 0, 0, 0, 0, 1, 0, 0
    else:
        xy_sent = padding(x_sent + y_sent[1:], max_sent_len = 193)
        to_append = xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                    xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                    xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                    0, 0, 0, \
                    0, 0, 0, 0, 0, 1, 0, 0
    test_set.append(to_append)
test_dataloader = DataLoader(EventDataset(test_set), batch_size=params['batch_size'], shuffle = False)

model = roberta_mlp_cons(num_classes, params, cuda)
model.to(cuda)
model.zero_grad()
print("# of parameters:", count_parameters(model))
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data.size())
model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters
    
mem_exp = exp(cuda, model, params['epochs'], params['learning_rate'], None, None, None, None, test_dataloader, params['finetune'], params['dataset'], None, IC_best_PATH, None, model_name)
mem_exp.evaluate(eval_data = params['dataset'], test = True, predict = f_out)
