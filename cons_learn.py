import os
from os import listdir
from os.path import isfile, join
import tqdm
import random
from nltk.corpus import wordnet as wn
from RectifierNetwork import RectifierNetwork
import torch 
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from segment_getter import *
from document_reader import *
import numpy as np
import collections
import json
import sys

random.seed(42)
torch.manual_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
cuda = torch.device('cuda')

### Read parameters ###
if len(sys.argv) > 0:
    rst_file_name = sys.argv[1] # e.g., 0425_0_cons.rst
    
args = {'dataset': "HiEve", \
        'add_seg': False, \
        '0000': True, \
        'hidden_rect': 10, \
        'lr': 0.001, \
        'fold': 4, \
        'batch_size': 8, \
        'epoch_num': 1000, \
        'exp_no': rst_file_name.replace(".rst", "")
       }

with open("config/" + args['exp_no'] + ".json", 'w') as config_file:
    json.dump(args, config_file)

file = open("rst_file/" + args["exp_no"] + ".rst", 'w')
label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
num_dict = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}

def label_to_num(label):
    return label_dict[label]
def num_to_label(num):
    return num_dict[num]

def my_random(my_list):
    index = random.randrange(0, len(my_list) - 1)
    my_list[index] = (my_list[index] + 1) % 2 
    return my_list

def verb_synset(mention):
    try:
        s = wn.synsets(mention, pos = wn.VERB)[0]
    except:
        lem = wn.lemmas(mention)
        s = wn.synsets(str(lem[0].derivationally_related_forms()[0].name()), pos = wn.VERB)[0]
    return s

def similarity(mention_1, mention_2):
    try:
        s1 = verb_synset(mention_1)
        s2 = verb_synset(mention_2)
        sim = s1.lch_similarity(s2)
    except:
        sim = 0
    return sim

def onehot_(alpha, dim = 4):
    identity_matrix = np.identity(dim)
    return [int(i) for i in identity_matrix[alpha].tolist()]

def sum_2_power(onehot, dim = 4):
    my_list = []
    for i in range(2 ** dim):
        binary = bin(i)[2:].zfill(dim)
        if args['0000']:
            my_sum = 0
            for j in range(dim):
                my_sum += int(binary[j]) * int(onehot[j])
            my_list.append(my_sum)
        else:
            if binary[0:4] != '0000':
                my_sum = 0
                for j in range(dim):
                    my_sum += int(binary[j]) * int(onehot[j])
                my_list.append(my_sum)
            
    return my_list

def convert(instance, fold):
    rel_ij = instance[0]
    rel_jk = instance[1]
    rel_ik = instance[2]
    if len(instance) == 3:
        onehot_ij = onehot_(rel_ij)
        onehot_jk = onehot_(rel_jk)
        onehot_ik = onehot_(rel_ik)
        power_ik = sum_2_power(onehot_ik)
        if args['0000']:
            assert len(onehot_ij + onehot_jk + power_ik) == 4+4+16
        else:
            assert len(onehot_ij + onehot_jk + power_ik) == 4+4+16-1
        return onehot_ij + onehot_jk + power_ik
    else:
        onehot_ij = onehot_(rel_ij) + [instance[3]]
        onehot_jk = onehot_(rel_jk) + [instance[4]]
        onehot_ik = onehot_(rel_ik) + [instance[5]]
        power_ik = sum_2_power(onehot_ik, fold)
        if fold == 4:
            assert len(onehot_ij + onehot_jk + power_ik + [instance[5]]) == 4+4+16+3
            return onehot_ij + onehot_jk + power_ik + [instance[5]]
        else:
            assert len(onehot_ij + onehot_jk + power_ik) == 5+5+32
            return onehot_ij + onehot_jk + power_ik

if args['dataset'] == "HiEve":
    dir_name = "./hievents_v2/processed/"
else:
    dir_name = "./IC/IC_Processed/"
onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == "tsvx"]

positives = {}
xy = []
z = []
for file_name in tqdm.tqdm(onlyfiles):
    my_dict = tsvx_reader(args['dataset'], dir_name, file_name)
    if args['dataset'] == 'HiEve':
        segments = segment_getter_HiEve(file_name.replace("tsvx", "xml"), my_dict)
    else:
        segments = segment_getter_IC(file_name[0:-5], my_dict)
    event_num = len(my_dict['event_dict'])
    for i in range(1, 1+event_num):
        for j in range(i+1, 1+event_num):
            for k in range(j+1, 1+event_num):
                alpha = my_dict['relation_dict'][(i, j)]['relation']
                beta = my_dict['relation_dict'][(j, k)]['relation']
                gamma = my_dict['relation_dict'][(i, k)]['relation']
                mention_i = my_dict['event_dict'][i]['mention']
                mention_j = my_dict['event_dict'][j]['mention']
                mention_k = my_dict['event_dict'][k]['mention']
                sent_id_i = my_dict['event_dict'][i]["sent_id"]
                sent_id_j = my_dict['event_dict'][j]["sent_id"]
                sent_id_k = my_dict['event_dict'][k]["sent_id"]
                
                if args['add_seg']:
                    delta = same_segment(sent_id_i, sent_id_j, segments)
                    epsilon = same_segment(sent_id_j, sent_id_k, segments)
                    zeta = same_segment(sent_id_i, sent_id_k, segments)
                    positive = convert([alpha, beta, gamma, delta, epsilon, zeta], fold = args['fold'])
                    if (alpha, beta, gamma, delta, epsilon, zeta) in positives.keys():
                        positives[(alpha, beta, gamma, delta, epsilon, zeta)] += 1
                    else:
                        positives[(alpha, beta, gamma, delta, epsilon, zeta)] = 1
                else:
                    positive = convert([alpha, beta, gamma], fold = args['fold'])
                    if (alpha, beta, gamma) in positives.keys():
                        positives[(alpha, beta, gamma)] += 1
                    else:
                        positives[(alpha, beta, gamma)] = 1
                    
                if alpha+beta+gamma <= 6:
                    xy.append(positive)
                    z.append([1])
                elif alpha+beta+gamma <= 8:
                    if random.uniform(0, 1) < 0.1:
                        xy.append(positive)
                        z.append([1])
                else:
                    if random.uniform(0, 1) < 0.0001:
                        xy.append(positive)
                        z.append([1])
                            
od = collections.OrderedDict(sorted(positives.items()))
print(od)
print(len(xy))
pos_num = len(xy)
if args['add_seg']:
    negatives = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            if (i, j, k, l, m, n) not in positives.keys():
                                negatives.append([i, j, k, l, m, n])
else:
    negatives = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if (i, j, k) not in positives.keys():
                    negatives.append([i, j, k])
print(negatives)      

for i in range(pos_num):
    negative = convert(negatives[random.randint(0, len(negatives) - 1)], fold = args['fold'])
    assert len(negative) == len(positive)
    xy.append(negative)
    z.append([0])

xyz = list(zip(xy, z))
random.shuffle(xyz)
xy, z = zip(*xyz)

######################
# Exp                #
######################

def predict(net, loader):
    correct = 0
    total = 0
    for index, (inp, gold) in enumerate(loader):
        inp = inp.to(cuda)
        pred = net(inp)
        pred = pred.to('cpu').numpy()
        for i in range(len(pred)):
            if pred[i] >= 0.5:
                pred_b = 1
            else:
                pred_b = 0

            if pred_b == gold[i]:
                correct += 1
            total += 1
                
    print("Correct: ", correct)
    print("Total: ", total)
    return correct/total 

def print_param(net):
    for name, param in net.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                
def train(data, lab, data_dev, lab_dev):
    # Data Loaders are formed for the data splits
    dataset = TensorDataset(torch.tensor(data).float(), torch.tensor(lab).float())
    loader = DataLoader(dataset, batch_size = args['batch_size'], shuffle = True)

    dataset_dev = TensorDataset(torch.tensor(data_dev).float(), torch.tensor(lab_dev).float())
    loader_dev = DataLoader(dataset_dev, batch_size = args['batch_size'], shuffle = True)
    
    net = RectifierNetwork(len(data[0]), args['hidden_rect'])
    print(len(data[0]))
    print(args['hidden_rect'])
    net.to(cuda)
  
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    loss = nn.BCELoss()
    prev_best_dev = 0.0
    best_epoch = 0
    for ep in range(args['epoch_num']):
        # Training Loop
        train_loss = 0.0
        for index, (inp, gold) in enumerate(loader):
            inp = inp.to(cuda)
            gold = gold.to(cuda)
            optimizer.zero_grad()
            output = net(inp)
            #print(output)
            #print(gold)
            out_loss = loss(output, gold) 
            train_loss += out_loss.item()
            out_loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            train_acc = predict(net, loader)
            dev_acc = predict(net, loader_dev)

        print("Epoch : {}".format(ep+1))
        print("Training loss: {:.4f}".format(train_loss/len(data)))
        print("Training Accuracy: {:.4f} ".format(train_acc))
        print("Development Accuracy: {:.4f} \n".format(dev_acc))
        if dev_acc > prev_best_dev:
            torch.save(net, "model_params/cons_learn/" + args['exp_no'] + ".pt")
            prev_best_dev = dev_acc
            best_epoch = ep
        
        if train_acc == 1.0:
            print("Train Accuracy reached 1. Ending early!!", file = file)
            print("Epoch: {}".format(ep+1), file = file)
            print("Training loss: {:.4f}".format(train_loss/len(data)), file = file)
            print("Training Accuracy: {:.4f} ".format(train_acc), file = file)
            print("Development Accuracy: {:.4f} \n".format(dev_acc), file = file)
            
    print("Development Accuracy: {:.4f} \n".format(prev_best_dev), file = file)
    print("Epoch: {}".format(best_epoch+1), file = file)
            
num_train = int(len(xy) * 0.8)
data = xy[0:num_train]
lab = z[0:num_train]
data_dev = xy[num_train:]
lab_dev = z[num_train:]
train(data, lab, data_dev, lab_dev)

