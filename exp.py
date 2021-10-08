import tqdm
import time
import datetime
import random
import numpy as np
from document_reader import *
from os import listdir
from os.path import isfile, join
from EventDataset import EventDataset
import sys
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from metric import metric, CM_metric
from transformers import RobertaModel
import os
import os.path
from os import path
import json
from json import JSONEncoder
import notify
from notify_message import *
from notify_smtp import *
from util import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class exp:
    def __init__(self, cuda, model, epochs, learning_rate, train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, MATRES_best_PATH, HiEve_best_PATH, load_model_path, model_name = None, roberta_size = "roberta-large"):
        self.cuda = cuda
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.finetune = finetune
        self.train_dataloader = train_dataloader
        self.valid_dataloader_HIEVE = valid_dataloader_HIEVE # do not distinguish b/w HiEve & IC
        self.test_dataloader_HIEVE = test_dataloader_HIEVE # do not distinguish b/w HiEve & IC
        
        ### fine-tune roberta or not ###
        # if finetune is False, we use fixed roberta embeddings before bilstm and mlp
        self.roberta_size = roberta_size
        if not self.finetune:
            self.RoBERTaModel = RobertaModel.from_pretrained(self.roberta_size).to(self.cuda)
        if self.roberta_size == 'roberta-base':
            self.roberta_dim = 768
        else:
            self.roberta_dim = 1024
        
        self.HiEve_best_F1 = -0.000001
        self.HiEve_best_prfs = []
        self.HiEve_best_PATH = HiEve_best_PATH
        
        self.IC_best_F1 = -0.000001
        self.IC_best_prfs = []
        self.IC_best_PATH = HiEve_best_PATH # Just accept as an input parameter
        
        self.load_model_path = load_model_path
        self.model_name = model_name
        self.best_epoch = 0
        self.file = open("./rst_file/" + model_name + ".rst", "w")
        
    def my_func(self, x_sent):
        # for Bi-LSTM
        my_list = []
        for sent in x_sent:
            my_list.append(self.RoBERTaModel(sent.unsqueeze(0))[0].view(-1, self.roberta_dim))
        return torch.stack(my_list).to(self.cuda)
    
    def train(self):
        total_t0 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True) # AMSGrad
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            self.model.train()
            self.total_train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                batch_to_cuda = [i.to(self.cuda) for i in batch]    
                alpha_logits, beta_logits, gamma_logits, loss = self.model(batch_to_cuda, loss_out = True)
                   
                self.total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("  Total training loss: {0:.2f}".format(self.total_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            if self.dataset in ["HiEve", "MATRES", "IC"]:
                flag = self.evaluate(self.dataset)
            else:
                '''
                Joint settings in JCL_ENNLP20
                '''
                flag = self.evaluate("HiEve")
                flag = self.evaluate("MATRES")
            if flag == 1:
                self.best_epoch = epoch_i
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        if self.dataset in ["HiEve", "Joint"]:
            print("  HiEve best F1_PC_CP_avg: {0:.3f}".format(self.HiEve_best_F1))
            print("  HiEve best precision_recall_fscore_support:\n", self.HiEve_best_prfs)
            print("  Dev best:", file = self.file)
            print("  HiEve best F1_PC_CP_avg: {0:.3f}".format(self.HiEve_best_F1), file = self.file)
            print("  HiEve best precision_recall_fscore_support:", file = self.file)
            print(self.HiEve_best_prfs, file = self.file)
        if self.dataset in ["IC"]:
            print("  IC best F1_PC_CP_avg: {0:.3f}".format(self.IC_best_F1))
            print("  IC best precision_recall_fscore_support:\n", self.IC_best_prfs)
            print("  Dev best:", file = self.file)
            print("  IC best F1_PC_CP_avg: {0:.3f}".format(self.IC_best_F1), file = self.file)
            print("  IC best precision_recall_fscore_support:", file = self.file)
            print(self.IC_best_prfs, file = self.file)
        return -0.000001, self.HiEve_best_F1, self.IC_best_F1
            
    def evaluate(self, eval_data, test = False, predict = False):
        # ========================================
        #             Validation / Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # Also applicable to test set.
        t0 = time.time()
            
        if test:
            if self.load_model_path:
                self.model = torch.load(self.load_model_path + self.model_name + ".pt")
            elif eval_data == "HiEve":
                self.model = torch.load(self.HiEve_best_PATH)
            elif eval_data == "IC":
                self.model = torch.load(self.IC_best_PATH)
            else: # MATRES
                self.model = torch.load(self.MATRES_best_PATH)
            self.model.to(self.cuda)
            print("")
            print("loaded " + eval_data + " best model:" + self.model_name + ".pt")
            if predict == False:
                print("(from epoch " + str(self.best_epoch) + " )")
            print("Running Evaluation on " + eval_data + " Test Set...")
            if eval_data == "MATRES":
                dataloader = self.test_dataloader_MATRES
            elif eval_data in ["HiEve", "IC"]:
                dataloader = self.test_dataloader_HIEVE # do not distinguish b/w HiEve & IC
            
        else:
            # Evaluation
            print("")
            print("Running Evaluation on Validation Set...")
            if eval_data == "MATRES":
                dataloader = self.valid_dataloader_MATRES
            else:
                dataloader = self.valid_dataloader_HIEVE # do not distinguish b/w HiEve & IC
            
        self.model.eval()
        
        y_pred = []
        y_gold = []
        y_gold_seg = []
        y_pred_seg = []
        y_logits = np.array([[0.0, 1.0, 2.0, 3.0]])
        
        # Evaluate data for one epoch
        for batch in dataloader:
            with torch.no_grad():
                batch_to_cuda = [i.to(self.cuda) for i in batch]    
                alpha_logits, beta_logits, gamma_logits = self.model(batch_to_cuda, loss_out = None)
            assert list(alpha_logits.size())[1] == 5
                
            # Move logits and labels to CPU
            label_ids = batch_to_cuda[18].to('cpu').numpy()
            label_seg = batch_to_cuda[15].to('cpu').numpy()
            y_predict = torch.max(alpha_logits[:, 0:4], 1).indices.cpu().numpy()
            y_predict_seg = torch.squeeze(alpha_logits[:, 4:]).to('cpu').numpy()
            
            y_pred.extend(y_predict)
            try:
                y_pred_seg.extend(y_predict_seg)
            except:
                y_pred_seg.append(y_predict_seg)
            y_gold.extend(label_ids)
            y_gold_seg.extend(label_seg)
            
            y_logits = np.append(y_logits, alpha_logits[:, 0:4].cpu().numpy(), 0) # for prediction result output
                
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("Eval took: {:}".format(validation_time))
        
        if predict:
            if predict[-4:] == "json":
                with open(predict, 'w') as outfile:
                    if eval_data == "MATRES":
                        numpyData = {"labels": "0 -- Before; 1 -- After; 2 -- Equal; 3 -- Vague", "array": y_logits}
                    else:
                        numpyData = {"labels": "0 -- Parent-Child; 1 -- Child-Parent; 2 -- Coref; 3 -- NoRel", "array": y_logits}
                    json.dump(numpyData, outfile, cls=NumpyArrayEncoder)
                try:
                    msg = message(subject=eval_data + " Prediction Notice",
                              text=self.dataset + "/" + self.model_name + " Predicted " + str(y_logits.shape[0] - 1) + " instances. (Current Path: " + os.getcwd() + ")")
                    send(msg)  # and send it
                except:
                    pass
                return 0
            else:
                with open(predict + "gold", 'w') as outfile:
                    for i in y_gold:
                        print(i, file = outfile)
                with open(predict + "pred", 'w') as outfile:
                    for i in y_pred:
                        print(i, file = outfile)   
        
        if eval_data in ["HiEve", "IC"]:
            # Report the final accuracy for this validation run.
            cr = classification_report(y_gold, y_pred, output_dict = True)
            rst = classification_report(y_gold, y_pred)
            F1_PC = cr['0']['f1-score']
            F1_CP = cr['1']['f1-score']
            F1_coref = cr['2']['f1-score']
            F1_NoRel = cr['3']['f1-score']
            F1_PC_CP_avg = (F1_PC + F1_CP) / 2.0
            print(rst)
            print("  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg))
            
            correct = 0
            total = 0
            for i in range(len(y_gold_seg)):
                total += 1
                if (y_pred_seg[i] >= 0.5 and y_gold_seg[i] == 1) or (y_pred_seg[i] < 0.5 and y_gold_seg[i] == 0):
                    correct += 1
            print("  Segmentation perf: {0:.3f}".format(correct/total))
            
            if test:
                print("  Test rst:", file = self.file)
                print(rst, file = self.file)
                print("  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg), file = self.file)
                print("  Segmentation perf: {0:.3f}".format(correct/total), file = self.file)
                
                try:
                    msg = message(subject=eval_data + " Test Notice", text = self.dataset + "/" + self.model_name + " Test results:\n" + "  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg) + " (Current Path: " + os.getcwd() + ")")
                    send(msg)  # and send it
                except:
                    pass
                
            if not test:
                try:
                    msg = message(subject=eval_data + " Validation Notice", text = self.dataset + "/" + self.model_name + " Validation results:\n" + "  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg) + " (Current Path: " + os.getcwd() + ")")
                    send(msg)  # and send it
                except:
                    pass
                if eval_data == "HiEve":
                    if F1_PC_CP_avg > self.HiEve_best_F1 or path.exists(self.HiEve_best_PATH) == False:
                        self.HiEve_best_F1 = F1_PC_CP_avg
                        self.HiEve_best_prfs = rst
                        torch.save(self.model, self.HiEve_best_PATH)
                        return 1
                else:
                    if F1_PC_CP_avg > self.IC_best_F1 or path.exists(self.IC_best_PATH) == False:
                        self.IC_best_F1 = F1_PC_CP_avg
                        self.IC_best_prfs = rst
                        torch.save(self.model, self.IC_best_PATH)
                        return 1
        return 0