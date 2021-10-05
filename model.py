import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from transformers import RobertaModel, RobertaConfig
from all_loss_aug import transitivity_loss_H_, transitivity_loss_T_, cross_category_loss_, segment_loss_
import numpy as np
from RectifierNetwork import RectifierNetwork
config = RobertaConfig.from_pretrained("roberta-large")
config.output_hidden_states = True

'''
HiEve Stats
'''
HierPC_h = 1802.0
HierCP_h = 1846.0
HierCo_h = 758.0
HierNo_h = 63755.0 
HierTo_h = HierPC_h + HierCP_h + HierCo_h + HierNo_h # total number of event pairs
hier_weights_h = [0.25*HierTo_h/HierPC_h, 0.25*HierTo_h/HierCP_h, 0.25*HierTo_h/HierCo_h, 0.25*HierTo_h/HierNo_h]

'''
IC Stats
'''
HierPC_i = 2248.0 # before ignoring implicit events: 2257
HierCP_i = 2338.0 # 2354
HierCo_i = 2353.0 # 2358
HierNo_i = 81887.0 # 81857
HierTo_i = HierPC_i + HierCP_i + HierCo_i + HierNo_i # total number of event pairs
hier_weights_i = [0.25*HierTo_i/HierPC_i, 0.25*HierTo_i/HierCP_i, 0.25*HierTo_i/HierCo_i, 0.25*HierTo_i/HierNo_i]

def sum_2_power(dim):
    my_list = []
    for i in range(2 ** dim):
        binary = bin(i)[2:].zfill(dim)
        my_list.append([int(i) for i in binary])
    return np.array(my_list)

# roberta + MLP + Constraints
class roberta_mlp_cons(nn.Module):
    def __init__(self, num_classes, lambdas, cuda, Sub = None, Mul = True, freq = None):
        super(roberta_mlp_cons, self).__init__()
        self.cuda = cuda
        self.dataset = lambdas['dataset']
        self.lambdas = lambdas
        self.Sub = Sub
        self.Mul = Mul
        self.add_loss = lambdas['add_loss']
        self.hidden_size = int(lambdas['roberta_hidden_size'])  # concat last 4 layer of RoBERTa
        self.MLP_size = int(lambdas['MLP_size'])
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax()
        self.model = RobertaModel.from_pretrained("roberta-large", config=config)
        self.hier_class_weights_h = torch.FloatTensor(hier_weights_h).cuda()
        self.hier_class_weights_i = torch.FloatTensor(hier_weights_i).cuda()
        self.HiEve_anno_loss = nn.CrossEntropyLoss(weight=self.hier_class_weights_h)
        self.IC_anno_loss = nn.CrossEntropyLoss(weight=self.hier_class_weights_i)
        self.seg_loss = nn.BCEWithLogitsLoss()
        #self.ones = torch.ones([batch_size, 1]).float().to(self.cuda)
        self.zero = Variable(torch.zeros(1), requires_grad=False)
        self.cons_net = torch.load("model_params/cons_learn/" + lambdas['cons_no'] + ".pt")
        self.relu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        
        for param in self.cons_net.parameters():
            param.requires_grad = False
            
        if lambdas['fold'] == 5:
            self.fold = 5
            self.sum_2_power = torch.from_numpy(sum_2_power(5)).float().to(self.cuda)
        else:
            self.fold = 4
            self.sum_2_power = torch.from_numpy(sum_2_power(4)).float().to(self.cuda)
        
        if freq is not None:
            self.fc1 = nn.Linear(self.hidden_size*4+1, self.MLP_size)  
        else:
            if self.Sub is None and self.Mul is None:
                self.fc1 = nn.Linear(self.hidden_size*4*2, self.MLP_size)
                self.fc2 = nn.Linear(self.MLP_size, num_classes)
            elif self.Sub is not None and self.Mul is not None:
                self.fc1 = nn.Linear(self.hidden_size*4*4, self.MLP_size) # 1024 * 4 * 4, 512 * 4 * 4
                self.fc2 = nn.Linear(self.MLP_size, num_classes)
                self.fc2_1 = nn.Linear(self.MLP_size, 1)
            else:
                self.fc1 = nn.Linear(self.hidden_size*4*3, self.MLP_size)
                self.fc15 = nn.Linear(self.MLP_size, self.MLP_size)
                self.fc2 = nn.Linear(self.MLP_size, num_classes)
                self.fc2_1 = nn.Linear(self.MLP_size, 1)
        
    def forward(self, batch, loss_out = None):
        batch_size = batch[1].size(0)
        
        '''
        Reference: https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        out = model(input_ids)
        hidden_states = out[2]
        # get last four layers
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        '''
        output_xy = torch.cat(tuple([self.model(batch[0])[2][i] for i in (-1, -2, -3, -4)]), dim=-1)
        output_yz = torch.cat(tuple([self.model(batch[5])[2][i] for i in (-1, -2, -3, -4)]), dim=-1)
        output_xz = torch.cat(tuple([self.model(batch[10])[2][i] for i in (-1, -2, -3, -4)]), dim=-1)
        
        output_xy_x = torch.cat([torch.mean(output_xy[i, batch[1][i].long():batch[2][i].long(), :].unsqueeze(0), dim=1) for i in range(0, batch_size)], 0)
        output_xy_y = torch.cat([torch.mean(output_xy[i, batch[3][i].long():batch[4][i].long(), :].unsqueeze(0), dim=1) for i in range(0, batch_size)], 0)
        
        output_yz_y = torch.cat([torch.mean(output_yz[i, batch[6][i].long():batch[7][i].long(), :].unsqueeze(0), dim=1) for i in range(0, batch_size)], 0)
        output_yz_z = torch.cat([torch.mean(output_yz[i, batch[8][i].long():batch[9][i].long(), :].unsqueeze(0), dim=1) for i in range(0, batch_size)], 0)
        
        output_xz_x = torch.cat([torch.mean(output_xz[i, batch[11][i].long():batch[12][i].long(), :].unsqueeze(0), dim=1) for i in range(0, batch_size)], 0)
        output_xz_z = torch.cat([torch.mean(output_xz[i, batch[13][i].long():batch[14][i].long(), :].unsqueeze(0), dim=1) for i in range(0, batch_size)], 0)
        # torch.Size([16, 4096])
        
        if self.Sub is None and self.Mul is None:
            alpha_representation = torch.cat((output_xy_x, output_xy_y), 1)
            beta_representation = torch.cat((output_yz_y, output_yz_z), 1)
            gamma_representation = torch.cat((output_xz_x, output_xz_z), 1)
            # torch.Size([16, 4096*2])
        elif self.Sub is not None and self.Mul is not None:
            subAB = torch.sub(output_xy_x, output_xy_y)
            subBC = torch.sub(output_yz_y, output_yz_z)
            subAC = torch.sub(output_xz_x, output_xz_z)
            mulAB = torch.mul(output_xy_x, output_xy_y)
            mulBC = torch.mul(output_yz_y, output_yz_z)
            mulAC = torch.mul(output_xz_x, output_xz_z)
            alpha_representation = torch.cat((output_xy_x, output_xy_y, subAB, mulAB), 1)
            beta_representation = torch.cat((output_yz_y, output_yz_z, subBC, mulBC), 1)
            gamma_representation = torch.cat((output_xz_x, output_xz_z, subAC, mulAC), 1)
            # torch.Size([16, 4096*4])
        elif self.Sub is not None and self.Mul is None:
            subAB = torch.sub(output_xy_x, output_xy_y)
            subBC = torch.sub(output_yz_y, output_yz_z)
            subAC = torch.sub(output_xz_x, output_xz_z)
            alpha_representation = torch.cat((output_xy_x, output_xy_y, subAB), 1)
            beta_representation = torch.cat((output_yz_y, output_yz_z, subBC), 1)
            gamma_representation = torch.cat((output_xz_x, output_xz_z, subAC), 1)
            # torch.Size([16, 4096*3])
        else:
            mulAB = torch.mul(output_xy_x, output_xy_y)
            mulBC = torch.mul(output_yz_y, output_yz_z)
            mulAC = torch.mul(output_xz_x, output_xz_z)
            alpha_representation = torch.cat((output_xy_x, output_xy_y, mulAB), 1)
            beta_representation = torch.cat((output_yz_y, output_yz_z, mulBC), 1)
            gamma_representation = torch.cat((output_xz_x, output_xz_z, mulAC), 1)
            assert alpha_representation.size()[1] == 4096*3
            # torch.Size([16, 4096*3])
            
        alpha_logits_no_cons_ = self.fc2(self.tanh(self.fc15(self.tanh(self.fc1(alpha_representation)))))
        beta_logits_no_cons_ = self.fc2(self.tanh(self.fc15(self.tanh(self.fc1(beta_representation)))))
        gamma_logits_no_cons_ = self.fc2(self.tanh(self.fc15(self.tanh(self.fc1(gamma_representation)))))
        
        alpha_seg_no_cons_ = self.fc2_1(self.tanh(self.fc15(self.tanh(self.fc1(alpha_representation)))))
        beta_seg_no_cons_ = self.fc2_1(self.tanh(self.fc15(self.tanh(self.fc1(beta_representation)))))
        gamma_seg_no_cons_ = self.fc2_1(self.tanh(self.fc15(self.tanh(self.fc1(gamma_representation)))))
        
        # b means before calculating constraints
        alpha_logits_b = self.softmax(alpha_logits_no_cons_) # torch.Size([batch_size, 4])
        beta_logits_b = self.softmax(beta_logits_no_cons_)
        gamma_logits_b = self.softmax(gamma_logits_no_cons_)
        
        alpha_seg_b = torch.sigmoid(alpha_seg_no_cons_)
        beta_seg_b = torch.sigmoid(beta_seg_no_cons_)
        gamma_seg_b = torch.sigmoid(gamma_seg_no_cons_)
        
        alpha_logits = torch.cat([alpha_logits_b, alpha_seg_b], 1)
        beta_logits = torch.cat([beta_logits_b, beta_seg_b], 1)
        gamma_logits = torch.cat([gamma_logits_b, gamma_seg_b], 1)
            
        '''
        Constraints & Power Set et al.
        '''
        if self.add_loss == 2:
            if self.fold == 4:
                gamma_2_pow = torch.cat([torch.sum(self.sum_2_power * gamma_logits_b[i], 1).view((1, 16)) for i in range(batch_size)], 0)
                cons_feature = torch.cat([alpha_logits_b, alpha_seg_b, beta_logits_b, beta_seg_b, gamma_2_pow, gamma_seg_b], 1)
            else:
                gamma_2_pow = torch.cat([torch.sum(self.sum_2_power * gamma_logits[i], 1).view((1, 32)) for i in range(batch_size)], 0)
                cons_feature = torch.cat([alpha_logits_b, alpha_seg_b, beta_logits_b, beta_seg_b, gamma_2_pow], 1)
        elif self.add_loss == 1:
            gamma_2_pow = torch.cat([torch.sum(self.sum_2_power * gamma_logits_b[i], 1).view((1, 16)) for i in range(batch_size)], 0)
            cons_feature = torch.cat([alpha_logits_b, beta_logits_b, gamma_2_pow], 1)
        else:
            do_nothing = 1

        '''
        Calculating loss
        '''
        if loss_out is None:
            # Do not calculate or output the loss
            return alpha_logits, beta_logits, gamma_logits
        else:
            loss = 0.0
            if self.dataset in ["HiEve", "IC"]:
                # subevent relation annotation loss
                if self.dataset == "HiEve":
                    loss += self.lambdas['lambda_annoH'] * (self.HiEve_anno_loss(alpha_logits_no_cons_, batch[18]) + self.HiEve_anno_loss(beta_logits_no_cons_, batch[19]) + self.HiEve_anno_loss(gamma_logits_no_cons_, batch[20]))
                else:
                    loss += self.lambdas['lambda_annoH'] * (self.IC_anno_loss(alpha_logits_no_cons_, batch[18]) + self.IC_anno_loss(beta_logits_no_cons_, batch[19]) + self.IC_anno_loss(gamma_logits_no_cons_, batch[20]))
                    
                if self.lambdas['lambda_annoS'] > 0.0:
                    # segmentation annotation loss
                    loss += self.lambdas['lambda_annoS'] * (self.seg_loss(alpha_seg_no_cons_, batch[15].unsqueeze(1)) + self.seg_loss(beta_seg_no_cons_, batch[16].unsqueeze(1)) + self.seg_loss(gamma_seg_no_cons_, batch[17].unsqueeze(1)))
                
                if self.add_loss > 0.0:
                    loss += self.lambdas['lambda_cons'] * (-1.0) * torch.log(self.cons_net(cons_feature)).sum()
                    
            return alpha_logits, beta_logits, gamma_logits, loss
