import tqdm
import time
import datetime
import random
random.seed(10)
from document_reader import *
from os import listdir
from os.path import isfile, join
from EventDataset import EventDataset
from torch.utils.data import Dataset, DataLoader
from util import *
import json

rel_type = {"IC": 0, "HiEve": 0, "MATRES": 1, "TB-Dense": 1, "CaTeRS": 2, "RED": 3}

def data(dataset, debugging, downsample, batch_size, undersmp_ratio = 0.4, shuffle = True):
    train_set = []
    valid_set = []
    test_set = []
    train_range = range(0, 60)
    valid_range = range(60, 80)
    test_range = range(80, 100)
    count_rel = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("Processing " + dataset + " dataset...")
    if dataset == "IC":
        dir_name = "./IC/IC_Processed/"
        max_sent_len = 193
    elif dataset == "HiEve":
        dir_name = "./hievents_v2/processed/"
        max_sent_len = 155
    else:
        print("Not supporting this dataset yet!")
        
    onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == "tsvx"]
    onlyfiles.sort()
    doc_id = -1
    t0 = time.time()
    for file_name in tqdm.tqdm(onlyfiles):
        doc_id += 1
        if doc_id in train_range:
            my_dict = tsvx_reader(dataset, dir_name, file_name)
            num_event = len(my_dict["event_dict"])
            #print("num_event", num_event)
                
            # range(a, b): [a, b)
            for x in range(1, num_event+1):
                for y in range(x+1, num_event+1):
                    for z in range(y+1, num_event+1):
                        x_sent_id = my_dict["event_dict"][x]["sent_id"]
                        y_sent_id = my_dict["event_dict"][y]["sent_id"]
                        z_sent_id = my_dict["event_dict"][z]["sent_id"]

                        x_sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
                        y_sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]
                        z_sent = my_dict["sentences"][z_sent_id]["roberta_subword_to_ID"]
                            
                        '''
                        concat two sentences following PairedRL
                        <s> sentence_1 </s> sentence_2 </s> <pad> ... <pad>
                        0, ..., 2, ..., 2, 1, ..., 1
                        '''
                        xy_sent = padding(x_sent + y_sent[1:], max_sent_len = max_sent_len)
                        yz_sent = padding(y_sent + z_sent[1:], max_sent_len = max_sent_len)
                        xz_sent = padding(x_sent + z_sent[1:], max_sent_len = max_sent_len)
                        
                        '''
                        HiEve PairedRL context max length: 155
                        IC PairedRL context max length: 193
                        '''
                        #_max = max(_max, len(xy_sent), len(yz_sent), len(xz_sent))
                            
                        x_position = my_dict["event_dict"][x]["roberta_subword_id"]
                        y_position = my_dict["event_dict"][y]["roberta_subword_id"]
                        z_position = my_dict["event_dict"][z]["roberta_subword_id"]
                            
                        x_subword_len = len(tokenizer.encode(my_dict["event_dict"][x]['mention'])) - 2
                        y_subword_len = len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
                        z_subword_len = len(tokenizer.encode(my_dict["event_dict"][z]['mention'])) - 2

                        #x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
                        #y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)
                        #z_sent_pos = padding(my_dict["sentences"][z_sent_id]["roberta_subword_pos"], pos = True)

                        xy = my_dict["relation_dict"][(x, y)]["relation"]
                        yz = my_dict["relation_dict"][(y, z)]["relation"]
                        xz = my_dict["relation_dict"][(x, z)]["relation"]
                            
                        x_seg_id = my_dict["event_dict"][x]["segment_id"]
                        y_seg_id = my_dict["event_dict"][y]["segment_id"]
                        z_seg_id = my_dict["event_dict"][z]["segment_id"]
                        
                        to_append = xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                                    yz_sent, y_position, y_position+y_subword_len, len(y_sent)-1+z_position, len(y_sent)-1+z_position+z_subword_len, \
                                    xz_sent, x_position, x_position+x_subword_len, len(x_sent)-1+z_position, len(x_sent)-1+z_position+z_subword_len, \
                                    float(x_seg_id==y_seg_id), float(y_seg_id==z_seg_id), float(x_seg_id==z_seg_id), \
                                    xy, yz, xz, rel_type[dataset], x, y, z, doc_id
                        '''
                        if xy == 3 and yz == 3:
                            pass
                        elif xy == 3 or yz == 3 or xz == 3:
                            if random.uniform(0, 1) <= downsample:
                                train_set.append(to_append)
                                count_rel[xy] += 1
                                count_rel[yz] += 1
                                count_rel[xz] += 1
                        else:
                            train_set.append(to_append)
                            count_rel[xy] += 1
                            count_rel[yz] += 1
                            count_rel[xz] += 1
                        '''    
                        if xy == 3:
                            if random.uniform(0, 1) <= downsample:
                                train_set.append(to_append)
                                count_rel[xy] += 1
                                count_rel[yz] += 1
                                count_rel[xz] += 1
                        else:
                            train_set.append(to_append)
                            count_rel[xy] += 1
                            count_rel[yz] += 1
                            count_rel[xz] += 1
        else:
            my_dict = tsvx_reader(dataset, dir_name, file_name)
            num_event = len(my_dict["event_dict"])
            for x in range(1, num_event+1):
                for y in range(x+1, num_event+1):
                    x_sent_id = my_dict["event_dict"][x]["sent_id"]
                    y_sent_id = my_dict["event_dict"][y]["sent_id"]
                    x_sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
                    y_sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]
                    xy_sent = padding(x_sent + y_sent[1:], max_sent_len = max_sent_len)

                    x_position = my_dict["event_dict"][x]["roberta_subword_id"]
                    y_position = my_dict["event_dict"][y]["roberta_subword_id"]
                        
                    x_subword_len = len(tokenizer.encode(my_dict["event_dict"][x]['mention'])) - 2
                    y_subword_len = len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2

                    #x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
                    #y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)
                        
                    x_seg_id = my_dict["event_dict"][x]["segment_id"]
                    y_seg_id = my_dict["event_dict"][y]["segment_id"]

                    xy = my_dict["relation_dict"][(x, y)]["relation"]

                    to_append = xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                                xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                                xy_sent, x_position, x_position+x_subword_len, len(x_sent)-1+y_position, len(x_sent)-1+y_position+y_subword_len, \
                                float(x_seg_id==y_seg_id), float(x_seg_id==y_seg_id), float(x_seg_id==y_seg_id), \
                                xy, xy, xy, rel_type[dataset], x, y, x, doc_id

                    if doc_id in valid_range:
                        if xy == 3:
                            if random.uniform(0, 1) <= undersmp_ratio:
                                valid_set.append(to_append)
                        else:
                            valid_set.append(to_append)
                    else:
                        '''
                        undersmp_ratio is set as 1.0 in PR-E_complex
                        0.4 in TacoLM (down-sample negative NoRel instances with a probability of 0.4)
                        '''
                        if xy == 3:
                            if random.uniform(0, 1) <= undersmp_ratio:
                                test_set.append(to_append)
                        else:
                            test_set.append(to_append)
                            
    if debugging:
        train_set = train_set[0:1000]
        valid_set = test_set = train_set
    elapsed = format_time(time.time() - t0)
    print(dataset + " Preprocessing took {:}".format(elapsed))
    print(dataset + f' training instance num: {len(train_set)}')
    print(dataset + f' validation instance num: {len(valid_set)}')
    print(dataset + f' test instance num: {len(test_set)}')
    
    if dataset in ["HiEve", "IC"]:
        num_classes = 4
        train_dataloader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle = shuffle)
        valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=batch_size, shuffle = False)    
        test_dataloader = DataLoader(EventDataset(test_set), batch_size=batch_size, shuffle = False) 
        return train_dataloader, None, None, valid_dataloader, test_dataloader, num_classes, count_rel