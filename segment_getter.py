from os import listdir
from os.path import isfile, join
import json
import networkx as nx
import xml.etree.ElementTree as ET
import tqdm
import numpy as np

def get_span(my_dict, connected_component):
    # connected component: several events
    # return: earliest sentence id, last sentence id
    return my_dict['event_dict'][min(connected_component)]['sent_id'], my_dict['event_dict'][max(connected_component)]['sent_id']

def get_sentID(my_dict, connected_component):
    sentIDs = []
    for i in connected_component:
        sentIDs.append(my_dict['event_dict'][i]['sent_id'])
    return sentIDs

def segments(G, my_dict):
    # a list of lists
    # each list in the return list: [seg_start_sent_id, seg_end_sent_id, event_ids, sent_ids]
    segments = []
    for connected_component in list(nx.connected_components(G)):
        seg_start, seg_end = get_span(my_dict, connected_component)
        segments.append([seg_start, seg_end, connected_component, get_sentID(my_dict, connected_component)])
    return segments

def reside_in(start, end, segmentation):
    # determine which segment_id
    for i in range(len(segmentation)-1):
        if start > segmentation[i] and end <= segmentation[i+1]:
            return i
    return -1
        
def count_cross_seg(sentIDs, segmentation):
    count = set()
    # for each event, it has a sentID, if all events appear in the same segment, return 0
    for sent in sentIDs:
        count.add(reside_in(sent, sent, segmentation))
    return len(count) - 1

def score(sdg, segmentation):
    # segmentation: [-1, 5, 10, 12, 17, 20, 24], or [-1, 5, 17, 24]
    # sdg: [[0, 5, {2, 3, 4, 8, 13}, [0, 0, 0, 2, 5]], [6, 10, {14, 20, 22, 23, 25}, [6, 9, 9, 9, 10]], [11, 12, {32, 33, 35}, [11, 11, 12]], [10, 24, {36, 37, 51, 56, 28, 30}, [12, 12, 20, 24, 10, 10]], [13, 17, {41, 43, 46}, [13, 16, 17]], [20, 20, {48, 49, 50}, [20, 20, 20]]]
    # sdg is always the same for one document
    # segmentation can be any possible combination
    score = 0
    reside = {}
    sdg_id = -1
    for start, end, con, sentIDs in sdg:
        # for one connected component, it corresponds to a key in dict "reside"
        sdg_id += 1
        res = reside_in(start, end, segmentation)
        reside[sdg_id] = res
        # if it resides in only one segment, score will increase (by the number of events)
        if res > -1:
            score += len(con)
        # else score will decrease
        else:
            score -= count_cross_seg(sentIDs, segmentation) * 1
    #"""        
    for i in range(len(sdg)):
        for j in range(1+i, len(sdg)):
            if reside[i] == reside[j] and reside[i] != -1:
                # if two connected components are within the same segment:
                score -= len(sdg[i][2])
    #"""
    #score += len(segmentation)
    
    return score

def all_possible_seg(sdg):
    segs = set()
    for start, end, con, _ in sdg:
        #if start - 1 >= 0:
        #    segs.add(start-1)
        segs.add(end)
    return segs

def find_segment(sent_id, target_segment):
    seg_num = len(target_segment)
    for i in range(seg_num - 1):
        if sent_id > target_segment[i] and sent_id <=target_segment[i+1]:
            return i
        
def same_segment(sent_id_1, sent_id_2, target_segment):
    if find_segment(sent_id_1, target_segment) == find_segment(sent_id_2, target_segment):
        return 1
    else:
        return 0
    
    
def segment_getter_HiEve(fname, my_dict):
    mypath = './hievents_v2/'
    tree = ET.parse(mypath+fname)
    root = tree.getroot()
    G = nx.Graph()
    DG = nx.DiGraph()
    if True:
        for child in root:
            if child.tag == "Relations":
                for RelationInfo in child:
                    if RelationInfo[2].text == "SuperSub":
                        G.add_edge(int(RelationInfo[0].text), int(RelationInfo[1].text))
                        DG.add_edge(int(RelationInfo[0].text), int(RelationInfo[1].text))
    
    sdg = segments(G, my_dict)
    if len(sdg) == 1:
        # remove the single root node and corresponding edges from graph and run again
        root = [n for n,d in DG.in_degree() if d==0] 
        G.remove_edges_from(DG.edges(root[0]))
        sdg = segments(G, my_dict)
    all_possible_segmentation = list(all_possible_seg(sdg))
    aps = sorted(all_possible_segmentation)
    flip = pow(2, len(aps))
    score_dict = {}
    for i in range(flip):
        str_ = str(bin(i))
        str_ = str_[2:]
        segmentation = [-1]
        for digit in range(len(str_)):
            if str_[digit] == "1":
                segmentation.append(aps[digit])
        score_dict[tuple(segmentation)] = score(sdg, segmentation)

    sorted_score = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse = True)}
    assert_value = 0
    for key, value in sorted_score.items():
        assert_value += 1
        target_segment = list(key)
        break
    assert assert_value == 1
    if len(my_dict['sentences']) - 1 in target_segment:
        return target_segment
    else:
        target_segment.append(len(my_dict['sentences']) - 1)
        return target_segment
# updated on Apr 04, 2021


def segment_getter_IC(fname, my_dict):
    mypath = "./IC/LDC2016E47_IC_Domain_Event_Annotation_From_CMU_V1.0/data/"
    tree = ET.parse(mypath+fname)
    root = tree.getroot()
    
    G = nx.Graph()
    DG = nx.DiGraph()
    
    relation_dict = {}
    num = 1
    relations = ['coreference', 'subevent_of', 'in_reporting', 'member_of']
    last_eventid = ''
    eventid2num = {}
    for sentence in root:
            for word in sentence:
                if word.get('wd'):
                    if word.get('eventid') and last_eventid != word.get('eventid'):
                        if word.get('event_type'):
                            event_type = word.get('event_type')
                        else:
                            event_type = 'event'
                        eventid2num[word.get('eventid')] = num
                        num += 1
                        last_eventid = word.get('eventid')
                    for relation in relations:
                        if word.get(relation):
                            if word.get(relation).find('+'):
                                eventid_list = word.get(relation).split('+')
                                for eventid in eventid_list:
                                    relation_dict[(word.get('eventid'), eventid)] = relation
                            else:
                                relation_dict[(word.get('eventid'), word.get(relation))] = relation
                                
    relation_dict_fixed = {}
    for key, value in relation_dict.items():
        try:
            relation_dict_fixed[(eventid2num[key[0]], eventid2num[key[1]])] = value
        except:
            why = 1
            #print(fname)
            #print(key[0])
            #print(key[1])                            

    for edge, rel in relation_dict_fixed.items():
        if rel in ['subevent_of', 'member_of']:
            G.add_edge(int(edge[1]), int(edge[0]))
            DG.add_edge(int(edge[1]), int(edge[0]))
                
    sdg = segments(G, my_dict)
    if len(sdg) == 1:
        # remove the single root node and corresponding edges from graph and run again
        root = [n for n,d in DG.in_degree() if d==0] 
        G.remove_edges_from(DG.edges(root[0]))
        sdg = segments(G, my_dict)
    all_possible_segmentation = list(all_possible_seg(sdg))
    aps = sorted(all_possible_segmentation)
    flip = pow(2, len(aps))
    score_dict = {}
    for i in range(flip):
        str_ = str(bin(i))
        str_ = str_[2:]
        segmentation = [-1]
        for digit in range(len(str_)):
            if str_[digit] == "1":
                segmentation.append(aps[digit])
        score_dict[tuple(segmentation)] = score(sdg, segmentation)

    sorted_score = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse = True)}

    assert_value = 0
    for key, value in sorted_score.items():
        assert_value += 1
        target_segment = list(key)
        break
    assert assert_value == 1
    
    if len(my_dict['sentences']) - 1 in target_segment:
        return target_segment
    else:
        target_segment.append(len(my_dict['sentences']) - 1)
        return target_segment
    
    
    
"""
def score(sdg, segmentation):
    # segmentation: [-1, 5, 10, 12, 17, 20, 24]
    # sdg: [[0, 5, {2, 3, 4, 8, 13}, [0, 0, 0, 2, 5]], [6, 10, {14, 20, 22, 23, 25}, [6, 9, 9, 9, 10]], [11, 12, {32, 33, 35}, [11, 11, 12]], [10, 24, {36, 37, 51, 56, 28, 30}, [12, 12, 20, 24, 10, 10]], [13, 17, {41, 43, 46}, [13, 16, 17]], [20, 20, {48, 49, 50}, [20, 20, 20]]]
    score = 0
    reside = {}
    sdg_num = -1
    for start, end, con, sentIDs in sdg:
        sdg_num += 1
        reside[sdg_num] = reside_in(start, end, segmentation)
        res = reside_in(start, end, segmentation)
        if res > -1:
            score += len(con)
        else:
            score -= count_cross_seg(sentIDs, segmentation)
    for i in range(len(sdg)):
        for j in range(1+i, len(sdg)):
            if reside[i] == reside[j]:
                score -= len(sdg[i][2])
    return score
"""
        