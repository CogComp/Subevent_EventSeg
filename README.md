Link to pretrained constraints: 



# Learning Constraints and Descriptive Segmentation for Subevent Detection

This is the repository for the resources in EMNLP 2021 Paper ["Learning Constraints and Descriptive Segmentation for Subevent Detection"](https://arxiv.org/pdf/2109.06316.pdf). This repository contains the source code and datasets used in our paper.

## Abstract

Event mentions in text correspond to real-world events of varying degrees of granularity. The task of subevent detection aims to resolve this granularity issue, recognizing the membership of multi-granular events in event complexes. Since knowing the span of descriptive contexts of event complexes helps infer the membership of events, we propose the task of event-based text segmentation (EventSeg) as an auxiliary task to improve the learning for subevent detection. To bridge the two tasks together, we propose an approach to learning and enforcing constraints that capture dependencies between subevent detection and EventSeg prediction, as well as guiding the model to make globally consistent inference. Specifically, we adopt Rectifier Networks for constraint learning and then convert the learned constraints to a regularization term in the loss function of the neural model. Experimental results show that the proposed method outperforms baseline methods by 2.3% and 2.5% on benchmark datasets for subevent detection, HiEve and IC, respectively, while achieving a decent performance on EventSeg prediction. 

## Dataset

Two datasets ([HiEve](https://github.com/CogComp/Subevent_EventSeg/tree/main/hievents_v2) and [IC](https://github.com/CogComp/Subevent_EventSeg/tree/main/IC)) are used for training in the paper. 

## How to train
### Environment Setup et al.
```
git clone git@github.com:CogComp/Subevent_EventSeg.git
conda env create -n conda-env -f env/environment.yml
pip install -r env/requirements.txt
python spacy -m en-core-web-sm

mkdir rst_file
mkdir model_params
cd model_params
mkdir HiEve_best
mkdir IC_best
cd ..
```
### Running experiments in the paper
`python main.py <DEVICE_ID> <RESULT_FILE>`

`<DEVICE_ID>`: choose from "gpu_0", "gpu_1", "gpu_5,6,7", etc.

`<RESULT_FILE>`: for example, "1236.rst"

### Example commands 
`nohup python main.py gpu_1 1236.rst > output_redirect/1236.out 2>&1 &`

To look at the standard output: `cat output_redirect/1236.out`

## How to predict

### Input & Output

Input should be a json file that contains a list of dictionaries. Each dictionary contains 6 key-value pairs, i.e., two sentences, two char id's denoting the start position of events, and the two event mentions. Examples can be found under [example](https://github.com/CogComp/Subevent_EventSeg/tree/main/example) folder.

Output will also be a json file under [output](https://github.com/CogComp/Subevent_EventSeg/tree/main/output) folder. The output contains a dictionary with two key-value pairs; one is labels, the other is predicted probabilities.

### How to run 
`python predict.py <INPUT_FILE> <OUTPUT_FILE>`

`<INPUT_FILE>`: a json file

`<OUTPUT_FILE>`: name for a json file

### Example commands
#### Command for predicting temporal relations
`python predict.py example/subevent_example_input.json predict_subevent.json`

#### [Link to pre-trained model](https://drive.google.com/drive/folders/1T_lOE75mzK86NzEhWDxC9rHKEdVc1_KB?usp=sharing)

### Sending notifications
[Changing the google account settings](https://www.google.com/settings/security/lesssecureapps)
## Reference
Bibtex:
```
@inproceedings{WZCR21,
    author = {Haoyu Wang and Hongming Zhang and Muhao Chen and Dan Roth},
    title = {{Learning Constraints and Descriptive Segmentation for Subevent Detection}},
    booktitle = {Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2021},
    url = "https://cogcomp.seas.upenn.edu/papers/WZCR21.pdf",
    funding = {KAIROS, BETTER},
}
```
