import argparse
import datetime
import logging
import os
import sys
import time
from timeit import default_timer as timer

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import loader.biotacsp_loader
import dataset.biotacsp
import network.utils
import utils.evaluation
import numpy as np

'''test for param loading'''
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import utils.plotconfusionmatrix

checkpoint_path='./ckpts/'
checkpoint_fname='train-GCN_test-True-2-Sep28_20-37-28_fold0_val.pkl'



dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=2, split="test",normalize=False)
#validation_sampler_ = SubsetRandomSampler(np.random.choice(range(dataset_.data.y.shape[0])))
eval_loader_ = DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)#,sampler=validation_sampler_)
print('loading dataset...')
device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#'cpu'
print(str(torch.cuda.get_device_name(0)))

net=torch.load(checkpoint_path+checkpoint_fname)
model_ = network.utils.get_network("GCN_test", dataset_.data.num_features, eval_loader_.dataset.num_classes).to(device_)
model_.load_state_dict(net['model_state'])

print(f'Loading saved ckpts:{checkpoint_fname}...')

print('evaluating....')
# utils.evaluation.eval(model_, device_, eval_loader_)

''' testing for loading param'''
acc_ = 0.0
y_ = []
preds_ = []

model_.eval()
#print(f'len_loader:{len(loader)}')
#print(f'loader[0]: {loader._get_iterator()}')
with torch.no_grad():

    for batch in eval_loader_:

        batch = batch.to(device_)
        # print(f'batch:{batch}')
        # pres_ro=model(batch)
        pred_ = model_(batch).max(1)[1]
        acc_ += pred_.eq(batch.y).sum().item()

        y__ = batch.y.cpu().numpy()
        pred__=pred_.cpu().numpy()

        y_.append(y__)
        preds_.append(pred__)
        print(f'y: {batch.y[0]}, pred_:{pred_[0]}')
        # print(f'preds_: {preds_}')



    acc_ /= len(eval_loader_)
    print(f'accuracy: {acc_}')

    prec_, rec_, fscore_, _ = precision_recall_fscore_support(y_, preds_, average='weighted')




conf_matrix_ = confusion_matrix(y_, preds_)
# print(f'y_:{y_}, pred_:{preds_}')
## Plot non-normalized confusion matrix
utils.plotconfusionmatrix.plot_confusion_matrix(conf_matrix_, classes=np.unique(y_),
                    title='Confusion matrix with TDR augmentation')