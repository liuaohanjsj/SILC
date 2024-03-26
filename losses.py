# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:17:07 2022

@author: Admin
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import OrderedDict

from cider.pyciderevalcap import CiderD
from pycocoevalcap.bleu.bleu import Bleu

def CE(output, target):
    '''
    Output: (B,L,C)。未经过softmax的logits
    Target: (B,L)
    '''
    output = output.reshape(-1, output.shape[-1])  # (*,C)
    target = target.reshape(-1).long()  # (*)
    return nn.CrossEntropyLoss()(output, target) #默认size_average=True，会把B*L所有词loss平均

def contrast_loss(sim):
    """
    sim: [B, miniB]
    """
    n = sim.shape[1]
    labels = torch.from_numpy(np.arange(n)).to(sim.device)
    loss = torch.FloatTensor([0]).to(sim.device)
    for i in range(sim.shape[0]//n):
        loss += nn.CrossEntropyLoss()(sim[i*n:i*n+n], labels) #这里已经对minibatch的每个元素结果取平均了
        loss += nn.CrossEntropyLoss()(sim[i*n:i*n+n].transpose(0,1), labels)
    return loss/2/(sim.shape[0]//n) #和CE保持一致，取平均。6.27(model 13)之前写的为啥是loss/2/n

def contrast_loss_i2s(sim, capseg):
    """
    sim_i2s: [B,N_sen,N_sen]。其中B可能是多个mini batch。
    这里N_sen是Batch里所有句子最大的n_sen，计算时为了每个句子正确，应每个句子单独算n_sen。
    capseg: [B, L]。句子开始应为id+100
    [b,i,j]是第b个报告，第i个句子对应的gated feature和第j个句子的sentence content相似度
    """
    loss = torch.FloatTensor([0]).to(sim.device)
    for i in range(sim.shape[0]):
        n_sen = torch.max(capseg[i]).cpu().long()-100
        labels = torch.from_numpy(np.arange(n_sen)).to(sim.device)
        loss += nn.CrossEntropyLoss()(sim[i, :n_sen, :n_sen], labels)
        #print(sim[i, :n_sen, :n_sen])
        #print(nn.CrossEntropyLoss()(sim[i, :n_sen, :n_sen], labels))
    return loss/sim.shape[0]

def contrast_loss_s2i(sim, capseg):
    """
    sim_s2i: [B/miniB * N_sen, miniB, miniB], [i,j,k]表示第j个报告的第i个句子feature，和其gating用在第k个CT上的feature的相似度
    capseg: [B, L]。句子开始应为id+100
    distinctiveness: [B/miniB*N_sen, miniB] 值为[0,1]。
    """
    assert capseg.shape[0]==sim.shape[1] #目前先只考虑单卡的情况
    loss = torch.FloatTensor([0]).to(sim.device)
    for i in range(sim.shape[1]):
        n_sen = torch.max(capseg[i]).cpu().long()-100 
        labels = torch.tensor([i]).repeat(n_sen).to(sim.device)
        batch_loss = nn.CrossEntropyLoss(reduce = False)(sim[:n_sen, i], labels) #[n_sen]
        loss += torch.mean(batch_loss)
        
    return loss/sim.shape[1]


def RewardCriterion(input, seq, reward, ignore_zero=True):
    """
    以下的B实际可能是batch*n_sample，简称B。
    input: [B,L,n_voc]，每个位置每个词预测的概率的log
    seq：[B,L]，实际采样得到的预测
    reward: [B,L]，每行所有值是相同的
    """
    temp = input.gather(2, seq.unsqueeze(2)).squeeze(2) #[B,L]
    if ignore_zero:
        mask = (seq>0).float()
    else:
        mask = (seq>=0).float()
    output = - temp * reward * mask 
    output = torch.sum(output) / (torch.sum(mask)+1e-6)
    return output

CiderD_scorer = None

def clear_scorer():
    global CiderD_scorer
    CiderD_scorer = None
def get_self_critical_reward(greedy_res, data_gts, gen_result, cider_weight=0, bleu_weight=1, sigma=6.0, bleu_n=4, tokenizer=None,
                             eos=2, sos=1, corpus_path='cider/data/mimic.p', cider_n=4, corpus_ids = None):
    """
    greedy_res: [B,L], tensor
    data_gts: [B,L], array
    gen_result: [B*seq_per_img, L], tensor, seq_per_img=1
    """
    global CiderD_scorer
    if CiderD_scorer is None:
        print('corpus path', corpus_path)
        if isinstance(corpus_ids, list): 
            #对每个句子根据其topic token分别使用corpus。
            #corpus_ids应是包含topic token id的长为B的list
            #此时corpus_path 应该是一个文件夹，里面包含以topic token id命名的很多p文件
            CiderD_scorer = {}
            paths = list(Path(corpus_path).glob('*.p'))
            for path in paths:
                CiderD_scorer[int(path.name[:-2])] = CiderD(df=str(path), n=cider_n, sigma=sigma)
        else:
            CiderD_scorer = CiderD(df=corpus_path, n=cider_n, sigma=sigma)
   
    def array_to_str(arr):
        out = ''
        for i in range(len(arr)):
            if arr[i]==0 or arr[i]==eos:
                break
            if arr[i]==sos:
                continue
            out += str(arr[i]) + ' '
        if len(out.strip())==0:
            out = '0'
        return out.strip()

    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts)
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
        if tokenizer is not None:
            print(tokenizer.decode(gen_result[i]))
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]
        if tokenizer is not None:
            print(tokenizer.decode(greedy_res[i]))
    #res前面B*seq_per_img是采样预测，后面B是greedy预测
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i])]
        if tokenizer is not None:
           print(tokenizer.decode(data_gts[i]))
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    if isinstance(corpus_ids, list):
        print(corpus_ids)
        corpus_ids_ = [corpus_ids[i // seq_per_img] for i in range(gen_result_size)]
    #gts_和res保持一致，个数为B*seq_per_img+B
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    if isinstance(corpus_ids, list):
        corpus_ids_.extend([corpus_ids[i] for i in range(batch_size)])

    if cider_weight > 0:
        if isinstance(corpus_ids, list):
            cider_scores = np.zeros(len(gts_))
            for i in range(len(gts_)):
                if corpus_ids_[i] not in CiderD_scorer: #极少数情况会出现，可能由于模型精度问题。
                    corpus_ids_[i] = 0
                _, temp = (CiderD_scorer[corpus_ids_[i]]).compute_score({i:gts_[i]}, res_[i:i+1])
                cider_scores[i] = temp
        else:
            _, cider_scores = CiderD_scorer.compute_score(gts_, res_) #[B*seq_per_img+B]
    else:
        cider_scores = 0
    if bleu_weight > 0:
        score, scores = Bleu(4).compute_score(gts_, res__) #score是所有序列总bleu，scores是每个样本的bleu
        if not isinstance(bleu_n, tuple):
            bleu_scores = np.array(scores[bleu_n-1]) #bleu_4是scores[3]
        else:
            bleu_scores = np.stack([np.array(scores[i-1]) for i in bleu_n], axis=0)
            bleu_scores = np.mean(bleu_scores, axis=0)
    else:
        bleu_scores = 0
    scores = cider_weight * cider_scores + bleu_weight * bleu_scores
    real_rewards = scores[:gen_result_size]#[B*seq_per_img], 无baseline的reward
    #sample预测的score 减去greedy预测的score（broadcast），大小为[B, seq_per_img]
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size) #reshape成1维的[B*seq_per_img]

    return scores, real_rewards
