# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:41:58 2023

@author: Admin
"""
import json
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from dataset import XRayDataset, Tokenizer
from configuration import Config
from models import ReportGenerator_contrast
from utils import to_device, Smoother, Logger, Checkpoint, Step, GradCAM, align_img
from utils import compute_scores, show_points, get_parameters, cosine_sim, set_seed

def get_model(conf):
    return ReportGenerator_contrast(n_token=conf['n_token'], max_l=conf['max_l'], 
                                    vbackbone=conf['vbackbone'], 
                                    SILC=conf['SILC'], MRG=conf['MRG'],
                                    normal_gen=conf['normal_gen'], normal_c=conf['normal_c'],
                                    n_content=conf['n_content'], real_n_content=conf['real_n_content'],
                                    text_d=conf['text_d'], use_vq=conf['VQ'],
                                    word_decoder_layer=conf['word_decoder_layer'],
                                    topic_decoder_layer=conf['topic_decoder_layer'],
                                    decoder_layer=conf['decoder_layer'],
                                    n_fuse_layer=conf['n_fuse_layer'],  
                                    multiview_fuse=conf['multiview_fuse'])

def get_dataset(conf, split, tokenizer, sample_lv=False):
    assert split in ['train', 'val', 'test'], split
    dataset = XRayDataset(conf['report_dir'], split=split, tokenizer=tokenizer, test=(split=='test'),
                           n_views=conf['n_views'], seg_caption = conf['SILC'],
                           decorate_sentences=conf['SILC'], paths = None, suffix=conf['data_suffix'],
                           nes=conf['no_empty_sentence'], delimiter=conf['delimiter'], augment=(split=='train' and conf['augment']),
                           image_size=conf['image_size'], image_pad=conf['image_pad'], sample_lv=(sample_lv if sample_lv else (conf['sample_lv'] if split=='train' else False)))
    return dataset

def evaluate(model, loader, tokenizer, output = False, output_topic=False, beam_size=1, max_l=100, repeat_single=True):
    """
    Infer the model on the validation/test set to evaluate the report generation performance. 
    """
    print(conf['output_dir'])
    Path(conf['output_dir']).mkdir(exist_ok = True, parents = True)
    if output:
        fp = open(conf['output_dir']+'/'+output, 'w')
    metrics = Smoother(1)
    gt_dict = {}
    pred_dict, pred_h_dict, pred_h2_dict = {},{},{}
    tot = 0

    with torch.no_grad():
        for (source, targets, debug) in tqdm(loader):
            source = to_device(source, 'cuda:0')
            targets = to_device(targets, 'cuda:0')
            if output_topic:
                optional_ret = ['topic_token', 'vq_idx'] #分别是预测的topic和真实的topic
            else:
                optional_ret = []
            pred, pred_h, pred_h2, optional = model(source, caption=None, 
                                                    caption_s=targets.get('find_s',None), 
                                                    capseg=targets.get('seg_s',None), beam_size=beam_size,
                                                    mode='test', optional_ret=optional_ret) #[B, L], [B, l1, l2]
            for i in range(targets['find'].shape[0]):
                gt_s = tokenizer.decode(targets['find'][i].cpu().numpy())
                pred_s = tokenizer.decode(pred[i].cpu().numpy())

                pred_s_h, pred_s_h2 = '', ''
                pred_h_token, pred_h2_token, = [], []
                
                for j in range(pred_h.shape[1]):
                    if np.sum(pred_h[i,j].cpu().numpy())==0: #一直（论文中）用的这个
                        break
                    temp = tokenizer.decode(pred_h[i,j].cpu().numpy())
                    if len(temp.strip())==0:
                        break
                    if output_topic:
                        pred_s_h += '['+str(optional['topic_token'][i,j].cpu().numpy())+']'
                    pred_s_h += temp + '|'
                    pred_h_token.extend(tokenizer.decode_as_token(pred_h[i,j].cpu().numpy()))
                for j in range(pred_h2.shape[1]):
                    temp = tokenizer.decode(pred_h2[i,j].cpu().numpy())
                    if temp=='':
                        continue
                    if output_topic:
                        pred_s_h2 += '['+str(optional['vq_idx'][i,j].cpu().numpy())+']'
                    pred_s_h2 += temp + '|'
                    pred_h2_token.extend(tokenizer.decode_as_token(pred_h2[i,j].cpu().numpy()))
                
                gt_token = tokenizer.decode_as_token(targets['find'][i].cpu().numpy())
                pred_token = tokenizer.decode_as_token(pred[i].cpu().numpy())
                
                #对于mimic multi-view，使用和single view时相同的sample，即每个image一个sample，即有些multi-view的sample要重复几次
                n_repeat = debug['n_views'][i] if conf['dataset']=='MIMIC_m' else 1
                for j in range(n_repeat):
                    gt_dict[tot] = [' '.join(gt_token[:max_l])]
                    pred_dict[tot] = [' '.join(pred_token[:max_l])]
                    pred_h_dict[tot] = [' '.join(pred_h_token[:max_l])]
                    pred_h2_dict[tot] = [' '.join(pred_h2_token)]
                    tot += 1
                
                if output:
                    fp.write(debug['paths'][0][i]+'\n'+gt_s+'\n')
                    if conf['normal_gen']:
                        fp.write(pred_s+'\n')
                    if pred_s_h=='':
                        pred_s_h = 'no report.'
                    fp.write(pred_s_h+'\n')
                    if pred_s_h2=='':
                        pred_s_h2 = 'no report.'
                    fp.write(pred_s_h2+'\n')
        pass
    score = compute_scores(gt_dict, pred_dict) #Only used in baseline model. Will be 0 for MRG model
    score_h = compute_scores(gt_dict, pred_h_dict) #MRG prediction
    score_h_gt = compute_scores(gt_dict, pred_h2_dict) #MRG Prediction bypassing topic decoder
    
    filt = lambda x: x if x>0 else None
    metrics.update(bleu = filt(score['BLEU_4']),
                   bleu_h = filt(score_h['BLEU_4']),
                   score_h_gt = filt(score_h_gt['BLEU_4']))
    print(score, score_h, score_h_gt)
    if output:
        fp.write(str(metrics.value())+'\n')
        fp.write('gen '+str(score)+'\n')
        fp.write('hier'+str(score_h)+'\n')
        fp.write('h gt'+str(score_h_gt)+'\n')
        fp.write(str(tot)+'\n')
        fp.close()
    return metrics

def test(conf, resume, split='test', max_l=100, beam_size=1, output_topic=False, shuffle=False,
         sample_lv=False):
    conf['output_dir'] = str(Path(resume).parent).replace('checkpoint', 'outputs')
    
    tokenizer = Tokenizer(conf['report_dir'], max_l, file_prefix=conf.get('token_prefix', ''), english=conf['english'])
    test_data = get_dataset(conf, split, tokenizer, sample_lv)
    test_loader = DataLoader(test_data, batch_size=conf['valid_batch'], shuffle=shuffle, num_workers=8, drop_last=False)
    
    model = get_model(conf)
    checkpoint = Checkpoint(model = model)
    if resume:
        checkpoint.resume(resume)
    model = nn.DataParallel(model)
    model.to('cuda:0')
    model.eval()
    
    output_file = 'pred_%s%s%s_beam%d.txt'%(split, resume.split('/')[-1][:-3],
                                            '' if not output_topic else '_topic', beam_size)
    metrics = evaluate(model, test_loader, tokenizer, output = output_file,
                       output_topic=output_topic, beam_size=beam_size, max_l=max_l)
    print(metrics.value())
    return metrics

def evaluate_clinical_improvement():
    def output_row(elements):
        for x in elements[:-1]:
            print(x, end=' & ')
        print(elements[-1], '\\\\')
    def impr_text(x, y):
        if y==0:
            return '%.03f (-)'%x
        if y>0:
            return '%.03f ($\\uparrow$%.03f)'%(x, x/y-1)
        return '%.03f ($\\downarrow$%.03f)'%(x, -x/y+1)
    freq = [0.068, 0.301, 0.436, 0.071, 0.477, 0.244, 0.108, 0.169, 0.311, 0.052, 0.365, 0.042, 0.067, 0.424]
    name = ['No finding', 'Enlarged cardiomediastinum', 'Cardiomegaly', 'Lung lesion','Lung opacity','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural effusion','Pleural other','Fracture','Support devices']
    baseline = []
    with open('baseline.txt', 'r') as fp:
        for i in range(14):
            row = fp.readline()
            baseline.append([float(x) for x in row.split()[-3:]])
    ours = []
    with open('ours.txt', 'r') as fp:
        for i in range(14):
            row = fp.readline()
            ours.append([float(x) for x in row.split()[-3:]])
    impr = []
    for i in range(14):
        improve = 0
        for j in range(0,3):
            if baseline[i][j]==0:
                continue
            improve += ours[i][j]/baseline[i][j]-1
        improve /= 3
        #improve = (sum(ours[i])-sum(baseline[i]))/(sum(baseline[i])+1e-9)
        impr.append(round(improve, 3))
        print(improve)
    #freq = [np.log(freq[i]) for i in range(14) if impr[i]!=0]
    for i in range(len(impr)):
        output_row([name[i], freq[i], baseline[i][0], baseline[i][1], baseline[i][2], 
                    impr_text(ours[i][0], baseline[i][0]), 
                    impr_text(ours[i][1], baseline[i][1]), 
                    impr_text(ours[i][2], baseline[i][2]), 
                    impr[i]])
    baseline_ave = [sum(baseline[x][i] for x in range(len(baseline)))/len(baseline) for i in range(3)]
    ours_ave = [sum(ours[x][i] for x in range(len(ours)))/len(ours) for i in range(3)]
    
    output_row(['Average', '-', baseline_ave[0], baseline_ave[1], baseline_ave[2], 
                impr_text(ours_ave[0], baseline_ave[0]), 
                impr_text(ours_ave[1], baseline_ave[1]), 
                impr_text(ours_ave[2], baseline_ave[2]), 
                '-'])
    freq = [freq[i] for i in range(14) if impr[i]!=0]
    name = [name[i] for i in range(14) if impr[i]!=0]
    baseline = [baseline[i] for i in range(14) if impr[i]!=0]
    
    ours = [ours[i] for i in range(14) if impr[i]!=0]
    impr = [impr[i] for i in range(14) if impr[i]!=0]
    
    plt.scatter(freq, impr)
    print(freq, ours)
    #plt.scatter([x for x in freq], ours)
    for i in range(len(freq)):
        plt.annotate(name[i], (freq[i]+0.005, impr[i]+0.01), xytext=None, arrowprops=None)
    linear_model = np.polyfit(freq, impr, 1)
    print(linear_model)
    linear_model_fn = np.poly1d(linear_model)
    x_s = np.arange(60)/100
    plt.plot(x_s, linear_model_fn(x_s), color="green")
    plt.xlim(0, 0.6)
    plt.xlabel('Positive ratio', fontsize=15)
    plt.ylabel('Relative Improvement', fontsize=15)
    plt.savefig('improve.svg', bbox_inches='tight')

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', type=float)
    parser.add_argument('-resume', type=str, default='')
    parser.add_argument('-max_l', type=int)
    parser.add_argument('-beam_size', type=int, default=1)
    parser.add_argument('-seed', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    version = args.version
    seed = args.seed
    set_seed(seed)
    conf = Config(version, 0, seed=seed)
    test(conf, args.resume, split='test', max_l=args.max_l, beam_size=args.beam_size)
    
