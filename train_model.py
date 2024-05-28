# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
import csv
import argparse

from dataset import XRayDataset, Tokenizer
from config import Config
from models import ReportGenerator_contrast, VQ
from losses import CE, contrast_loss, contrast_loss_i2s, contrast_loss_s2i, RewardCriterion, get_self_critical_reward, clear_scorer
from utils import to_device, Smoother, Logger, Checkpoint, Step, GradCAM, align_img
from utils import compute_scores, show_points, get_parameters, cosine_sim, set_seed
from preprocess.process_cider import prepare_topic_token_cider

def compute_batch(model, source, targets, verbose = False, optional_ret = [], model_optional = [],
                  RL = False, tokenizer=None, epoch=0):
    """
    performs a forward step for a batch of data and compute the losses
    """
    source = to_device(source, 'cuda:0')
    targets = to_device(targets, 'cuda:0')
    length = torch.max(targets['len']).item()
    ret = {}
    optional = {}
    losses = {}
    
    if not RL: #Joint training of the whole model. Does not involve reinforcement learning.
        pred, sim, sim_s2i, pred_s, gating, \
        sen_token_pred, vq_idx, losses['loss_vq'], losses['loss_used'], optional = \
            model(source, targets['find'],
                  caption_s=targets.get('find_s',None), capseg=targets.get('seg_s',None),
                  verbose=False, optional_ret=model_optional) #[B, L, n_token], [B, miniB]
        if 'c' not in conf['ignore_loss']:
            losses['loss_c'] = contrast_loss(sim)
        if conf['SILC']:
            losses['loss_silc'] = contrast_loss_s2i(sim_s2i, targets['seg_s'])

        if conf['MRG']:
            length2 = torch.max(targets['find_s'].count_nonzero(dim=1))
            t = targets['find_s'][:, 1:length2]
            t = torch.where(t==1, 2, t) #把t的1(sos)替换为2(eos)进行loss计算
            losses['loss_g_s'] = CE(pred_s[:, :length2-1], t)
            temp = torch.nn.functional.normalize(gating, dim=2) #[B,n_sen,n_content]
            sim_gating = torch.matmul(temp, temp.transpose(1,2))
            losses['loss_topic'] = contrast_loss_i2s(sim_gating, targets['seg_s'])
            
        if conf['MRG']:
            assert conf['VQ']
            losses['loss_g_sen'] = CE(sen_token_pred[:,:-1], vq_idx) #1、这里不错位相减了。2、GT vq_idx中最长句子后面没有结束向量，但其他句子有
        
    else: #Reinforcement Learning. We use Self-Critical Sequence Training
        if conf['MRG']:
            max_sen_num = torch.max(targets['seg_s']).long().item()-100-(0 if conf['no_empty_sentence'] else 1) #batch中所有报告最大句子数。要弄成int而不是tensor，否则多卡时一个tensor喂给多个卡会出问题
            #-1是因为最后一句实际是空句子，GT只有sos eos，就不优化了。反正短的句子也有这一个句子被优化。
        else:
            max_sen_num = None
        model.eval()
        with torch.no_grad():
            #Use greedy prediction as as baseline (See SCST). b=batch，B=b*n_sen。
            #greedy_pred: [b*n_sen,L] 使用GT句子feature预测的所有句子
            #greedy_sen_token: [b,l]
            #greedy_pred0: [b,n_sen2, L]，两层解码预测的所有句子。第一层预测的n_sen2和真实的n_sen不一定一样
            greedy_pred, _, gt_sen_token, greedy_sen_token, _, greedy_pred0, eos = \
                            model(source, caption=None,  
                                  caption_s=targets.get('find_s',None), 
                                  capseg=targets.get('seg_s',None), beam_size=1, mode='greedy',
                                  max_sen_num=max_sen_num, decode_first=conf['RL_first'],
                                  decode_second=(not conf['RL_first_only'])) #[B, L]
            #print('greedy', greedy_pred.shape)
        model.train()
        #Predict again by sampling based on the probabilities, instead of greedy prediction
        #sample_sen_token前面的SOS已去掉，但其他tensor前面还是有SOS的，包括sample_sen_probs
        sample_pred, sample_probs, gt_sen_token, sample_sen_token, sample_sen_probs, sample_pred0, eos = \
                            model(source, caption=None, 
                                  caption_s=targets.get('find_s',None), 
                                  capseg=targets.get('seg_s',None), beam_size=1, mode='sample',
                                  max_sen_num=max_sen_num, decode_first=conf['RL_first'],
                                  decode_second=(not conf['RL_first_only'])) #[B, L], [B,L,n_token]。层级式时B=实际B*max_sen_num
        if sample_probs is not None:
            sample_probs = torch.log(sample_probs+1e-8) #log prob
        if conf['RL_first'] and conf['MRG']: #Train the topic decoder
            sample_sen_probs = torch.log(sample_sen_probs[:,1:]+1e-8) #log prob
            b = source.shape[0]
            if conf['RL_first_direct']: #Directly optimize the predicted topic tokens
                eos = eos.cpu().numpy()
                reward, real_reward = get_self_critical_reward(greedy_sen_token, gt_sen_token.cpu().numpy(), sample_sen_token, conf['cider_weight'],
                                                               conf['bleu_weight'], bleu_n=conf['bleu_n'],
                                                               tokenizer=None, eos=eos, sos=10000, corpus_path=str(Path(conf['model_dir']).parent)+'/mimic.p' if not conf['no_corpus'] else 'corpus',
                                                               cider_n=conf['cider_n']) #[b]
                reward = torch.from_numpy(reward).float().to(sample_sen_token.device)
                losses['loss_RL_first'] = RewardCriterion(sample_sen_probs, sample_sen_token, reward.unsqueeze(1).repeat(1,sample_sen_token.shape[1]), ignore_zero=False)
                losses['first_real_reward'] = torch.tensor([real_reward.mean()])
            """
            else: #Decode the predicted topic tokens into sentences (report), and optimize the generated report
                greedy_report = torch.zeros((b, conf['max_l']), dtype=torch.int64)
                for i in range(b):
                    tot = 0
                    for j in range(greedy_pred0.shape[1]):
                        for k in range(1, greedy_pred0.shape[2]): #从1开始忽略SOS
                            if greedy_pred0[i,j,k]==0 or greedy_pred0[i,j,k]==2:
                                break
                            if tot<greedy_report.shape[1]:
                                greedy_report[i,tot] = greedy_pred0[i,j,k]
                                tot += 1
                sample_report = torch.zeros((b, conf['max_l']), dtype=torch.int64)
                for i in range(b):
                    tot = 0
                    for j in range(sample_pred0.shape[1]):
                        for k in range(1, sample_pred0.shape[2]):
                            if sample_pred0[i,j,k]==0 or sample_pred0[i,j,k]==2:
                                break
                            if tot<sample_report.shape[1]:
                                sample_report[i,tot] = sample_pred0[i,j,k]
                                tot += 1
                reward, real_reward = get_self_critical_reward(greedy_report, targets['find'].cpu().numpy(), sample_report, conf['cider_weight'],
                                                               conf['bleu_weight'], sigma=20.0, bleu_n=conf['bleu_n'],
                                                               tokenizer=tokenizer) #[b] 2.20 sigma由6改为20
                reward = torch.from_numpy(reward).float().to(sample_pred.device)
                losses['loss_RL_first'] = RewardCriterion(sample_sen_probs, sample_sen_token, reward.unsqueeze(1).repeat(1,sample_sen_token.shape[1]))
                losses['report_real_reward'] = torch.tensor([real_reward.mean()])
                """
        if not conf['RL_first_only']: #Train the word decoder
            greedy_pred, sample_pred, sample_probs = greedy_pred[:,1:], sample_pred[:,1:], sample_probs[:,1:] #ignore the SOS token
            if not conf['MRG']:
                gts = targets['find'].cpu().numpy()
            else:
                caption = targets['find_s'].cpu().numpy()
                gts = np.zeros((caption.shape[0]*max_sen_num, caption.shape[1]), dtype=np.int32) #一定要int，否则后面转str会和pred不一样
                for i in range(caption.shape[0]):
                    k = i*max_sen_num-1
                    for j in range(caption.shape[1]):
                        if caption[i,j]==0:
                            break
                        if targets['seg_s'][i,j]>100:
                            if k+1-i*max_sen_num==torch.max(targets['seg_s']).long().item()-101: #后一个空句子
                                break
                            k += 1
                            l = 0
                        gts[k,l] = caption[i,j]
                        l += 1
                #print('gts', gts.shape, '\n', gts)
                #print('greedy_pred', greedy_pred.shape, '\n', greedy_pred)
            if conf['no_corpus']:
                corpus_path = 'corpus'
            elif conf['separate_corpus']:
                corpus_path = str(Path(conf['model_dir']).parent)+'/cider' 
            else:
                corpus_path = ('cider/data/IU.p' if 'IU' in conf['dataset'] else 'cider/data/mimic.p')
            
            reward, real_reward = get_self_critical_reward(greedy_pred, gts, sample_pred, conf['cider_weight'], conf['bleu_weight'], 
                                                           sigma=6.0, bleu_n=conf['bleu_n'], tokenizer=None, 
                                                           corpus_path = corpus_path,
                                                           cider_n=conf['cider_n'],
                                                           corpus_ids = gt_sen_token.flatten().cpu().numpy().tolist() if conf['separate_corpus'] else None) #[B]
            reward = torch.from_numpy(reward).float().to(sample_pred.device)
            #print('reward', reward.shape, '\n', reward)
            losses['loss_RL'] = RewardCriterion(sample_probs, sample_pred, reward.unsqueeze(1).repeat(1,sample_pred.shape[1]), ignore_zero=False) #2023.11.2在0_1 R55训练，出现所有句子都是0，ignore_zero改为False
            losses['reward'] = torch.tensor([reward.mean()])
            losses['real_reward'] = torch.tensor([real_reward.mean()])
        
        if not conf['MRG']:
            pred = greedy_pred
        else:
            if not conf['RL_first_only']:
                pred = greedy_pred.reshape(source.shape[0], -1, greedy_pred.shape[1])
            else:
                pred = greedy_sen_token
        pred_s = None
    if not RL and conf['w_g']:
        losses['loss_g'] = CE(pred[:, :length-1], targets['find'][:, 1:length]) #错开一位计算loss
    for x in optional_ret:
        ret[x] = eval(x)
    return pred, losses, pred_s, ret, optional

def validate(model, loader, epoch=0):
    """
    Compute the losses under eval mode on the validation set
    """
    valid_losses = Smoother(len(loader))
    with torch.no_grad():
        for (source, targets, _) in tqdm(loader):
            pred, losses, _,_,_ = compute_batch(model, source, targets, epoch=epoch)
            loss = torch.FloatTensor([0]).to('cuda:0')
            for x in losses:
                if x[:5]=='loss_':
                    loss += eval('conf["w_%s"]'%x[5:])*losses[x]
            losses['loss'] = loss
            valid_losses.update(loss={x:losses[x].item() for x in losses})
    return valid_losses.value()

def evaluate(model, loader, tokenizer, output = False, n=10000, output_topic=False, beam_size=1, max_l=100, repeat_single=True):
    """
    Infer the model on the validation/test set to evaluate the report generation performance. Returns NLG metrics rather than losses.
    """
    print(conf['output_dir'])
    Path(conf['output_dir']).mkdir(exist_ok = True, parents = True)
    if output:
        fp = open(conf['output_dir']+'/'+output, 'w')
    metrics = Smoother(n)
    gt_dict = {}
    pred_dict, pred_h_dict, pred_h2_dict, cnt = {},{},{},{}
    tot = 0
    n_sen, n_sen_baseline = 0, 0
    
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
            if tot+targets['find'].shape[0]>n:
                break
            for i in range(targets['find'].shape[0]):
                gt_s = tokenizer.decode(targets['find'][i].cpu().numpy())
                pred_s = tokenizer.decode(pred[i].cpu().numpy())
                n_sen_baseline += len(pred_s.split('.'))
                
                pred_s_h, pred_s_h2 = '', ''
                pred_h_token, pred_h2_token, = [], []
                
                for j in range(pred_h.shape[1]):
                    #if j==pred_h.shape[1]-1 or np.sum(pred_h[i,j+1].cpu().numpy())==0:#1.27.2023试下这个，这个可以把最后sos eos的空句子删掉
                    if np.sum(pred_h[i,j].cpu().numpy())==0: #一直（论文中）用的这个
                        break
                    temp = tokenizer.decode(pred_h[i,j].cpu().numpy())
                    if len(temp.strip())==0:
                        break
                    n_sen += 1
                    if output_topic:
                        pred_s_h += '['+str(optional['topic_token'][i,j].cpu().numpy())+']'
                    pred_s_h += temp + '|'
                    pred_h_token.extend(tokenizer.decode_as_token(pred_h[i,j].cpu().numpy()))
                for j in range(pred_h2.shape[1]):
                    #if j==pred_h2.shape[1]-1 or np.sum(pred_h2[i,j+1].cpu().numpy())==0:#1.27.2023加入这个（之前完全没有），这个可以把最后sos eos的空句子删掉
                    #    break
                    temp = tokenizer.decode(pred_h2[i,j].cpu().numpy())
                    if temp=='':
                        continue #会有中间有空句子的情况吗？没有的话就break。
                    if output_topic:
                        pred_s_h2 += '['+str(optional['vq_idx'][i,j].cpu().numpy())+']'
                    pred_s_h2 += temp + '|'
                    pred_h2_token.extend(tokenizer.decode_as_token(pred_h2[i,j].cpu().numpy()))
                
                gt_token = tokenizer.decode_as_token(targets['find'][i].cpu().numpy())
                pred_token = tokenizer.decode_as_token(pred[i].cpu().numpy())
                
                cnt[tot] = debug['n_views'][i] if conf['dataset']=='MIMIC_m' and repeat_single else 1
                #for j in range(n_repeat):
                gt_dict[tot] = [' '.join(gt_token[:max_l])]
                pred_dict[tot] = [' '.join(pred_token[:max_l])]
                pred_h_dict[tot] = [' '.join(pred_h_token[:max_l])]
                pred_h2_dict[tot] = [' '.join(pred_h2_token)]
                tot += 1
                    
                if output:
                    fp.write(debug['paths'][0][i]+'\n'+gt_s+'\n'+pred_s+'\n')
                    if pred_s_h=='':
                        pred_s_h = 'no report.'
                    fp.write(pred_s_h+'\n')
                    if pred_s_h2=='':
                        pred_s_h2 = 'no report.'
                    fp.write(pred_s_h2+'\n')
    print(n_sen/tot)
    score = compute_scores(gt_dict, pred_dict)
    #score_h = compute_scores(gt_dict, pred_h_dict)
    #score_h2 = compute_scores(gt_dict, pred_h2_dict)
    '''
    pred_dict_random = {}
    ids = list(range(tot))
    random.shuffle(ids)
    for i,x in enumerate(ids):
        pred_dict_random[x] = pred_h_dict[i]
    score_h_random = compute_scores(gt_dict, pred_dict_random)
    '''
    #对于mimic multi-view，使用和single view时相同的sample，即每个image一个sample，即有些multi-view的sample要重复几次
    for x in cnt: 
        for i in range(cnt[x]-1):
            #这里要保证tot确实是sample的个数
            gt_dict[tot] = gt_dict[x]
            pred_dict[tot] = pred_dict[x]
            pred_h_dict[tot] = pred_h_dict[x]
            pred_h2_dict[tot] = pred_h2_dict[x]
            tot += 1
    score_h = compute_scores(gt_dict, pred_h_dict)
    score_h2 = compute_scores(gt_dict, pred_h2_dict)
    
    filt = lambda x: x if x>0 else None #如果模型有一些结果没输出，score为0，不进行metric的更新
    metrics.update(bleu = filt(score['BLEU_4']),
                   bleu_h = filt(score_h['BLEU_4']),
                   bleu_h2 = filt(score_h2['BLEU_4']),
                   #bleu_hh = filt(score_hh['BLEU_4']),
                   )
    print(score, score_h, score_h2)
    if output:
        fp.write(str(metrics.value())+'\n')
        fp.write('gen '+str(score)+'\n')
        fp.write('hier'+str(score_h)+'\n')
        #fp.write('h sv'+str(score_hh)+'\n') #single view
        fp.write('h gt'+str(score_h2)+'\n')
        #fp.write('h rand'+str(score_h_random)+'\n')
        fp.write(str(tot)+'\n')
        fp.write(str(n_sen_baseline/tot)+'\n')
        fp.write(str(n_sen/tot))
        fp.close()
    return metrics

def get_model():
    return ReportGenerator_contrast(n_token=conf['n_token'], max_l=conf['max_l'], 
                                    vbackbone=conf['vbackbone'], n_content=conf['n_content'], SILC=conf['SILC'],
                                    MRG=conf['MRG'],
                                    text_d=conf['text_d'], use_vq=conf['VQ'],
                                    word_decoder_layer=conf['word_decoder_layer'],
                                    topic_decoder_layer=conf['topic_decoder_layer'],
                                    decoder_layer=conf['decoder_layer'],
                                    n_fuse_layer=conf['n_fuse_layer'], real_n_content=conf['real_n_content'], normal_gen=conf['normal_gen'], 
                                    multiview_fuse=conf['multiview_fuse'], normal_c=conf['normal_c'],
                                    pretrain=conf['pretrain'])

def get_dataset(split, tokenizer, sample_lv=False):
    assert split in ['train', 'val', 'test'], split
    dataset = XRayDataset(conf['report_dir'], split=split, tokenizer=tokenizer, test=(split=='test'),
                           n_views=conf['n_views'], seg_caption = conf['SILC'],
                           decorate_sentences=conf['SILC'], paths = None, suffix=conf['data_suffix'],
                           nes=conf['no_empty_sentence'], delimiter=conf['delimiter'], augment=(split=='train' and conf['augment']),
                           image_size=conf['image_size'], image_pad=conf['image_pad'], sample_lv=(sample_lv if sample_lv else (conf['sample_lv'] if split=='train' else False)))
    return dataset

def train(resume = None, stage = 1, RL=False, reset_epoch=False, transfer=False):
    tokenizer = Tokenizer(conf['report_dir'], conf['max_l'], file_prefix=conf.get('token_prefix', ''), english=conf['english'])
    
    train_data = get_dataset('train', tokenizer)
    valid_data = get_dataset('val', tokenizer)
    
    train_loader = DataLoader(train_data, batch_size=conf['batch_size'], shuffle=True, num_workers=max(8, conf['minibatch']), drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=max(8, conf['minibatch']), drop_last=True)
    

    model = get_model()
    step = Step()
    #checkpoint = Checkpoint(model = model, optimizer = optimizer)
    checkpoint = Checkpoint(model = model, step = step)
    start_epoch = 0
    if resume:
        checkpoint.resume(resume, conf['transfer_param'])
        if not reset_epoch:
            try:
                start_epoch = int(resume.split('_')[-1][:-3])+1
            except:
                start_epoch = step.value//(len(train_loader)*conf['batch_size'])
        else:
            step.clear()
    model = nn.DataParallel(model)
    model.to('cuda:0')
    #model放到device后再创建optimizer
    
    if stage==1:
        #optimizer = torch.optim.Adam(model.parameters(), betas = conf['adam_beta'], lr=conf['lr'], weight_decay=conf['weight_decay'])
        backbone = list(map(id, model.module.image_encoder.parameters()))
        others = list(filter(lambda p:id(p) not in backbone, model.parameters()))
        optimizer = torch.optim.Adam([{'params':model.module.image_encoder.parameters(), 'lr':conf['lr_backbone']},
                                      {'params':others, 'lr':conf['lr']}],
            betas = conf['adam_beta'], weight_decay=conf['weight_decay'] )
    elif stage==4: #用在RL时
        optimizer = torch.optim.Adam(get_parameters(model.module, ['word_decoder', 'sen_d_trans']),
                                     betas = conf['adam_beta'], lr=conf['lr'], weight_decay=conf['weight_decay'])
    elif stage==5: #用在RL时，训练baseline是看是否需要对image encoder做优化，还是只优化decoder也可以
        optimizer = torch.optim.Adam(get_parameters(model.module, ['decoder']),
                                     betas = conf['adam_beta'], lr=conf['lr'], weight_decay=conf['weight_decay'])
    elif stage==7: #用在RL时
        optimizer = torch.optim.Adam(get_parameters(model.module, ['topic_decoder']),
                                     betas = conf['adam_beta'], lr=conf['lr'], weight_decay=conf['weight_decay'])

    if stage!=1 or transfer:
        checkpoint.contents['best_metrics'] = {}
    if resume and not transfer:
        conf['model_dir'] = str(Path(resume).parent)
    if transfer:
        conf['model_dir'] = conf['model_dir']+'_t'+Path(resume).parent.name
    if stage!=1 or RL:
        conf['model_dir'] = conf['model_dir']+'/S%d_%s%s'%(stage, resume.split('/')[-1][:-3], '_RL' if RL else '')+\
                            ('' if not RL else str(conf['train_version']))
                            
    train_loss = Smoother(100)
    logger = Logger(conf['model_dir']+'/log%d.txt'%version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['model_dir'])
    
    Path(conf['model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch, conf['n_epoch']):
        print(conf['model_dir'], epoch)
        logger.log('new epoch', epoch)
        for (source, targets, debug) in tqdm(train_loader):
            step.forward(source.shape[0])
            
            pred, losses, pred_s, _,_ = compute_batch(model, source, targets, RL=RL, tokenizer=tokenizer, epoch=epoch)
            loss = torch.FloatTensor([0]).to('cuda:0')
            for x in losses:
                if x[:5]=='loss_':
                    loss += eval('conf["w_%s"]'%x[5:])*losses[x]
            losses['loss'] = loss
            train_loss.update(loss={x:losses[x].item() for x in losses})
            
            optimizer.zero_grad() #清空梯度
            loss.backward()
            optimizer.step() #优化一次
            if (RL and step.reach_cycle(100)) or (not RL and step.reach_cycle(1000)):
                logger.log(step.value, train_loss.value())
                logger.log(tokenizer.decode(targets['find'][0].cpu().numpy()))
                
                if pred.dim()==2: #RL返回的[B,L]
                    logger.log(tokenizer.decode(pred[0].cpu().numpy(), ['SOS', 'PAD'] if RL else ['SOS','EOS','PAD']))
                elif RL and pred.dim()==3: #RL返回[B,n_sen,L]
                    logger.log('|'.join([tokenizer.decode(pred[0,i].cpu().numpy(), ['SOS', 'PAD']) for i in range(pred.shape[1])]))
                elif pred.dim()==3: #[B,L,n_token]
                    logger.log(tokenizer.decode(torch.argmax(pred[0], 1).cpu().numpy()))
                if conf['MRG'] and not RL:
                    logger.log(tokenizer.decode(targets['find_s'][0,1:].cpu().numpy(), ['PAD']))
                    logger.log(tokenizer.decode(torch.argmax(pred_s[0], 1).cpu().numpy(), ['PAD']))
                    
                writer.add_scalars('train loss', train_loss.value(), step.value)
                writer.add_scalar('lr', optimizer.param_groups[0]["lr"], step.value)
            if step.reach_cycle(10000):
                model.eval()
                val_loss = validate(model, valid_loader, epoch=epoch) #dict
                logger.log('valid loss', step.value, val_loss)
                writer.add_scalars('valid loss', val_loss, step.value)
                if 'loss_g' not in val_loss:
                    valid_metric = 0
                else:
                    valid_metric = val_loss['loss_g']
                checkpoint.update(conf['model_dir']+'/model.pt', loss=-valid_metric)
                model.train()
            
            if RL and (conf['RL_first_only'] and step.reach_cycle(50000) or
                       not conf['RL_first_only'] and step.reach_cycle(8000)) \
            or not RL and epoch>0 and step.reach_cycle(100000): 
                model.eval()
                metrics = evaluate(model, valid_loader, tokenizer)
                logger.log('valid', step.value, metrics.value())
                writer.add_scalars('valid metric', metrics.value(), step.value)
                #writer.add_histogram('valid bleus', metrics.value(key='bleu',mean=False), step.value)
                checkpoint.update(conf['model_dir']+'/model.pt', metrics = metrics.value())
                model.train()
            
        if conf['n_epoch']<=40 or epoch%10==0 or (RL and ('MIMIC' in conf['dataset'] or epoch%10==0)):
            checkpoint.save(conf['model_dir']+'/model_%d.pt'%epoch)
            model.eval()
            metrics = evaluate(model, valid_loader, tokenizer, max_l=(60 if 'IU' in conf['dataset'] else 100))
            logger.log('valid', step.value, metrics.value())
            writer.add_scalars('valid metric', metrics.value(), step.value)
            #writer.add_histogram('valid bleus', metrics.value(key='bleu',mean=False), step.value)
            checkpoint.update(conf['model_dir']+'/model.pt', metrics = metrics.value())
            model.train()
        
    logger.close()
    writer.close()
    print(version)

def test(resume, split='test', max_l=100, beam_size=1, use_vq = False, output_topic=False, n=10000, shuffle=False,
         mode='eval', repeat_single=True, sample_lv=False):
    conf['output_dir'] = str(Path(resume).parent).replace('checkpoint', 'outputs')
    
    tokenizer = Tokenizer(conf['report_dir'], max_l, file_prefix=conf.get('token_prefix', ''), english=conf['english'])
    test_data = get_dataset(split, tokenizer, sample_lv)
    test_loader = DataLoader(test_data, batch_size=conf['valid_batch'], shuffle=shuffle, num_workers=8, drop_last=False)
    
    model = get_model()
    
    checkpoint = Checkpoint(model = model)
    if resume:
        checkpoint.resume(resume)
    model = nn.DataParallel(model)
    model.to('cuda:0')
    if mode=='eval':
        model.eval()
    if use_vq:
        vq = VQ(1024, 512)
        checkpoint = Checkpoint(vq=vq)
        checkpoint.resume(str(Path(resume).parent)+'/vq_loss.pt')
        vq.to('cuda:0')
        vq.eval()
    else:
        vq = None
    output_file = 'pred_random_%s%d%s%s_beam%d%s%s%s.txt'%(split, version, resume.split('/')[-1][:-3],
                                              '' if not output_topic else '_topic', beam_size,
                                              '' if mode=='eval' else '_t', '' if repeat_single else '_newsample',
                                              '' if not sample_lv else ('_lessview' if sample_lv is True else '_oneview'))
    metrics = evaluate(model, test_loader, tokenizer, output = output_file,
                       n=n, output_topic=output_topic, beam_size=beam_size, max_l=max_l, repeat_single=repeat_single)
    print(metrics.value())
    print(resume)
    return metrics

def encode_sentences(resume, mode='token', split = 'train', shuffle=False, n=-1, clean_VQ=False, load_VQ=None):
    #mode='token'生成token编号，'topic'生成具体topic
    conf['output_dir'] = str(Path(resume).parent).replace('checkpoint', 'outputs')
    print(conf['output_dir'])
    Path(conf['output_dir']).mkdir(exist_ok=True, parents=True)

    tokenizer = Tokenizer(conf['report_dir'], conf['max_l'], file_prefix=conf.get('token_prefix', ''), english=conf['english'])
    
    model = get_model()
    checkpoint = Checkpoint(model = model)
    checkpoint.resume(resume)
    if clean_VQ:
        model.VQ.set_empty_feature()
    if load_VQ:
        model.load_VQ(load_VQ)
    model = nn.DataParallel(model)
    model.to('cuda:0')
    model.eval()
    data = get_dataset(split, tokenizer)
    
    loader = DataLoader(data, batch_size=conf['valid_batch'], shuffle=shuffle, num_workers=8, drop_last=False)
    sen_tokens = []
    sen_topics = []
    topics = []
    tot = 0
    for (source, targets, debug) in tqdm(loader):
        tot += source.shape[0]
        if n>0 and tot>n:
            break
        targets = to_device(targets, 'cuda:0')
        if mode=='token':
            with torch.no_grad():
                tokens, beforeVQ, afterVQ, content = model.module.encode_sentence(targets['find_s'], targets['seg_s'], ret_VQ=True)
            for i in range(tokens.shape[0]): #B
                sen_tokens.append(tokens[i].cpu().numpy())
                capseg = targets['seg_s'][i]
                for j in range(tokens.shape[1]): #n_sen
                    if tokens[i,j]!=0:
                        sen_mask = ((capseg==j+1)|(capseg==j+101))
                        sen = tokenizer.decode((targets['find_s'][i]*sen_mask).detach().cpu().numpy())
                        topics.append((tokens[i,j].cpu().numpy(), beforeVQ[i,j].detach().cpu().numpy(), afterVQ[i,j].detach().cpu().numpy(), sen, content[i,j].detach().cpu().numpy()))  
        else:
            with torch.no_grad():
                _,_,sen_topic = model.module.encode_sentence(targets['find_s'], targets['seg_s'], ret_VQ=False) #[n_sen,B,D]
            sen_topic = sen_topic.cpu().numpy()
            for i in range(sen_topic.shape[0]):
                for j in range(sen_topic.shape[1]):
                    if np.sum(np.abs(sen_topic[i,j]-1/sen_topic.shape[2]))<0.01: #空句子的topic
                        continue
                    sen_topics.append(sen_topic[i,j])
    if mode=='token':
        with open(conf['output_dir']+'/sen_tokens_%s_%d.txt'%(split,n), 'w') as fp:
            for sen in sen_tokens:
                for x in sen:
                    fp.write(str(x)+' ')
                #fp.write(str(sen[1])+' ')
                #fp.write(tokenizer.decode(sen[2])+' ')
                #fp.write(str(sen[3]))
                fp.write('\n')
        topics = sorted(topics, key = lambda x:x[0])
        with open(conf['output_dir']+'/sen_with_tokens_%s_%d.txt'%(split,n), 'w') as fp:
            for x in topics:
                fp.write(str(x[0])+'\t'+x[3]) #token及对应句子
                fp.write('\n')
    else:
        np.savetxt(conf['output_dir']+'/sen_topics_%s_%d.txt'%(split,n), np.asarray(sen_topics), fmt='%.4f')
        return
    if n==-1:
        return
    l = len(topics) #topics是一堆句子的list。每个句子是一个tuple，包含一些内容。
    sentences = {int(x[0]):[] for x in topics} #{token_id: [(sen, id),]}。其中id是此句子在topics中的index
    contents = {int(x[0]):[] for x in topics} #{token_id: [content,]}。

    for i,x in enumerate(topics):
        sentences[int(x[0])].append((x[3],i))
        contents[int(x[0])].append(x[4])
        #sentences[i][j]和contents[i][j]对应的是同一个句子
    return
    
    unique_VQ = {int(x[0]):x[2].tolist() for x in topics}
    temp = np.stack([x[1] for x in topics]+[np.array(unique_VQ[x]) for x in unique_VQ], axis=0)
    perplexity = 5
    points = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, init='pca',
                  random_state=10).fit_transform(temp) #前部分是l个句子topic的坐标、后部分是一些VQ topic的坐标
    show_points([points[:l], points[l:]], 
                ['blue', 'green'], [0.1, 1], conf['output_dir']+'/images/topic_%s_p%d.svg'%(resume.split('/')[-1][:-3], perplexity),
                labels = [('blue', 'original'), ('green','vector-quantized')], loc='upper left', group=True)
    return
    Path(conf['output_dir']+'/images').mkdir(exist_ok=True)
    p = {}
    '''
    for x in contents:
        if len(contents[x])<=5:
            continue
        for perplexity in [5,10,15]:
            for n_iter in [500,1000,2000]:
                p[x] = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='pca',
                          random_state=10).fit_transform(contents[x])
                show_points([p[x]], ['blue'], [0.1], conf['output_dir']+'/images/content_%s_t%d_%d_%d.svg'%(resume.split('/')[-1][:-3], x, perplexity, n_iter),
                            [[y[0] for y in sentences[x]]])
    '''
    with open(conf['output_dir']+'/sentence_topic.txt', 'w') as fp:
        tot = 0
        for x in unique_VQ:
            fp.write(str(x)+' '+str(points[l+tot])+'\n')
            for i,s in enumerate(sentences[x]):
                fp.write(s[0]+' '+str(points[s[1]])) #句子，以及topic其在图上的坐标
                #if x in p:
                #    fp.write(' '+str(p[x][i])) #此句子的content，在自己所属的VQ topic的句子的content分布图上的坐标
                fp.write('\n')
            tot += 1


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='train')
parser.add_argument('-version', type=float)
parser.add_argument('-train_version', type=int, default=0)
parser.add_argument('-stage', type=int, default=1)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-resume', type=str, default='')
parser.add_argument('-reset_epoch', action='store_true')
parser.add_argument('-suffix', type=str, default='')
args = parser.parse_args()

if args.mode!='x':
    version = args.version
    if version==int(version):
        version = int(version)
    train_version = args.train_version #0
    set_seed(args.seed) 
    print('version', version, train_version)
    conf = Config(version, train_version, seed=args.seed)
    
    if args.resume=='':
        args.resume = None
    if args.train_version>0: #Reinforcement learning
        args.stage=conf['stage']
        #否则，stage应该是1或2，不使用强化学习
    train(args.resume, stage=args.stage, RL=(args.train_version>0), reset_epoch=args.reset_epoch)
else:
    version = 1
    train_version = 0
    seed = 0
    set_seed(seed) #同一类型GPU是没问题的。但是用V100和2080跑出来不一样！
    print('version', version, train_version)
    conf = Config(version, train_version, seed=seed)
    dir = str(version)+ ('' if seed==0 else '_%d'%seed)
    
    train()
    '''
    #encode_sentences('checkpoint/%s/model_bleu_h.pt'%dir, split='train', n=-1)
    #prepare_topic_token_cider(conf['output_dir']+'/sen_tokens_train_-1.txt', conf['model_dir'])
    train_version = 25
    set_seed(seed)
    print('begin reinforcement finetuning of the topic decoder', version, train_version)
    conf = Config(version, train_version, seed=seed)
    train('%s/model_bleu_h.pt'%conf['model_dir'], stage=conf['stage'], RL=True, reset_epoch=True)
    
    train_version = 52
    set_seed(seed)
    clear_scorer()
    conf = Config(version, train_version, seed=seed)
    train('%s/S7_model_bleu_h_RL21/model_bleu_h.pt'%conf['model_dir'], stage=conf['stage'], RL=True, reset_epoch=True)
    '''
    
    '''
    train_version = 52
    set_seed(seed)
    clear_scorer()
    conf = Config(version, train_version, seed=seed)
    train('%s/model_bleu_h.pt'%conf['model_dir'], stage=conf['stage'], RL=True, reset_epoch=True)
    
    encode_sentences('checkpoint/%s/S4_model_bleu_h_RL%d/model_bleu_h.pt'%(dir, train_version), split='train', n=-1)
    prepare_topic_token_cider(conf['output_dir']+'/sen_tokens_train_-1.txt', 'checkpoint/%s/S4_model_bleu_h_RL%d'%(dir,train_version))
    
    train_version = 21
    clear_scorer()
    set_seed(seed)
    print('begin reinforcement finetuning of the topic decoder', version, train_version)
    conf = Config(version, train_version, seed=seed)
    train('%s/S4_model_bleu_h_RL52/model_bleu_h.pt'%conf['model_dir'], stage=conf['stage'], RL=True, reset_epoch=True)
    '''
