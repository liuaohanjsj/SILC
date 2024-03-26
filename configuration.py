# -*- coding: utf-8 -*-
import torch
import numpy as np

class Config(dict):
    def version_config(self, version, train_v):
        hp = {
              1:{'dataset':'IU', 'lr':3e-5, 'vbackbone':'resnet101', 'batch':32, 'vb':32, 'cs':True, 'MRG':True, 'VQ':1024, 'normal_gen':False, 'normal_c':False, 'nes':True},
              1.1:{'dataset':'IU_l',},
              1.2:{'dataset':'IU_f'},
              1.3:{'mv_fuse':'max'},
              1.4:{'nes':True, 'transfer_param':['image_encoder'], 'n_epoch':200},
              
              2:{'dataset':'MIMIC', 'lr':3e-5, 'vbackbone':'resnet101', 'batch':32, 'vb':32, 'cs':True, 'MRG':True, 'VQ':1024, 'n_epoch':8, 'normal_gen':False, 'normal_c':False, 'nes':True},
              }
        for v in hp:
            if not isinstance(v,int):
                for x in hp[int(v)]:
                    if x not in hp[v]:
                        hp[v][x] = hp[int(v)][x]
        
        self['lr'] = hp[version].get('lr', 1e-4)
        self['lr_backbone'] = hp[version].get('lr_backbone', self['lr'])
        self['w_g'] = hp[version].get('w_g', 1)
        self['w_c'] = hp[version].get('w_c', 0.1)
        self['w_silc'] = hp[version].get('w_silc', 0.3)
        self['w_g_s'] = hp[version].get('w_g_s', 1)
        self['w_g_sen'] = hp[version].get('w_g_sen', 1)
        self['w_topic'] = 1
        self['w_vq'] = 1
        self['w_used'] = 0

        self['w_RL'] = 1
        self['w_RL_first'] = 1
        self['ignore_loss'] = hp[version].get('ignore_loss', [])
        for x in self['ignore_loss']:
            self['w_'+x] = 0
        
        self['vbackbone'] = hp[version].get('vbackbone', 'resnet101')
        self['minibatch'] = hp[version].get('batch', 4)
        self['batch_size'] = self['minibatch'] * torch.cuda.device_count()
        self['valid_batch'] = hp[version].get('vb', 4) * torch.cuda.device_count()
        
        self['n_content'] = hp[version].get('nc', 49)
        self['real_n_content'] = hp[version].get('real_nc', None)
        self['SILC'] = hp[version].get('cs', False)
        self['MRG'] = hp[version].get('MRG', False)
        self['text_d'] = hp[version].get('text_d', 512)
        self['VQ'] = hp[version].get('VQ', False)
        self['word_decoder_layer'] = hp[version].get('wdl', 6)
        self['topic_decoder_layer'] = hp[version].get('tdl', 6) 
        self['decoder_layer'] = hp[version].get('dl', 6)
        
        self['no_empty_sentence'] = hp[version].get('nes', False)

        self['delimiter'] = hp[version].get('delimiter', None)

        self['weight_decay'] = hp[version].get('weight_decay', 0)
        self['augment'] = hp[version].get('augment', False)
        self['sample_lv'] = hp[version].get('sample_lv', False)
        
        self['n_fuse_layer'] = hp[version].get('n_fuse_layer', 3)
        
        self['normal_gen'] = hp[version].get('normal_gen', True)
        self['normal_c'] = hp[version].get('normal_c', True)
        if self['w_c']==0:
            self['normal_c'] = False
        self['image_size'] = hp[version].get('image_size', 224)
        self['multiview_fuse'] = hp[version].get('mv_fuse', 'mean')
        self['image_pad'] = hp[version].get('image_pad', 'blank')
        
        self['max_l'] = hp[version].get('max_l', 100)
        self['dataset'] = hp[version]['dataset']
        self['transfer_param'] = hp[version].get('transfer_param', None)
        self['pretrain'] = hp[version].get('pretrain', True)
        
    
        if 'MIMIC' in self['dataset']:
            self['report_dir'] = './preprocess/mimic'
        elif 'IU' in self['dataset']:
            self['report_dir'] = './preprocess/IU'
        if self['dataset']=='MIMIC':
            self['data_suffix']='_R2Gen'
            self['token_prefix'] = 'R2Gen_'
            self['n_views'] = 1
        elif self['dataset']=='MIMIC_m':
            self['data_suffix']='_multiview'
            self['token_prefix'] = 'R2Gen_'
            self['n_views'] = 3
        elif self['dataset']=='IU':
            self['data_suffix']='_R2Gen'
            self['token_prefix'] = 'R2Gen_'
            self['n_views'] = 2
        elif self['dataset']=='IU_f':
            self['data_suffix']='_frontal'
            self['token_prefix'] = 'R2Gen_'
            self['n_views'] = 1
        elif self['dataset']=='IU_l':
            self['data_suffix']='_lateral'
            self['token_prefix'] = 'R2Gen_'
            self['n_views'] = 1
        elif self['dataset']=='IU_t':
            self['data_suffix']='_R2Gen'
            self['token_prefix'] = 'mimic_'
            self['n_views'] = 2
        else:
            self['data_suffix']=''
        if 'MIMIC' in self['dataset'] or self['dataset']=='IU_t':
            self['n_token'] = 5500
        else:
            self['n_token'] = 1500
        #if 'MIMIC' in self['dataset'] or 'IU' in self['dataset']:
        self['english'] = True
        
        
        self['n_epoch'] = hp[version].get('n_epoch', 35 if 'IU' not in self['dataset'] else 300) 
        self['adam_beta'] = hp[version].get('beta', (0.9,0.98))
        
        train_hp = {0:{},
                    1:{'batch': 32, 'stage':7, 'lr': 1e-6, 'RL_first':True, 'RL_first_only':True, 'RL_first_direct':True, 'cider_w':1, 'cider_n':1, 'bleu_w':0, 'bleu_n':0, 'n_epoch':5 if 'MIMIC' in self['dataset'] else 100},
                    2:{'batch': 8, 'stage':4, 'lr': 1e-6, 'RL_first':False, 'cider_w':0, 'bleu_w':1, 'bleu_n':(1,2,3,4), 'n_epoch':1 if 'MIMIC' in self['dataset'] else 50},
                    3:{'batch': 8, 'stage':4, 'lr': 1e-6, 'RL_first':False, 'cider_w':1, 'bleu_w':1, 'bleu_n':(1,2,3,4), 'n_epoch':1 if 'MIMIC' in self['dataset'] else 50},
                    4:{'batch': 8, 'stage':4, 'lr': 1e-6, 'RL_first':False, 'cider_w':1, 'bleu_w':0, 'cider_n':4, 'n_epoch':1 if 'MIMIC' in self['dataset'] else 50},
                    }
        self['RL_first'] = train_hp[train_v].get('RL_first', False)
        self['RL_first_only'] = train_hp[train_v].get('RL_first_only', False)
        self['RL_first_direct'] = train_hp[train_v].get('RL_first_direct', False)
        
        self['cider_weight'] = train_hp[train_v].get('cider_w', 0)
        self['bleu_weight'] = train_hp[train_v].get('bleu_w', 1)
        self['bleu_n'] = train_hp[train_v].get('bleu_n', 4)
        self['cider_n'] = train_hp[train_v].get('cider_n', 4)
        self['separate_corpus'] = train_hp[train_v].get('separate_corpus', False)
        self['no_corpus'] = train_hp[train_v].get('no_corpus', False)
        
        self['stage'] = train_hp[train_v].get('stage', 1)
        if 'n_epoch' in train_hp[train_v]:
            self['n_epoch'] = train_hp[train_v]['n_epoch']
        if 'beta' in train_hp[train_v]:
            self['adam_beta'] = train_hp[train_v]['beta']
        if train_v:
            self['batch_size'] = train_hp[train_v].get('batch', self['minibatch']) * torch.cuda.device_count()
            self['lr'] = train_hp[train_v].get('lr', 5e-6)
        
    def __init__(self, version = 0, train_version = 0, seed=0):
        print('gpu num', torch.cuda.device_count())
        self['version'] = version
        self['train_version'] = train_version
        self['model_dir'] = './checkpoint/'+str(version)
        if seed>0:
            self['model_dir'] += '_%d'%seed
        self['output_dir'] = './outputs/'+str(version)
        
        self['pad_id'] = 0
        self['sos_id'] = 1
        self['beam_size'] = 1
        self.version_config(version, train_version)
        print('gpu num', torch.cuda.device_count())
        print('batch', self['batch_size'])

