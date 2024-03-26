# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:48:50 2022

@author: Admin
"""
import torch
import torch.nn as nn
from pathlib import Path
import time
import json
import numpy as np
import random
import cv2
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import traceback
from torch.nn.functional import normalize

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from cider.pyciderevalcap import CiderD

def generate_group_mask(attn_group, n_head):
    """
    attn_group: [B,L]，第i个句子开头位置值为i+100，其他位置为i。句子序号从1开始，不会超过100
    返回的attn_mask每个group只能看到自己group中的内容
    """
    attn_mask = torch.full((attn_group.shape[0], attn_group.shape[1], attn_group.shape[1]),float(0)).to(attn_group.device) #[B, L, L]
    n_sen = torch.max(attn_group).long()-100
    sen_i = torch.arange(1, n_sen+1).unsqueeze(1).unsqueeze(2).to(attn_group.device) #[n_sen,1,1]
    group_mask = (attn_group==sen_i)|(attn_group==sen_i+100) #[n_sen,B,L]。[0,3,10]表示第3个报告的10位置是否属于第1个句子
    index = group_mask.nonzero(as_tuple=True) #([X],[X],[X])，分别为n_sen维度、B维度、L维度。比如第3个报告的10位置属于第1个句子，则会有一个(0,3,10)三元组
    #对index的每个三元组进行操作。如(0,3,10)，则attn_mask[3,10]，即第3个报告的第10号位置可以关注到的mask（即第1个句子的mask，长为L），应该是group_mask[0,3]，也就是第1个句子的mask
    attn_mask[index[1:3]] = group_mask[index[0:2]].float()-1 #可以关注到的弄成0，不能关注的-1
    attn_mask[attn_mask.nonzero(as_tuple=True)] = float('-inf') #不能关注的-1弄成-inf
    #不能直接repeat。repeat是123123，但我想要112233
    attn_mask = attn_mask.unsqueeze(1).repeat(1,n_head,1,1).reshape(-1,attn_group.shape[1], attn_group.shape[1]) #[B*nhead, L, L]
    return attn_mask

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameters(model, pars):
    ret = [{'params': getattr(model, x).parameters()} for x in pars]
    print(ret)
    return ret

def show_points(points, colors, alphas, output_path, s = 100, figsize=15, texts=None,
                labels = None, loc='lower right', group=False):
    """
    points: [[N,2]] if group else [N,2]。colors、alphas何其保持一致
    labels: [(color, label),]
    """
    
    plt.figure(figsize=(figsize,figsize))
    if group:
        for g in range(len(points)): #group
            plt.scatter(points[g][:,0], points[g][:,1], s=s, color=colors[g], alpha = alphas[g])
        #if texts is not None:
        #    for i in range(len(points)):
         #       plt.annotate(texts[i], points[i], xytext=None, arrowprops=None)
    else:
        plt.scatter(points[:,0], points[:,1], s=s, color=colors, alpha = alphas)
        if texts is not None:
            for i in range(len(points)):
                plt.annotate(texts[i], points[i], xytext=None, arrowprops=None)
    if labels is not None:
        classes = [plt.scatter([], [], s=200, color=x[0]) for x in labels]
        plt.legend(classes, [x[1] for x in labels], loc=loc, fontsize=20)
        
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def output_tensor(x, precision=3):
    print(np.round(x.detach().cpu().numpy(), precision))

def cosine_sim(a, b):
    return torch.sum(normalize(a, dim=-1) * normalize(b, dim=-1), dim=-1)

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    # Set up scorers
    scorers = [
        (Bleu(4), ['BLEU_%d'%i for i in range(1, 5)]+['ratio']),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        #(CiderD(df='cider/data/mimic.p', sigma=20.0), 'CiderD')
        (CiderD(), 'CiderD'),
        #(Cider(), 'Cider') #和CiderD结果一样
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        if method=='CiderD':
            temp = [{'image_id':i, 'caption':res[i]} for i in res]
            score, scores = scorer.compute_score(gts, temp)
        else:
            score, scores = scorer.compute_score(gts, res)
        
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def F1_score(scores, threshold=0.5):
    gt = np.greater(np.array([x[0] for x in scores]), threshold)
    pred = np.greater(np.array([x[1] for x in scores]), threshold)
    recall = np.sum(gt*pred)/np.sum(gt)
    precision = np.sum(gt*pred)/np.sum(pred)
    return 2*recall*precision/(recall+precision)

def draw_roc(scores, label = '', ax = None, axins = None, draw = True, CI = '',
             place = 0, color = None, linestyle = None, newlineCI = False):
    """
    scores: [(gt,pred),]
    """
    scores = copy.deepcopy(scores)
    scores.sort(key = lambda x: x[1], reverse = True)
    #scores.sort(key = lambda x: (x[1], -x[0]), reverse = True)
    #print(len(scores))
    ap = sum([x[0] for x in scores]) #actual positive
    an = len(scores)-ap #actual negative
    if ap==0 or an==0:
        return -1
    x, y = [0], [0]
    tp, fp, ret, i = 0, 0, 0, 0
    while i<len(scores):
        #print(i)
        memtp = tp
        memfp = fp
        now_score = scores[i][1]
        while i<len(scores) and scores[i][1]==now_score:
            if scores[i][0]:
                tp += 1
            else:
                fp += 1
            i += 1
        ret += (fp-memfp)/an*(memtp+tp)/2/ap #如果使用这种方法，就可能出现斜线，从而可以计算梯形面积
        #无论[(1,0.6), (0,0.6)]还是[(0,0.6), (1,0.6)]结果都是0.5
        x.append(fp/an) #这里x不是specificity(tn/an)，而是1-specificity，即fp/an
        y.append(tp/ap)
    '''
    if draw:
        sentence = '\'%'+'%ds'%place+'\'%label' 
        if newlineCI:
            label = eval(sentence) + ('AUC=%.3f'%ret).center(14) + '\n' + CI
        else:
            label = eval(sentence)+' AUC=%.3f'%ret + CI
        
        if ax is None:
            plt.plot(x, y, label=label, color=color, linestyle=linestyle)
        else:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle)
        if axins is not None:
            axins.plot(x, y)
    '''
    return round(ret, 4)


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, np.ndarray):
        data = to_device(torch.from_numpy(data), device)
    elif isinstance(data, tuple):
        data = tuple(to_device(item,device) for item in data)
    elif isinstance(data, list):
        data = list(to_device(item,device) for item in data)
    elif isinstance(data, dict):
        data = dict((k,to_device(v,device)) for k,v in data.items())
    else:
        raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.', type(data), data)
    return data


class Smoother():
    def __init__(self, window):
        self.window = window
        #之前只是一个数的smoother，现在因为loss可能有很多个，弄成了多个数的smoother
        #正好tensorboard add_scalars可以支持加dict。我的Logger也做相应修改 
        self.num = {}
        self.sum = {}
    def update(self, **kwargs):
        """
        为了调用方便一致，支持kwargs中有值为None的，会被忽略
        kwargs中一些值甚至可以为dict，也就是再套一层。
        示例: update(a=1, b=2, c={'c':1, 'd':3})，相当于update(a=1, b=2, c=1, d=3)
        如果值为参数的None的话忽略
        """
        values = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    values[x] = kwargs[key][x] #有可能会覆盖，如update(a=1,b={'a':2})
            else:
                values[key] = kwargs[key]
        for key in values:
            if values[key] is None:
                continue
            if key not in self.num:
                self.num[key] = []
                self.sum[key] = 0
            self.num[key].append(values[key])
            self.sum[key] += values[key]

            if len(self.num[key])>self.window:
                self.sum[key] -= self.num[key][-self.window-1]
            if len(self.num[key])>self.window*2:
                self.clear(key)
        pass
    def clear(self, key):
        del self.num[key][:-self.window]
    def value(self, key = None, mean=True):
        if mean:
            if key is None:
                return {key: self.sum[key] / min(len(self.num[key]),self.window) for key in self.num}
            return self.sum[key] / min(len(self.num[key]),self.window)
        if key is None:
            return {key: np.array(self.num[key]) for key in self.num}
        return np.array(self.sum[key])
    def keys(self):
        return self.num.keys()

class Step():
    def __init__(self):
        self.step = 0
        self.round = {}
    def clear(self):
        self.step = 0
        self.round = {}
    def forward(self, x):
        self.step += x
    def reach_cycle(self, mod, ignore_zero = True):
        now = self.step // mod
        if now==0 and ignore_zero:
            return False
        if mod not in self.round or self.round[mod]!=now: #新过了一个或多个cycle
            self.round[mod] = now
            return True
        return False
    def state_dict(self):
        return {'step': self.step, 'round':self.round}
    def load_state_dict(self, state, strict=False):
        self.step = state['step']
        self.round = state['round']
    @property
    def value(self):
        return self.step

class Logger():
    def __init__(self, file_name, mode = 'w', buffer = 100):
        (Path(file_name).parent).mkdir(exist_ok = True, parents = True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer
    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y)+':'+str(x[y])+' ')
            else:
                self.fp.write(str(x)+' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt>=self.buffer or time.time()-self.stamp>5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass
    def close(self):
        self.fp.close()

class Checkpoint():
    def __init__(self, **contents):
        """
        contents每个元素都需要有load_state_dict()方法
        """
        self.contents = contents
        self.contents['best_metrics'] = {}
    def update(self, file_path, logger = None, **kwargs):
        """
        根据metrics选择性地更新保存当前最好模型
        metrics: {metric_name: float 或 None}，越大越好。None的话忽略
        file_path: 保存文件名，*.pt
        """
        metrics = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    metrics[x] = kwargs[key][x] #有可能会覆盖，如update(a=1,b={'a':2})
            else:
                metrics[key] = kwargs[key]
        for metric in metrics:
            if metrics[metric] is None:
                continue
            if metric not in self.contents['best_metrics'] or metrics[metric]>self.contents['best_metrics'][metric]:
                self.contents['best_metrics'][metric] = metrics[metric]
                torch.save(self._get_contents(), file_path[:-3]+'_%s.pt'%metric)
                #torch.save(self.contents['optimizer'].state_dict(), file_path[:-3]+'_%s.pt'%metric)
                print('new best metric', metric, metrics[metric])
                if logger is not None:
                    logger.log('new best metric', metric, metrics[metric])
        pass
    def _get_contents(self):
        ret = {}
        for key in self.contents:
            if isinstance(self.contents[key], nn.DataParallel):
                ret[key] = self.contents[key].module.state_dict()
            elif hasattr(self.contents[key], 'state_dict'):
                ret[key] = self.contents[key].state_dict()
            else:
                ret[key] = self.contents[key]
        return ret
    def save(self, file_path):
        torch.save(self._get_contents(), file_path)
        
    def resume(self, file_path, param = None):
        memory = torch.load(file_path)
        self.contents['best_metrics'] = memory.pop('best_metrics')
        for key in memory:
            if key not in self.contents:
                print('loaded key not in contents:', key)
                continue
            try:
                if isinstance(self.contents[key], nn.Module) and param is not None:
                    target_params = {}
                    for x in memory[key]:
                        #if x in param:
                        for p in param:
                            if x[:len(p)]==p: #only need the prefix of the param name fits the target
                                target_params[x] = memory[key][x]
                    memory[key] = target_params
                    print('target params', target_params.keys())
                if isinstance(self.contents[key], nn.DataParallel):
                    self.contents[key].module.load_state_dict(memory[key], strict=False if param is not None else True)
                else:
                    self.contents[key].load_state_dict(memory[key], strict=False if param is not None else True)
            except (Exception, BaseException) as e:
                exstr = traceback.format_exc()
                print(exstr)
        pass

class SaveOutput():
    def __init__(self, target_layer):
        target_layer.register_forward_hook(self.save_output)
    def save_output(self, module, input, output):
        self.output = output

class OutputGradient():
    def __init__(self, target_layer, reshape_transform = None):
        self.reshape_transform = reshape_transform
        target_layer.register_forward_hook(self.save_output_gradient)
        target_layer.register_forward_hook(self.save_output)
        
    def save_output_gradient(self, module, input, output):
        #print('output required grad', output.requires_grad)
        def store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradient = grad.cpu().detach()
            #print(torch.sum(self.gradient))
        output.register_hook(store_grad)
    def save_output(self, module, input, output):
        if self.reshape_transform is not None:
            output = self.reshape_transform(output)
        self.output = output.cpu().detach()

def align_img(img, m):
    """
    img: 2D tensor
    m: 2D tensor，已以序列为单位归一化到[0,1]。因为在整个序列中归一化，所以最大值不一定是1
    """
    img = img.numpy()
    img = np.uint8((img-np.min(img))/(np.max(img)-np.min(img))*255)
    color_img = np.stack((img,img,img), axis=2)
    m = m.numpy()
    m = cv2.resize(m, (img.shape[1], img.shape[0]))
    heat_img = cv2.applyColorMap(np.uint8(m*255), cv2.COLORMAP_JET)
    heat_img = cv2.addWeighted(color_img, 0.7, heat_img, 0.3, 0)
    #cv2.imwrite(path, heat_img)
    return heat_img

class GradCAM():
    def __init__(self, source, target_layer, output_path, reshape_transform = None, n_slice=32):
        """
        source: tensor [B, n_slice, 3, H, W]
        target_layer: 网络层，输出为[B*n_slice, D, H, W]
        output_path: list, 长B
        """
        self.source = source
        self.saver = OutputGradient(target_layer)
        self.output_path = output_path
        for path in output_path:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.reshape_transform = reshape_transform
        self.n_slice = n_slice
        
    def get_map(self):
        #print(self.saver.gradient.shape)
        #print(self.saver.gradient.shape, self.saver.output.shape)
        if len(self.saver.gradient.shape)==4: #[B*n_slice, D, H, W]
            gc = torch.mean(self.saver.gradient, dim=(2,3), keepdim=True)*self.saver.output #[B*n_slice, D, H, W]
            #gc = torch.mean(torch.clamp(gc, min=0), dim=1) #[B*n_slice, H, W] #目前用这个
            gc = torch.clamp(torch.mean(gc, dim=1), min=0) #gradcam论文中是先平均再取绝对值。结果明显更集中。
            gc = self.reshape_transform(gc) #[B, n_slice, H, W]
        elif len(self.saver.gradient.shape)==3: #[B, n_topic, D]
            gc = torch.mean(self.saver.gradient, dim=2, keepdim=True)*self.saver.output #[B, n_topic, D]
            gc = torch.mean(torch.clamp(gc, min=0), dim=2) #[B, n_topic]
        
        #print(gc.shape)
        return gc
    def output_map(self, name, series = None):
        """
        name: list，要输出的每个序列的图像名字。长为B
        series: list，要输出batch中的哪些序列
        """
        if series is None:
            series = range(self.source.shape[0])
        assert self.source.shape[0]==len(name)
        gc = self.get_map()
        for b in series:
            maxn = torch.max(gc[b])
            temp = gc[b]/maxn
            if len(self.saver.gradient.shape)==4:
                prefix = ''
                output_images = []
                for i in range(self.n_slice):
                    img = self.source[b,i,0].cpu() #第b个序列第i个图像，第0个通道。
                    img = align_img(img, temp[i])
                    output_images.append(img)
                output_image = np.array(output_images) #[32,224,224,3]
                if self.n_slice==32: #CT
                    output_image = np.reshape(output_image, (4,8,224,224,3))
                    output_image = np.transpose(output_image, (0,2,1,3,4))
                    output_image = np.reshape(output_image, (4*224, 8*224, 3))
                elif self.n_slice==1:#MIMIC
                    output_image = output_image[0]
            else:
                prefix = 'topic'
                output_image = temp.numpy()[np.newaxis]*255 #[1,n_topic]
                output_image = cv2.resize(output_image, (output_image.shape[1]*30, 100), interpolation=cv2.INTER_NEAREST)
                output_image = np.concatenate((output_image, np.ones((20, output_image.shape[1]))*255), axis=0)
                for i in range(temp.shape[0]):
                    cv2.putText(output_image, str(i), (i*30+10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            cv2.imwrite('%s/%s%s.png'%(self.output_path[b], prefix, name[b]), output_image)

