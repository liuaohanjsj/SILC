# -*- coding: utf-8 -*-
from torchvision import models
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from dataset import Tokenizer
import mytransformers
from utils import output_tensor, generate_group_mask

class VQ(nn.Module):
    def __init__(self, n, d, softmax=True):
        super().__init__()
        self.n = n
        self.d = d
        self.vectors = nn.Embedding(n, d)
        self.set_empty_feature(0)
        if softmax:
            self.vectors.weight = nn.parameter.Parameter(nn.functional.softmax(self.vectors.weight, dim=1))
        self.upper = torch.tensor([self.n-1], dtype=torch.int64)
    def update(self, idx):
        """
        Keep track of the recently used vectors
        idx: [x], values in 0-n
        """
        if not hasattr(self, 'pool'):
            self.pool = torch.full((0,self.n), 0, dtype=torch.bool).to(idx.device)
        if self.training:
            temp = torch.full((1,self.n), 0, dtype=torch.bool).to(idx.device)
            temp[0,idx] = 1
            self.pool = torch.cat([self.pool, temp], dim=0) #[l+1,n]
            if self.pool.shape[0]>100:
                self.pool = self.pool[1:]
        used = torch.tensor([0]).to(idx.device) if self.pool.shape[0]==0 else torch.sum(torch.max(self.pool, dim=0).values)
        #print('used vector', used)
        return used
    def forward(self, v):
        """
        v: [..., d], the features to be quantized. Find their nearest neighbors in the codebook and return them.
        Note this VQ module is different from the VQ in VQVAE.
        """
        vectors = self.vectors.weight #[n,d]
        """
        The following two lines is the naive way to find the target indexes. 
        However, this sometimes causes the empty feature (see self.set_empty_feature) be selected
        for some real features in v. We do not want this happen because we want the empty feature be constant and not optimized.
        #diff = torch.sum(torch.square(v.unsqueeze(-2)-vectors), dim=-1) #[..., n]
        #idx = torch.argmin(diff, dim=-1) #[...]
        """
        #Use the following method to prevent the above situation:
        diff = torch.sum(torch.square(v.unsqueeze(-2)-vectors[1:]), dim=-1) #[..., n-1]. ignore the first empty feature in the codebook
        idx = torch.argmin(diff, dim=-1)+1 #[...]
        mask = torch.ne(v, 1/self.d) #[..., d]
        mask = torch.max(mask, dim=-1).values #[...], 1 for real features in v and 0 for empty features
        idx = torch.where(mask, idx, 0)
        
        used = self.update(idx.reshape(-1))
        
        onehot = nn.functional.one_hot(idx, num_classes=self.n).unsqueeze(-2).float() #[..., 1, n]
        out = torch.matmul(onehot, vectors).squeeze(-2) #[...,d]
        loss = torch.mean(torch.square(out - v.detach()))
        if self.training:
            v_ignore_eos = torch.where(v==1/self.d, torch.tensor([-1000.0], dtype=torch.float32).to(v.device), v).detach() #把空句子的1/32向量忽视掉，否则几乎所有随机向量都是和它最接近，都优化到它上去了
            diff2 = torch.mean(torch.square(v_ignore_eos.unsqueeze(-2)-vectors), dim=-1) #[..., n]
            min_diff_4_each_embed = torch.min(diff2.reshape(-1, self.n), dim=0).values #[n]，每个embed向量找v中和自己最小的差距
            unused = torch.nonzero(~torch.max(self.pool, dim=0).values)
            loss += torch.mean(min_diff_4_each_embed[unused]) #push the recently unused features towards their nearest neighbors in the input
            out = out.detach()
        return out, idx, loss, used
    def get_feature(self, idx):
        #v: [...]
        idx = torch.minimum(idx, self.upper.to(idx.device))
        onehot = nn.functional.one_hot(idx, num_classes=self.n).unsqueeze(-2).float() #[..., 1, n]
        out = torch.matmul(onehot, self.vectors.weight).squeeze(-2) #[...,d]
        return out #[...,d]
    def set_empty_feature(self, idx=0):
        """
        As the sentence number in each report may be different, empty sentences were padded after each report.
        The empty sentences has a topic value of [1/N, 1/N, ..., 1/N], which we refer to as the "empty feature".
        Make a fixed empty feature in the codebook. It will not be optimized.
        """
        w = self.vectors.weight.data #[n,d]
        w[idx] = 1/w.shape[1]
        self.vectors.weight = nn.parameter.Parameter(w)

class ReportGenerator_contrast(nn.Module):
    def __init__(self, n_token, max_l, pad_id = 0, sos_id = 1,
                 vbackbone='resnet101', SILC = False, MRG=False, 
                 normal_gen=True, normal_c=True, 
                 n_content=49, real_n_content=None, 
                 text_d = 512, use_vq=False, 
                 word_decoder_layer=6, topic_decoder_layer=6, decoder_layer=6, 
                 n_fuse_layer=3, multiview_fuse=None, pretrain=True):
        """
        The main model of our work. This model contains both the SILC and the MRG module.
        """
        super().__init__()
        #self.image_encoder and self.use combined is the Image encoder in the paper
        self.image_encoder = ImageEncoder(fuse=multiview_fuse, backbone=vbackbone, output_d=text_d, pretrain=pretrain)
        self.fuse = TransformerEncoder(n_fuse_layer, input_l=n_content, d=text_d, n_head=8)
        if real_n_content is not None: #Use real_n_content to do ablation on the hyperparameter N
            n_content = real_n_content
        
        if normal_gen: #The baseline generation model
            self.decoder = ReportDecoder(n_token=n_token, max_l=max_l, pad_id=pad_id, sos_id=sos_id, input_l=n_content, 
                                         d=text_d, n_layer=decoder_layer)
        if normal_c: #normal contrastive, CLIP
            self.text_encoder = TransformerEncoder(6, input_l=max_l, n_token=n_token, d=text_d)
            self.image_projection = nn.Linear(text_d, text_d)
            self.text_projection = nn.Linear(text_d, text_d)
            self.t = nn.parameter.Parameter(torch.FloatTensor([1]), requires_grad=True)
        
        if SILC: #SILC module components:
            self.sentence_encoder = TransformerEncoder(6, input_l=max_l, n_token=n_token, d=text_d)
            self.find_sen_topic = nn.Sequential(nn.Linear(text_d, n_content, bias=False), nn.Softmax(dim=-1))
            self.find_sen_content = nn.Linear(text_d, text_d, bias=False)
            self.find_image_content = nn.Linear(text_d, text_d)
            self.t_s = nn.parameter.Parameter(torch.FloatTensor([1]), requires_grad=True)
        if MRG: #MRG module components: 
            assert use_vq and SILC, "MRG is based on SILC, and should have a VQ component"
            self.sen_d_trans = nn.Linear(n_content, text_d, bias=False)
            self.topic_decoder = ReportDecoder(n_token=use_vq+1, max_l=30, pad_id=-1, sos_id=use_vq, input_l=n_content, 
                                               d=text_d, n_layer=topic_decoder_layer)
            self.word_decoder = ReportDecoder(n_token=n_token, max_l=max_l, pad_id=pad_id, sos_id=sos_id, input_l=n_content, 
                                              d=text_d, n_layer=word_decoder_layer, nhead=8)
            self.n_vectors = use_vq
            self.VQ = VQ(use_vq, n_content)
        self.SILC = SILC
        self.MRG = MRG
        self.text_d = text_d
        self.sos_id = sos_id
        self.n_content = n_content
        self.n_token = n_token
    
    def _infer(self, image_feature, caption_s, capseg, beam_size, mode, max_sen_num, decode_first, decode_second, optional_ret):
        """
        Infer the model (decoders) in an auto-regressive manner.
        mode: "test" for inference, "greedy" and "sample" for reinforcement learning
        """
        B = image_feature.shape[0]
        out = torch.full((B, 0), 0)
        if hasattr(self, 'decoder') and (mode=='test' or not self.MRG):
            out, probs = self.decoder(image_feature, caption=None, top_k=beam_size, mode='greedy' if mode=='test' else mode) #[B, L]
        if mode!='test' and not self.MRG:
            return out, probs, 0,0,0,0,0,0,0,0
        out_s = torch.full((B, 0, 0), 0.0)
        out_s2 = torch.full((B, 0, 0), 0.0)
        if self.MRG:
            eos_topic = torch.full((1, self.n_content), 1/self.n_content).to(image_feature.device) #topic for empty sentences padded after real sentences. Can be used as a End sign
            vq_topic, eos_idx, loss_vq, _ = self.VQ(eos_topic)

            if mode=='test' or decode_first:
                topic_token, topic_probs = self.topic_decoder(image_feature, None, eos_id = eos_idx[0], top_k=1, mode='greedy' if mode=='test' else mode)
                topic_token = topic_token[:,1:] #[B, l] ignore the first topic token (sos token)
                topic_feature = self.VQ.get_feature(topic_token) #[B, l, D]
                meaningful = torch.ne(topic_token, eos_idx[0]).long() #[B, l]
                if decode_second:
                    sen_features_pred = topic_feature*meaningful.unsqueeze(2) #为了返回sen_features看的时候能区分哪些是有效feature
                    out_s,_ = self.decode_second(image_feature, topic_feature, meaningful, beam_size=beam_size, mode='greedy'if mode=='sample' else mode)
                else:
                    out_s = None
            else:
                topic_token, topic_probs, out_s = None, None, None
            
            #We can test the word decoder using the GT sentence topics predicted by SILC, bypassing the topic decoder.
            if capseg is not None: 
                text_feature = self.sentence_encoder(caption_s, attn_group=capseg) #[B,L,768]
                
                if max_sen_num is None:
                    max_sen_num = torch.max(capseg).long()-100 #The max number of sentences of the reports in this batch
                #print('max_sen_num', max_sen_num)
                sen_feature, sen_content, sen_topic = self.get_sen_features(capseg, max_sen_num, text_feature) #[n_sen,B,D]
                topic_feature = sen_topic.transpose(0,1) #[B, n_sen, D]
                topic_feature, vq_idx, loss_vq, _ = self.VQ(topic_feature)
                meaningful = torch.ne(vq_idx, eos_idx[0]).long() #[B, n_sen]
                
                if decode_second:
                    out_s2, probs = self.decode_second(image_feature, topic_feature, meaningful, beam_size=beam_size, mode=mode)
                else:
                    out_s2, probs = torch.zeros((1,1,1)), None
            if mode!='test':
                return out_s2.reshape(-1, out_s2.shape[2]), probs, vq_idx, topic_token, topic_probs, out_s, eos_idx[0]
        optional = {}
        for x in optional_ret:
            optional[x] = eval(x)
        return out, out_s, out_s2, optional
    
    def forward(self, images, caption, caption_s = None, capseg = None, 
                beam_size=1, verbose = False, optional_ret = [], mode='test',
                max_sen_num=None, decode_first=False, decode_second=True):
        """
        caption: [B,L], used for baseline model and the conventional CLIP training. If None, the model will do inference rather than forward
        caption_s: [B,L], Similar to caption, despite having a <SOS> token before each sentence.
        capseg: [B,L] Corresponds to caption_s, indicating the segment of each sentence.
            For the i-th sentence (starting from 1), capseg will be i+100 at the beginning of the sentence, and i elsewhere. For example:
            caption_s[0]: [<SOS> Lung volume is normal . <SOS> There is no pleural effusion . <EOS> <PAD> <PAD> ...]
            capseg[0]:    [ 101   1     1    1    1    1  102  2     2  2     2       2     2  0     0     0    ...]
        """
        image_feature = self.image_encoder(images) #[B,n_content,512]
        
        image_feature = self.fuse(image_feature) #[B, n_content, 512]
        image_feature = image_feature[:,:self.n_content] #[B, n_content, 512]
        image_feature_fused = image_feature[:,0] #[B, 512] Only used for normal CLIP training.
        if caption is None:
            return self._infer(image_feature, caption_s, capseg, beam_size, mode, max_sen_num, decode_first, decode_second, optional_ret)
        
        if hasattr(self, 'decoder'): #Baseline model
            out, attn2 = self.decoder(image_feature, caption) #[B, L, n_token]
        else:
            out = torch.full((caption.shape[0], caption.shape[1], self.n_token), 0.0).to(caption.device)
        if hasattr(self, 'text_encoder'):
            text_feature = self.text_encoder(caption) #[B,L,768]或[B,L,512]，取决于是否使用bert
            text_feature_whole = text_feature[:,0] #[B,768]
        if hasattr(self, 'sentence_encoder'):
            text_feature_layers = self.sentence_encoder(caption_s, attn_group=capseg, verbose=verbose, layer_outputs=True) #[n_layer,B,L,768]
            text_feature = text_feature_layers[-1] #[B,L,768]
        if self.SILC:
            image_content = self.find_image_content(image_feature) #[B,n_content,512]

            max_sen_num = torch.max(capseg).long() #max number of sentences for all reports in the batch
            if hasattr(self, 'sentence_encoder'):
                max_sen_num = max_sen_num-100 
            sen_feature, sen_content, sen_topic = self.get_sen_features(capseg, max_sen_num, text_feature) #[n_sen,B,D]

            #[n_sen,1,B,n_content]×[512,n_content,B] = [n_sen,512,B,B], [i,:,j,k]为batch中第j个报告第i个句子的topic用到第k个CT得到的feature
            features = torch.matmul(sen_topic.unsqueeze(1), image_content.transpose(0,2)) #[n_sen,512,B,B]
            features = normalize(features, dim=1)
            #[n_sen,512,B,1] * [n_sen,512,B,B]=[n_sen,512,B,B]再在512维度sum得[n_sen,B,B]
            sim_s2i = torch.sum(sen_content.transpose(1,2).unsqueeze(3) * features, dim=1) #[n_sen,B,B] 
            sim_s2i = sim_s2i*torch.exp(self.t_s)
            topic_feature = sen_topic.transpose(0,1) #[B, n_sen, n_content]
            if self.MRG:
                vq_topic, vq_idx, loss_vq, used = self.VQ(topic_feature) #[B, n_sen, n_content]
                sen_feature_arranged = self.get_sen_feature_arranged(capseg, max_sen_num, vq_topic.transpose(0,1))
                #此时sen_feature_arranged D维度应该和image_feature一样
                out_s, _ = self.word_decoder(image_feature, caption_s, caption_f = sen_feature_arranged, attn_group=capseg, verbose=verbose)
                
                sos = torch.full((vq_idx.shape[0], 1), self.n_vectors).to(vq_idx.device) #Use n_vectors as the <SOS> token for topic decoder
                whole_caption = torch.cat([sos, vq_idx], dim=1) #[B, 1+n_sen]
                sen_token_pred, _  = self.topic_decoder(image_feature, whole_caption) #[B, 1+n_sen, nv+1]
        else:
            sim_s2i, topic_feature = 0, 0
        if not self.MRG:
            sen_token_pred, out_s = 0, 0
            vq_idx = torch.FloatTensor([0]).to(images.device)
            loss_vq = torch.FloatTensor([0]).to(images.device)
            used = torch.FloatTensor([0]).to(images.device)

        if hasattr(self, 'text_encoder'): #CLIP
            image_shared = self.image_projection(image_feature_fused) #[B, 512]
            text_shared = self.text_projection(text_feature_whole) #[B, 512]
            image_shared = normalize(image_shared, dim=1)
            text_shared = normalize(text_shared, dim=1)
            sim = torch.mm(image_shared, text_shared.transpose(0,1))*torch.exp(self.t) #[B,B]
        else:
            sim = torch.full((images.shape[0], images.shape[0]), 0.0).to('cuda:0')
        optional = {}
        for x in optional_ret:
            optional[x] = eval(x)
        ret = (out, sim, sim_s2i, out_s, topic_feature, 
               sen_token_pred, vq_idx, loss_vq, used, optional)
        return ret
    def decode_second(self, image_feature, sen_features, meaningful,
                      verbose = False, beam_size=1, mode='test'):
        """
        image_feature: [B,n_content,512]
        sen_features: [B,l,D], sentence topic features
        meaningful: [B,l]
        return: [B, l, l2]
        """
        l = sen_features.shape[1]
        sen_features = sen_features.reshape(sen_features.shape[0]*l, -1) #[B*l, D]
        meaningful = meaningful.reshape(-1)
        if hasattr(self, 'sen_d_trans'):
            sen_features = self.sen_d_trans(sen_features) #[B*l,512]

        temp_image_feature = image_feature.unsqueeze(0).repeat(l,1,1,1).transpose(0,1).reshape(-1, *image_feature.shape[1:]) #[B*l, n_content, 512] 
        max_l = 50 if mode=='test' else 25 #max length of each sentence
        out_s, probs = self.word_decoder(temp_image_feature, caption=None, caption_f=sen_features,
                                      meaningful=meaningful, max_l=max_l, verbose=verbose, top_k=beam_size, mode='greedy' if mode=='test' else mode) #[B*l, l2], [B*l, l2, n_token]
        out_s = out_s*meaningful.unsqueeze(1) #把没有意义的句子feature生成的token也全弄成零PAD
        out_s = out_s.view(image_feature.shape[0], -1, out_s.shape[1]) #[B, l, l2]
        return out_s, probs
    def get_sen_feature(self, capseg, max_sen_num, text_feature):
        sen_i = torch.arange(1, max_sen_num+1).unsqueeze(1).unsqueeze(2).to(capseg.device) #[n_sen,1,1]
        capseg = capseg.unsqueeze(0) #[1,B,L]
        sen_mask = (capseg==sen_i+100).unsqueeze(3) #[n_sen,B,L,1]
        sen_feature = torch.sum(text_feature.unsqueeze(0)*sen_mask, dim=2) #[n_sen,B,768]
        return sen_feature
    def get_sen_features(self, capseg, max_sen_num, text_feature):
        """
        text_feature: [B,L,768]
        capseg: [B,L]
        """
        sen_feature = self.get_sen_feature(capseg, max_sen_num, text_feature) #[n_sen,B,D]
        sen_topic = self.find_sen_topic(sen_feature) #[n_sen, B, n_content]
        sen_content = self.find_sen_content(sen_feature) #[n_sen, B, text_d]
        sen_content = normalize(sen_content, dim=-1)
        return sen_feature, sen_content, sen_topic

    def get_sen_feature_arranged(self, capseg, max_sen_num, sen_topic):
        """
        sen_topic: [n_sen, B, D]
        """
        d = sen_topic.shape[2]
        sen_feature_arranged = torch.full((capseg.shape[0], capseg.shape[1], d), 0.0).to(capseg.device) #[B,L,d]。注意初值要写成float 0.0后面才能赋值成功

        sen_i = torch.arange(1,max_sen_num+1).unsqueeze(1).unsqueeze(2).to(capseg.device) #[n_sen,1,1]
        sen_sos_index = (capseg==sen_i+100).nonzero(as_tuple=True) #([X],[X],[X])，分别是n_sen维度、B维度、L维度
        #sen_feature_arranged每个句子sos的位置设为句子feature
        sen_feature_arranged[sen_sos_index[1:3]] = sen_topic[sen_sos_index[0:2]] #[n_content]
        if hasattr(self, 'sen_d_trans'):
            sen_feature_arranged = self.sen_d_trans(sen_feature_arranged) #[B,L,512]
        return sen_feature_arranged #[B,L,d]
    
    def get_sim_s2i(self, images, caption, capseg):
        """
        caption is caption_s in forward
        """
        assert self.SILC
        image_feature = self.image_encoder(images) #[B,1,512] or [B,n_content,512]
        image_feature = self.fuse(image_feature) #[B, n_content, 512]
        text_feature_layers = self.sentence_encoder(caption, attn_group=capseg, verbose=False, layer_outputs=True) #[n_layer,B,L,768]
        text_feature = text_feature_layers[-1] #[B,L,768]
        
        image_content = self.find_image_content(image_feature) #[B,n_content,512]

        max_sen_num = torch.max(capseg).long()
        if hasattr(self, 'sentence_encoder'):
            max_sen_num = max_sen_num-100 
        sen_feature, sen_content, sen_topic = self.get_sen_features(capseg, max_sen_num, text_feature) #[n_sen,B,D]
        
        #以下：[n_sen,1,B,n_content]×[512,n_content,B] = [n_sen,512,B,B], [i,:,j,k]为batch中第j个报告第i个句子的topic用到第k个CT得到的feature
        features = torch.matmul(sen_topic.unsqueeze(1), image_content.transpose(0,2))  #[n_sen,512,B,B]
        features = normalize(features, dim=1)
        
        #以下[n_sen,512,B,1] * [n_sen,512,B,B]=[n_sen,512,B,B]再在512维度sum得[n_sen,B,B]
        sim_s2i = torch.sum(sen_content.transpose(1,2).unsqueeze(3) * features, dim=1) #[n_sen,B,B] 
        return sim_s2i, features, sen_content, sen_topic
    
    def encode_sentence(self, caption, capseg, ret_VQ=False):
        text_feature = self.sentence_encoder(caption, attn_group=capseg) #[B,L,768]
        max_sen_num = torch.max(capseg).long()-100 #batch中所有报告最大句子数
        #三个feature都是 [n_sen,B,D]
        sen_feature, sen_content, sen_topic = self.get_sen_features(capseg, max_sen_num, text_feature)
        if ret_VQ:
            sen_features = sen_topic
            sen_features = sen_features.transpose(0,1) #[B, n_sen, D]
            sen_features, vq_idx, loss_vq, _ = self.VQ(sen_features)
            return vq_idx, sen_topic.transpose(0,1), sen_features, sen_content.transpose(0,1)
        return sen_feature, sen_content, sen_topic

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, input_l = 32, n_token = None, d = 512, n_head = 8, pad_id=0):
        """
        n_layers不是图片的个数，而是transformer的层数
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_head)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers) 
        self.transformer_encoder = mytransformers.TransformerEncoder(encoder_layer, num_layers=n_layers) 
        self.posit_embedding = nn.Embedding(input_l, d)
        if n_token is not None:
            self.token_embedding = nn.Embedding(n_token, d)
        self.nhead = n_head
        self.pad_id = pad_id
        self.input_l = input_l
    def forward(self, inputs, input_f=None, attn_group = None, verbose = False, layer_outputs=False):
        """
        inputs: [B,S,512]或[B,S], S可以是n_content（输入为图像序列时）或文本长度（输入为文本序列时）
        input_f: [B,S,512]，需保证inputs本身是[B,S]
        attn_group: [B,S] capseg
        """
        posit_index = torch.arange(inputs.shape[1]).unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device) #(B,S)
        if attn_group is not None: #Make independent position embedding for each sentence:
            n_sen = torch.max(attn_group).long()-100
            for i in range(1, n_sen+1): 
                head_mask = (attn_group==i+100) 
                index = head_mask.nonzero(as_tuple=False) #[X,2], X<=B
                for x in index:
                    posit_index[x[0], x[1]:] = torch.arange(inputs.shape[1]-x[1]).to(inputs.device)
        source_posit_embed = self.posit_embedding(posit_index) # [B,S,512]
        if inputs.dim()==2:
            padding_mask = (inputs == self.pad_id) #[B,S]。bool类型mask，True地方会被忽略
        else:
            padding_mask = None
        if inputs.dim()==2:
            inputs = self.token_embedding(inputs) #[B,S,512]
        if input_f is not None:
            inputs = inputs + input_f #[B,S,512]
        
        source_embed = inputs + source_posit_embed
        source_embed = torch.transpose(source_embed, 0, 1)
        attn_mask = torch.full((inputs.shape[1], inputs.shape[1]),0.0).to(inputs.device)
        if attn_group is not None:
            attn_mask = generate_group_mask(attn_group, self.nhead).to(inputs.device)

        output = self.transformer_encoder(src=source_embed, mask=attn_mask, src_key_padding_mask=padding_mask,
                                          layer_outputs=layer_outputs) #[S, B, 512]或[n_layer, S, B, 512]
        output = torch.transpose(output, -2, -3) #[B, S, 512] 或 [n_layer, B, S, 512]
        return output
    
class ImageEncoder(nn.Module):
    def __init__(self, fuse = None, backbone = 'resnet101', output_d = 512, pretrain=True):
        """
        """
        print('pretrained backbone', pretrain)
        super().__init__()
        
        if 'resnet' in backbone:
            if backbone=='resnet101':
                resnet = models.resnet101(pretrained = pretrain)
                self.backboneD = 2048
            elif backbone=='resnet50':
                resnet = models.resnet50(pretrained = pretrain)
                self.backboneD = 2048
            elif backbone=='resnet18':
                resnet = models.resnet18(pretrained = pretrain)
                self.backboneD = 512
            #最后两层：avepool和linear
            layers = list(resnet.children())[:-2]
        elif backbone=='densenet121':
            densenet = torch.hub.load('pytorch/vision:v0.8.0', 'densenet121', pretrained=True)
            self.backboneD = 1024
            layers = list(densenet.children())[:-1]
        self.model = nn.Sequential(*layers) 
        self.output = nn.Linear(self.backboneD, output_d)
        self.backbone = backbone
        if fuse is None:
            fuse = 'mean'
        assert fuse in ['max', 'mean', 'cat', None], fuse
        self.fuse = fuse
        
    def forward(self, images):
        """
        images: [batch, n_view, 3, H, W]
        """
        batch = images.shape[0]
        n_view = images.shape[1]
        images = images.view(-1, *(images.shape[2:])) #[batch*n_view, 3, H, W]
        feature = self.model(images) #[batch*n_view, D, 7, 7], assuming H=224 and W=224
        
        feature = feature.view(batch, n_view, self.backboneD, -1) #[batch, n_view, D, 49]
        if self.fuse=='mean':
            feature = torch.mean(feature, dim=1) #[batch, D, 49]
        elif self.fuse=='max':
            feature = torch.max(feature, dim=1).values #[batch, D, 49]
        elif self.fuse=='cat':
            feature = torch.flatten(feature.transpose(1,2), start_dim=2) #[batch, D, n_view*49]
        feature = feature.transpose(1,2) #[batch, 49, D]
        feature = self.output(feature) #[batch, 49, output_d]
        #print(feature.shape)
        return feature

class ReportDecoder(nn.Module):
    def __init__(self, n_token, max_l, pad_id, sos_id, input_l = 32,
                 nhead = 8, n_layer = 6, d=512):
        """
        Transformer decoder
        """
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.nhead = nhead
        self.d = d
        self.n_token = n_token
        self.max_l = max_l
        self.token_embedding = nn.Embedding(n_token, d)
        self.posit_embedding = nn.Embedding(max_l, d)
        self.source_posit_embedding = nn.Embedding(input_l, d)
        
        #decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        #self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6) 
        decoder_layer = mytransformers.TransformerDecoderLayer(d_model=d, nhead=nhead)
        self.transformer_decoder = mytransformers.TransformerDecoder(decoder_layer, num_layers=n_layer) 
        
        self.output = nn.Linear(d, n_token) #output layer
        self.output.weight = self.token_embedding.weight
        
    def forward(self, source, caption, caption_f = None, attn_group = None, top_k = 1, verbose=False,
                meaningful=None, eos_id=2, layer_outputs=False, max_l=None, mode='greedy'):
        """
        source: [B,S,E], S=1 or n_content
        caption: [B,L], token index。如果做feature的生成，则caption值应为1或0，0表示pad
        caption_f: [B,L,E]，在caption得到token embedding后再加上caption_f。或者[B,E]，在infer时使用，每个句子最开始加上caption_f
        attn_group: [B,L]. if None, generate each row as a whole sequence. Otherwise, generate independent sentences
        """
        if caption is None:
            return self._infer(source=source, caption_f = caption_f, top_k=top_k, meaningful=meaningful, eos_id=eos_id, max_l=max_l, verbose=verbose,
                               mode=mode) # (B,l)
        posit_index = torch.arange(caption.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(caption.device) #(B,L)
        if attn_group is not None: #Make independent position embedding for each sentence:
            n_sen = torch.max(attn_group).long()-100
            for i in range(1, n_sen+1): 
                head_mask = (attn_group==i+100) 
                index = head_mask.nonzero(as_tuple=False) #[X,2], X<=B
                for x in index:
                    posit_index[x[0], x[1]:] = torch.arange(caption.shape[1]-x[1]).to(caption.device)
            if verbose:
                print(posit_index[0])
        target_embed = self.posit_embedding(posit_index) #[B,L,E]
        if hasattr(self, 'token_embedding'):
            target_embed += self.token_embedding(caption) #[B,L,E]
        if caption_f is not None:
            target_embed = target_embed + caption_f #[B,L,E]
        padding_mask = (caption == self.pad_id) #[B,L]
        
        attn_mask = self.generate_square_subsequent_mask(caption.shape[1]).to(caption.device) #[L,L]
        if attn_group is not None:
            temp = generate_group_mask(attn_group, self.nhead).to(caption.device) #Each sentence (segment) can only see elements in itself
            attn_mask = torch.minimum(attn_mask, temp) #[B*nhead, L, L]
        posit_index = torch.arange(source.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(source.device) #(B,S)
        
        source_posit_embed = self.source_posit_embedding(posit_index) # [B,S,E]
        source_embed = source + source_posit_embed
        
        target_embed = torch.transpose(target_embed, 0, 1)
        source_embed = torch.transpose(source_embed, 0, 1)
        
        out, attn, attn2 = self.transformer_decoder(tgt=target_embed, memory=source_embed, tgt_mask=attn_mask, tgt_key_padding_mask=padding_mask,
                                             need_weights=True, layer_outputs=layer_outputs)

        attn = torch.transpose(attn, 0, 1) #[batch, n_layer, L, S]. No use
        attn2 = torch.transpose(attn2, 0, 1) #No use
        out = torch.transpose(out, -2, -3) #[B, L, E] 或 [n_layer, B, L, E]
        if hasattr(self, 'output'):
            out = self.output(out) #[B, L, n_token]
        return out, attn2
    
    def _infer(self, source, caption_f = None, top_k=1, verbose = False, 
               meaningful = None, eos_id = 2, max_l = None, mode='greedy', 
               ):
        """
        source: [B,S,E], S=1 or n_content
        caption_f: [B,E]
        meaningful: [B]
        mode: when 'greedy', use beam search or greedy decoding to generate the best result;
              when 'sample', sample texts based on predicted probabilities, which are used for reinforcement learning
        """
        if mode=='sample':
            assert top_k==1
        if max_l is None:
            max_l = self.max_l
        outputs = torch.ones((top_k, source.shape[0], 1), dtype=torch.long).to(source.device) * self.sos_id # (K,B,1) SOS
        probs = torch.zeros((source.shape[0], 1, self.n_token), dtype=torch.float32).to(source.device) #[B,1,n_token]
        scores = torch.zeros((top_k, source.shape[0]), dtype=torch.float32).to(source.device) # (K,B)
        not_over = torch.full((top_k, source.shape[0]), 1).to(source.device) #[K,B]
        if meaningful is not None:
            not_over = torch.minimum(not_over, meaningful.unsqueeze(0))
        now_f = None
        if caption_f is not None:
            assert len(caption_f.shape)==2
            now_f = caption_f.unsqueeze(1) #[B,1,E]
            append_f = torch.full(now_f.shape, 0.0).to(now_f.device) #[B,1,E]，全零
        attn_group = None
        #if verbose:
        #   tokenizer = Tokenizer('./preprocess/mimic', 100, file_prefix='R2Gen_', english=True)

        for token_i in range(1, max_l):
            possible_outputs = []
            possible_scores = []
            possible_not_over = []
            
            for k in range(top_k if token_i>1 else 1):
                output = outputs[k] # (B,L)
                score = scores[k] # (B)
                
                out, attn2 = self.forward(source, output, caption_f=now_f, attn_group=attn_group) #[B, L, n_token], [B, L, L]
                prob = nn.functional.softmax(out, dim=2)[:,-1] #[B, n_token]
                if mode=='sample':
                    idx = torch.distributions.Categorical(probs=prob.detach()).sample().unsqueeze(1) #[B, 1]
                    val = prob.gather(1, idx) #[B,1]
                    probs = torch.cat([probs, prob.unsqueeze(1)], 1) #[B, token_i+1, n_token]
                else:
                    val, idx = torch.topk(prob, top_k) # (B,K) 
                log_val = torch.log(val+1e-8) # (B,K) Use logsum to perform multiply
                
                for i in range(top_k):
                    new_output = torch.cat([output, idx[:,i].view(-1,1)], dim=-1) # (B,L+1)
                    new_score = score + log_val[:,i].view(-1) # (B)
                    #2023.11.5 以下，本来应该直接用eos判断over即可，但是nes情况下caption分句出了点问题，最后一个句子后面没有eos直接pad，所以模型可能不会预测eos
                    new_not_over = torch.minimum(not_over[k], torch.ne(new_output[:,-1], eos_id).long()*torch.ne(new_output[:,-1], self.pad_id).long()) #[B]
                    
                    possible_outputs.append(new_output.unsqueeze(0)) # (1,B,L+1)
                    possible_scores.append(new_score.unsqueeze(0)) # (1,B)
                    possible_not_over.append(new_not_over.unsqueeze(0)) # (1,B)
            
            possible_outputs = torch.cat(possible_outputs, dim=0) # (K^2,B,L+1)
            possible_scores = torch.cat(possible_scores, dim=0) # (K^2,B)
            possible_not_over = torch.cat(possible_not_over, dim=0) # (K^2,B)

            # Pruning the solutions
            val, idx = torch.topk(possible_scores, top_k, dim=0) # (K,B)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0],1) # (K,B)
            outputs = possible_outputs[idx,col_idx] # (K,B,L+1)
            scores = possible_scores[idx,col_idx] # (K,B)

            not_over = possible_not_over[idx,col_idx] # (K,B)
            #print(token_i, torch.sum(not_over))
            if caption_f is not None:
                now_f = torch.cat([now_f, append_f], dim=1) #[B,L+1,E]
            if torch.sum(not_over)==0: #Stop generation when all sentences end
                break
        val, idx = torch.topk(scores, 1, dim=0) # (1,B)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0],1) # (K,B)
        output = outputs[idx,col_idx].squeeze(0) # (B,L)
        score = scores[idx,col_idx] # (1,B)
        return output, probs # (B,L), [B,L,n_token]
    
    def generate_square_subsequent_mask(self, sz):
        #float mask, -inf无法关注，0可以关注
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
