import torch
from torch import *
from torch import nn
import numpy as np
import torch.nn.functional as F

class TransformerASR(nn.Module):

    def __init__(self, num_feats, output_len, d_attn=256, cnn_layers=2, encoder_layers=1,decoder_layers=1,use_downsample=False):
        super(TransformerASR,self).__init__()
        self.features = num_feats
        self.cnn_layers = nn.ModuleList(
            [nn.Conv2d(1,256,3,stride=2,padding=1)] + [nn.Conv2d(256,256,3,stride=2,padding=1) for _ in np.arange(cnn_layers - 1)]
        )
        self.downsample = use_downsample
        self.cnn_features = int(self.features/(2 * cnn_layers)) * 256
        self.d_attn = d_attn
        self.dim_reduce = nn.Linear(self.cnn_features, self.d_attn)
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(self.d_attn, 8) for _ in np.arange(encoder_layers)]
        )

        self.time_reduction = nn.Linear(self.d_attn * 2, self.d_attn)

        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(self.d_attn,8) for _ in np.arange(decoder_layers)]
        )
        self.decoder_embed = nn.Embedding(output_len + 2,self.d_attn,padding_idx=output_len + 1)
        self.output = nn.Linear(self.d_attn,output_len + 1)
        self.ctc_output = nn.Linear(self.d_attn,output_len + 1)

    def _flatten(self,data):
        #[batch_size,num_filters,seq,num_features]
        s = data.size()
        data = data.view(s[2],s[0],s[1],s[3])
        #[seq,batch_size,num_filters * num_features]
        return data.flatten(-2)

    def forward(self,src_data,tgt_data,use_stepwise=False):
        data = src_data

        data, ctc_out = self.encode(data)

        data = self.decode(data,tgt_data,use_stepwise)

        return data,ctc_out

    def encode(self,in_data):
        data = in_data
        data = data.unsqueeze(1)
        for cnn in self.cnn_layers:
            data = cnn(data)
            data = relu(data)

        data = self._flatten(data)
        data = self.dim_reduce(data)
        data = self._pos_embed(data)
        data = self.encoder_layers[0](data)
        if self.downsample:
            d_size = data.size()
            if d_size[0] % 2 != 0:
                data = F.pad(data,(0,0,0,0,0,1))
                d_size = data.size()
            data = data.transpose(0,1).reshape(int(d_size[0]/2),d_size[1],d_size[2]*2)
            data = self.time_reduction(data)

            data = tanh(data)

        for x in np.arange(1,len(self.encoder_layers)):
            data = self.encoder_layers[x](data)

        ctc_out = self.ctc_output(data)
        ctc_out = softmax(ctc_out,-1,torch.float64)
        return data, ctc_out

    def decode(self,encoder_data,tgt_data,use_stepwise=False):
        data = encoder_data
        padding_mask=None
        tgt_data = tgt_data.transpose(0,1)
        tgt_data = self.decoder_embed(tgt_data)
        if not use_stepwise:
            padding_mask = triu(ones(tgt_data.size(0),tgt_data.size(0)),diagonal=1)
        for dec in self.decoder_layers:
            data = dec(tgt_data, data, tgt_mask=padding_mask)

        data = self.output(data)

        data = F.softmax(data,-1,torch.float64)
        return data

    def _pos_embed(self,data):
        embed_dim = data.size(-1)
        seq_length = data.size(0)

        pos_encodings = []
        for x in arange(seq_length):
            temp = []
            for y in arange(embed_dim):
                if y % 2 == 0:
                    temp.append(sin(x/(10000**(y/embed_dim))))
                else:
                    temp.append(cos(x/(10000**((y-1)/embed_dim))))
            temp = stack(temp)
            pos_encodings.append(temp)
        pos_encodings = stack(pos_encodings)
        data = data.transpose(0,1)
        with no_grad():
            data = data + pos_encodings
        data = data.transpose(0,1)
        return data

        


