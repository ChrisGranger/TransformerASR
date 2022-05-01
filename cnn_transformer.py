from turtle import forward
from torch import *
from torch import nn

class TransformerASR(nn.Module):

    def __init__(self, num_feats, output_len, cnn_layers=2, encoder_layers=2,decoder_layers=1):
        self.features = num_feats

        self.cnn_layers = nn.ModuleList(
            [nn.Conv2d(1,256,3,stride=2,padding=1)] + [nn.Conv2d(256,256,3,stride=2,padding=1) for _ in range(cnn_layers - 1)]
        )

        self.cnn_features = int((self.features - 1)/2 + 1)

        if self.cnn_features % 4 == 0:
            self.pool_pad = 0
        else:
            self.pool_pad = 4 - (self.cnn_features % 4)

        self.pooling_layer = nn.MaxPool2d([1,4],padding=self.pool_pad)
        self.pooling_features = int((self.cnn_features + 2*self.pool_pad - 4)/4 + 1)
        self.d = self.pooling_features * 256
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(self.d, 8) for _ in range(encoder_layers)]
        )

        self.time_reduction = nn.Linear(self.d * 2, self.d)

        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(self.d,6) for _ in range(decoder_layers)]
        )

        self.output = nn.Linear(self.d,output_len)
        self.ctc_output = nn.linear(self.d,output_len + 1)

    def _flatten(self,data):
        #[batch_size,num_filters,seq,num_features]
        s = data.size()
        data = data.view(s[2],s[0],s[1],s[3])
        #[seq,batch_size,num_filters * num_features]
        return data.flatten(-2)

    def forward(self,src_data,tgt_data,use_stepwise=False):
        data = src_data
        for cnn in self.cnn_layers:
            data = cnn(data)
            data = relu(data)
        
        data = self.pooling_layer(data)

        data = self._flatten(data)

        data = self._pos_embed(data)

        data = self.encoder_layers[0](data)

        d_size = data.size()
        data = data.transpose(0,1).reshape(d_size[1],d_size[0]/2,d_size[2]*2)

        data = self.time_reduction(data)

        data = tanh(data)

        for x in range(1,len(self.encoder_layers)):
            data = self.encoder_layers[x](data)

        ctc_out = self.ctc_output(data)
        ctc_out = softmax(ctc_out)

        padding_mask = triu(ones(tgt_data.size(0),tgt_data.size(0)),diagonal=1)

        for dec in decoder_layers:
            data = dec(data, tgt_mask=padding_mask)

        data = self.output(data)

        data = softmax(data)

        return data,ctc_out

    def _pos_embed(data):
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
        pos_encodings = FloatTensor(pos_encodings)
        with torch.no_grad():
            data = data + pos_encodings
        return data

        


