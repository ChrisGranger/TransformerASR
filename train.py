from cnn_transformer import TransformerASR
from torch import *
from torch import nn
import numpy as np
import xml.etree.ElementTree as ET
import re
import string
from torchaudio.transforms import MelSpectrogram, TimeStretch, TimeMasking, FrequencyMasking
from torchaudio import load
import torch.nn.functional as F


win_size = 4000 # in frames
hop_size = 160 # in frames
n_mels = 80

class BatchProcessor():

    def __init__(self,data,texts,indexes,batch_size=10):
        self.batch_size = batch_size
        self.indexes = indexes
        self.feature_len = data[0].size(-1)
        self.batches, self.texts_enc,self.texts = self._get_sorted_batches(data,texts)
        self._curr_batch_index = 0

    def get_next(self):
        ret_batch,ret_text_enc,ret_text = self.batches[self._curr_batch_index], self.texts_enc[self._curr_batch_index], self.texts[self._curr_batch_index]
        self._curr_batch_index += 1
        return ret_batch, ret_text_enc, ret_text

    def _get_sorted_batches(self,data,texts):
        sort_data, sort_texts = self._argsort_with_key(data,texts,lambda x : data[x].size(0))

        batches = []
        encoded_text_batches = []
        text_batches = []
        curr_ind = 0
        while True:
            if curr_ind < len(sort_data) - self.batch_size:
                curr = sort_data[curr_ind:(curr_ind + self.batch_size)] 
                batches.append(self._pad(curr))
                encoded_text_batches.append(self._batch_to_one_hot(sort_texts[curr_ind:(curr_ind + self.batch_size)]))
                text_batches.append(sort_texts[curr_ind:(curr_ind+self.batch_size)])
                curr_ind += self.batch_size
            else:
                curr = sort_data[curr_ind:]
                batches.append(self._pad(curr))
                encoded_text_batches.append(self._batch_to_one_hot(sort_texts[curr_ind:]))
                text_batches.append(sort_texts[curr_ind:])
                break
        return batches, encoded_text_batches, text_batches
            
    def _get_max_len(self,data):
        max_len = 0
        for x in data:
            if x.size(0) > max_len:
                max_len = x.size(0)

        return max_len

    def _pad(self,data):
        max_l = self._get_max_len(data)
        padded = []
        for x in data:
            padded.append(F.pad(x,(0,0,0,max_l - x.size(0))))
        return stack(padded)

    def _batch_to_one_hot(self,text_batch):
        curr_batch = []
        for text in text_batch:
            curr = []
            curr.append(zeros(len(self.indexes) + 1))
            for ch in text:
                curr.append(self._one_hot_encode(ch))
            curr_batch.append(stack(curr))
        return self._pad(curr_batch)

    def _one_hot_encode(self,byte):
        z = zeros(len(self.indexes) + 1)
        z[self.indexes[byte]] = 1
        return z

    def _argsort_with_key(self,arr,texts,key_func):
        idxs = sorted(arange(0,len(arr)),key=key_func)
        ret_arr = []
        ret_texts = []
        for x in idxs:
            ret_arr.append(arr[x])
            ret_texts.append(texts[x])

        return ret_arr, ret_texts


def load_data(xml_trans_file,base_dir):
    tree = ET.parse(xml_trans_file)
    root = tree.getroot()
    count = 0
    text = []
    audio_files = []
    ids = {}
    curr_id = 0
    for x in root:
        for y in x:
            text.append(clean_text(y[0].text))
            for ch in text[-1]:
                if ch not in ids:
                    ids[ch] = curr_id
                    curr_id += 1
            audio_files.append(y.attrib["audio"])
    specs = []
    to_spec = MelSpectrogram(n_fft=win_size,win_length=win_size,hop_length=hop_size,n_mels=n_mels)
    count = 0
    for d in audio_files:
        waveform, _ = load(base_dir + "/"  + d, normalize=True)
        m_spec = to_spec(waveform).squeeze(0)
        specs.append(m_spec.transpose(-1,-2))
        count += 1
        if count >= 1000:
            break

    b = BatchProcessor(specs,text,ids)
    return b

def clean_text(text):
    text,_ = re.subn("[s]", "",text)
    text = text.strip().lower()
    text = text.translate(text.maketrans("","",string.punctuation))
    return text

def ctc_score(seq,output,index_map):
    assert output.size(-1) == len(index_map) + 1, "Output size is incorrect!"
    out_len = output.size(0)
    eps_seq = '@' + '@'.join(ch for ch in seq) + "@"
    lattice = zeros(out_len,len(eps_seq))
    
    lattice[0][0] = -log(output[0][-1])

    first_idx = index_map[eps_seq[1]]

    lattice[0][1] = -log(output[0][first_idx])

    for out in arange(0,out_len - 1):
        for ch in arange(0,len(eps_seq)):
            c = lattice[out][ch]
            if c == 0:
                continue
            elif eps_seq[ch] == "@":
                lattice[out + 1][ch] += -log(output[out + 1][-1]) + c
                if ch < len(eps_seq) - 1:
                    lattice[out + 1][ch + 1] += -log(output[out + 1][index_map[eps_seq[ch + 1]]]) + c
            else:
                lattice[out + 1][ch] += -log(output[out + 1][index_map[eps_seq[ch]]]) + c
                if ch < len(eps_seq) - 1:
                    lattice[out + 1][ch + 1] += -log(output[out + 1][-1]) + c
    print(lattice)
    return lattice[-1][-1] + lattice[-1][-2]

def test_ctc():
    seq = "abcb"
    out = rand(10,4)
    idx = {"a":0,"b":1,"c":2}
    print(ctc_score(seq,out,idx))

def beam_decode(model,data,indeces,beam_size=10):
    ids = { y:x for x,y in indeces.items() }

    beam = [0] * beam_size
    seqs = [""] * beam_size
    data_copy = data
    enc = model.encode(data)

    num_feat = enc.size(-1)

    start = zeros(len(indeces) + 1)
    end_idx = len(indeces)
    beam_tensors = [list(start) for _ in arange(1,num_feat)]
    num_complete = 0
    while True:
        for num,b in enumerate(beam_tensors):
            if seqs[num][-1] == "@":
                continue
            tens = FloatTensor(b).unsqueeze(0)
            curr = model.decode(enc,tens,use_stepwise=True)

            best = argmax(curr[0,-1])
            beam[num] += -log(curr[0,-1,best])
            if best == end_idx:
                seqs[num] += "@"
                num_complete += 1
            else:
                seqs[num] += ids[best]
            one_hot = [0]*(len(indeces) + 1)
            one_hot[best] = 1
            beam_tensors[num].append(one_hot)


        if num_complete == beam_size:
            break

    for num,b in enumerate(beam):
        score = ctc_score(seq[num][:-2],data_copy,indeces)
        beam[num] = b/len(seqs[num]) + score

    return seqs[np.argmax(beam)]

def L(tgt, seqs):
    ctc = nn.CTCLoss()

if __name__ == '__main__':
    b = load_data('/home/chris/Downloads/nchlt_xho/transcriptions/nchlt_xho.trn.xml',"/home/chris/Downloads")
    