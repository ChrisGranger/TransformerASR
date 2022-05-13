from cnn_transformer import TransformerASR
from torch import *
from torch import nn,optim
import numpy as np
import xml.etree.ElementTree as ET
import re
import string
from torchaudio.transforms import MelSpectrogram, TimeStretch, TimeMasking, FrequencyMasking
from torchaudio import load
import torch.nn.functional as F
import torch
import sys


win_size = 400 # in samples
hop_size = 160 # in samples
n_mels = 80

class BatchProcessor():

    def __init__(self,data,texts,indexes,batch_size=10):
        self.batch_size = batch_size
        self.indexes = indexes
        self.feature_len = data[0].size(-1)
        self.strt_idx = len(self.indexes)
        self.pad_idx = len(self.indexes) + 1
        self.batches, self.texts_enc,self.texts, self.texts_dec = self._get_sorted_batches(data,texts)
        self._curr_batch_index = 0
        self.order = np.arange(0,len(self.batches))
        np.random.shuffle(self.order)

    def get_next(self):
        if self._curr_batch_index >= len(self.batches):
            return None
        ret_batch,ret_text_enc,ret_text, ret_dec = self.batches[self.order[self._curr_batch_index]], self.texts_enc[self.order[self._curr_batch_index]],\
            self.texts[self.order[self._curr_batch_index]],self.texts_dec[self.order[self._curr_batch_index]]
        self._curr_batch_index += 1
        return ret_batch, ret_text_enc, ret_text, ret_dec

    def get_num_batches(self):
        return len(self.batches)

    def reset(self):
        self._curr_batch_index = 0
        self.order = np.arange(0,len(self.batches))
        np.random.shuffle(self.order)

    def _get_sorted_batches(self,data,texts):
        sort_data, sort_texts = self._argsort_with_key(data,texts,lambda x : data[x].size(0))

        batches = []
        encoded_text_batches = []
        text_batches = []
        dec_batches = []
        curr_ind = 0
        while True:
            if curr_ind < len(sort_data) - self.batch_size:
                curr = sort_data[curr_ind:(curr_ind + self.batch_size)] 
                batches.append(self._pad(curr))
                encoded_text_batches.append(self._batch_to_indexes(sort_texts[curr_ind:(curr_ind + self.batch_size)]))
                text_batches.append(sort_texts[curr_ind:(curr_ind+self.batch_size)])
                dec_batches.append(self._batch_to_indexes(sort_texts[curr_ind:(curr_ind + self.batch_size)],apply_end=True))
                curr_ind += self.batch_size
            else:
                curr = sort_data[curr_ind:]
                batches.append(self._pad(curr))
                encoded_text_batches.append(self._batch_to_indexes(sort_texts[curr_ind:]))
                text_batches.append(sort_texts[curr_ind:])
                dec_batches.append(self._batch_to_indexes(sort_texts[curr_ind:],apply_end=True))
                break
        return batches, encoded_text_batches, text_batches, dec_batches
            
    def _get_max_len(self,data):
        max_len = 0
        for x in data:
            if x.size(0) > max_len:
                max_len = x.size(0)

        return max_len

    def _pad(self,data):
        max_l = self._get_max_len(data)
        if max_l % 2 != 0:
            max_l += 1
        padded = []
        for x in data:
            padded.append(F.pad(x,(0,0,0,max_l - x.size(0))))
        return stack(padded)

    def _get_longest_string(self, data):
        max_len = 0
        for x in data:
            if len(x) > max_len:
                max_len = len(x)

        return max_len

    def _batch_to_indexes(self,text_batch, apply_end=False):
        longest = self._get_longest_string(text_batch) + 2
        curr_batch = []
        for text in text_batch:
            curr = []
            if not apply_end:
                curr.append(self.strt_idx)
            for ch in text:
                curr.append(self.indexes[ch])
            if apply_end:
                curr.append(self.strt_idx)
            while len(curr) < longest:
                curr.append(self.pad_idx)
            curr_batch.append(LongTensor(curr))
        return stack(curr_batch)

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
            audio_files.append(re.sub("nchltAux1/","",y.attrib["audio"]))
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
    text,_ = re.subn("\[s\]", "",text)
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
        score = ctc_score(seqs[num][:-1],data_copy,indeces)
        beam[num] = b/len(seqs[num]) + score

    return seqs[np.argmax(beam)]

def L(enc_data, dec_data, ctc_tgt_seqs, ctc_tgt_lengths, dec_seqs, blank_label, pad_idx):
    input_lengths = full(size=(enc_data.size(1),), fill_value=enc_data.size(0), dtype=torch.long)
    ctc = nn.CTCLoss(blank=blank_label, reduction='mean')
    l_ctc = ctc(enc_data,ctc_tgt_seqs,input_lengths,ctc_tgt_lengths)
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    l_ce = ce(dec_data.view(-1,dec_data.size(-1)),dec_seqs.view(-1)) / ctc_tgt_lengths.size(0)
    return 0.7 * l_ce + 0.3 * l_ctc

def get_seq_lengths(strings):
    ls = []

    for x in strings:
        ls.append(len(x))

    return LongTensor(ls)

if __name__ == '__main__':
    b = load_data("/home/chris/Downloads/xho-aux1/xho/info/nchltAux1_xho.xml","/home/chris/Downloads/xho-aux1")

    n_epochs = 10

    model = TransformerASR(80,len(b.indexes))
    optimizer = optim.Adam(model.parameters(), 0.0001)
    n_epochs = 10
    blank_label = len(b.indexes)
    model.train()
    for x in np.arange(n_epochs):
        batch = b.get_next()
        t_loss = 0
        print(f"Epoch: {x}")
        batches_so_far = 0

        while batch is not None:
            ctc_seq = batch[1]
            ctc_seq = ctc_seq[:,1:]
            dec_seq = batch[-1]
            out_data,ctc_data = model(batch[0],batch[1])
            lengths = get_seq_lengths(batch[2])

            loss = L(log(ctc_data),out_data,ctc_seq,lengths,dec_seq,blank_label,b.pad_idx)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            t_loss += loss.item()

            #print(f"\tbatch_loss: {loss.item()}")

            batch = b.get_next()
            batches_so_far += 1
            sys.stdout.write('\r\x1b[K' + f"{batches_so_far/b.get_num_batches()}")

        print(f"\nAVG Loss: {t_loss/b.get_num_batches()}")
        print()
        b.reset()

