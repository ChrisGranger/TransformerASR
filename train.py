from cnn_transformer import TransformerASR
from torch import *
import numpy as np

def load_data():
    return


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

def beam_decode(model,data,beam_size=10,indeces):
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

if __name__ == '__main__':
    data = load_data()

    