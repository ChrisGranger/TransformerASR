from cnn_transformer import TransformerASR
from torch import *

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