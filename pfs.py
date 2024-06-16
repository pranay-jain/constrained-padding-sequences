from collections import defaultdict
import math

from load_dataset import load_dataset, load_sequence_counts
from utils import create_strides, main_tgt_length
from lp import linear_program


k = 2
prefix_closed = True
cap_sequences = False
cap_length = 4 # if cap_sequences is enabled, then this will be the truncated length

# dataset = 'autocomplete'
# dataset = "linode_from_index"
dataset = 'wikipedia'
#dataset = 'synthetic'

pad_factors = {}
pad_factors["autocomplete"]      = [1.05, 1.25, 1.50, 2.00]
pad_factors["linode_from_index"] = [1.05, 1.25, 1.50, 2.00]
pad_factors["wikipedia"]         = [1.05, 1.25, 1.50, 2.00]
pad_factors["synthetic"]         = [2.0]

c = 1.05


if __name__ == "__main__":
    vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges, Q = load_dataset(dataset, cap_sequences, cap_length)
    s_seq_counts = load_sequence_counts(dataset)

    uniq_sizes, stride_uniq_sizes = create_strides(vertices, c, k)

    if prefix_closed:
        lp, valid_ranges = linear_program(c, vertices, prefix_closed_sequences, uniq_sizes, stride_uniq_sizes, max_length)
    else:
        lp, valid_ranges = linear_program(c, vertices, sequences, uniq_sizes, stride_uniq_sizes, max_length)
    
    pad_scheme = {}

    for v, valid_list in valid_ranges.items():
        i = 0
        pad_list = []
        for s in valid_list:
            var = f"pys[{v},{s}]"
            prob = lp.getVarByName(var).X
            if prob > 0:
                pad_list.append((s, prob))
        
        pad_scheme[v] = pad_list
    
    i_inf_res = []    
    for tgt_length in range(1, max_length + 1):
        max_probs = defaultdict(float)
    
        for seq in sequences:
            main_tgt_length(seq, pad_scheme, max_probs, tgt_length)
    
        i_inf = math.log2(sum(max_probs.values()))
        
        print(f"i_inf for target sequence length {tgt_length} = {i_inf}")
    
        i_inf_res.append(i_inf)
    