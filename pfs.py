from collections import defaultdict
import math
import argparse


from load_dataset import load_dataset, load_sequence_counts
from utils import create_strides, main_tgt_length
from lp import linear_program

cap_sequences = False
cap_length = 4 # if cap_sequences is enabled, then this will be the truncated length


def run_pfs(dataset, c, k, prefix_closed):
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
    
    return pad_scheme


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for PFS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset",
        choices=['wikipedia', 'linode_from_index', 'autocomplete', 'synthetic'])
    parser.add_argument("-c", "--pad_factor", help="Maximum allowed padding factor for PFS or PWOD pad scheme.", default=1.25)
    parser.add_argument("-k", "--stride", help="", default=2)
    parser.add_argument("--prefix_closed", help="increase verbosity", default=True)
    
    args = parser.parse_args()
    config = vars(args)
    print(config['prefix_closed'])

    run_pfs(
        config['dataset'],
        float(config['pad_factor']), 
        int(config['stride']),
        config['prefix_closed'])
