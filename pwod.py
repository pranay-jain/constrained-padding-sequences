from collections import defaultdict
import math
import argparse


from load_dataset import load_dataset, load_sequence_counts
from utils import pwod, main_tgt_length

cap_sequences = False
cap_length = 4 # if cap_sequences is enabled, then this will be the truncated length


def run_pwod(dataset, c):
    vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges, Q = load_dataset(dataset, cap_sequences, cap_length)
    s_seq_counts = load_sequence_counts(dataset)

    uniq_sizes = sorted(set(vertices.values()))

    size_map, block_tracker = pwod(uniq_sizes, c)
    pwod_sol = len(set(size_map.values()))

    pad_scheme = {}

    for v, s in vertices.items():        
        pad_scheme[v] = [(size_map[s],1.0)]
    
    i_inf_res = []    
    for tgt_length in range(1, max_length+1):
        max_probs = defaultdict(float)
    
        for seq in sequences:
            main_tgt_length(seq, pad_scheme, max_probs, tgt_length)
    
        i_inf = math.log2(sum(max_probs.values()))    
        i_inf_res.append(i_inf)
    
    return pad_scheme, i_inf_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for PWOD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset",
        choices=['wikipedia', 'linode_from_index', 'autocomplete', 'synthetic'])
    parser.add_argument("-c", "--pad_factor", help="Maximum allowed padding factor for PFS or PWOD pad scheme.", default=1.25)
    
    args = parser.parse_args()
    config = vars(args)

    run_pwod(
        config['dataset'],
        float(config['pad_factor']))
