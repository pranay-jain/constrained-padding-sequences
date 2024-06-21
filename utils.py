import math
from collections import defaultdict

# ---- Exported Functions -----

def main_tgt_length(s, pad_s, max_probs, tgt_length):
    if len(s) >= tgt_length:
        rec_tgt_length(s, pad_s, [0]*tgt_length, 0, 1.0, tgt_length, max_probs)

def main_tgt_length_mvmd(s, pad_s, max_probs, tgt_length, dataset):
    if len(s) >= tgt_length:
        rec_tgt_length_mvmd(s, pad_s, [0]*tgt_length, 0, 1.0, tgt_length, max_probs, dataset)

def create_strides(vertices, c, k=2):
    uniq_sizes = sorted(set(vertices.values()))
    
    size_map, block_tracker = pwod(uniq_sizes, c)
    pwod_sol = len(set(size_map.values()))
    
    floor_set = get_floors(uniq_sizes, c)

    floor_list = sorted(list(floor_set))
    
    floor_maxes = []

    for i, s in enumerate(uniq_sizes):
        if s in floor_set:
            l = []
            
            s_alt = s
            while s_alt <= c*s:
                l.append(s_alt)
                s_alt += 1
        
            for chunk in chunks(l, k):
                if len(chunk) > 0:
                    floor_maxes.append(chunk[-1])
        
    stride_uniq_sizes = sorted(set(floor_maxes))
    
    return uniq_sizes, stride_uniq_sizes

def pwod(size_list, max_over):
    u_sizes = sorted(size_list, reverse=True)
    size_map = {}
    
    cur_ceil = u_sizes[0]
    cur_floor = cur_ceil / max_over
    
    num_blocks = 1
    block_tracker = {}
    
    for size in u_sizes:
        if size < cur_floor:
            cur_ceil = size
            cur_floor = cur_ceil / max_over
            num_blocks += 1
        size_map[size] = cur_ceil
        block_tracker[size] = num_blocks
                
    return size_map, block_tracker


# ---- Helper Functions -----

def rec_tgt_length(s, pad_s, cur_sizes, i, prev_prob, tgt_length, max_probs):
    # cur_v = '~'.join(s[:i+1])       # SWITCH - only for linode
    cur_v = s[i]                    # Autocomplete
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == tgt_length - 1:
            final_s = tuple(cur_sizes)
            if cur_prob > max_probs[final_s]:
                max_probs[final_s] = cur_prob
                
        else:
            rec_tgt_length(s, pad_s, cur_sizes, i+1, cur_prob, tgt_length, max_probs)

def rec_tgt_length_mvmd(s, pad_s, cur_sizes, i, prev_prob, tgt_length, max_probs, dataset):
    if dataset == "linode_from_index":
        cur_v = '~'.join(s[:i+1])       # SWITCH - only for linode
    else:
        cur_v = s[i]                    # Autocomplete
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == tgt_length - 1:
            final_s = tuple(cur_sizes)
            if cur_prob > max_probs[final_s]:
                max_probs[final_s] = cur_prob
                
        else:
            rec_tgt_length_mvmd(s, pad_s, cur_sizes, i+1, cur_prob, tgt_length, max_probs, dataset)

def get_floors(size_list, c):
    floor_list = []
    
    this_floor = size_list[0]
    
    for s in size_list:
        if c * this_floor < s:
            floor_list.append(this_floor)
            this_floor = s
            
    floor_list.append(this_floor)
    
    return set(floor_list)

def rec(s, pad_s, cur_sizes, i, prev_prob):
    cur_v = s[i]
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == len(s) - 1:
            final_s = tuple(cur_sizes)
            if cur_prob > max_probs[final_s]:
                max_probs[final_s] = cur_prob
                
        else:
            rec(s, pad_s, cur_sizes, i+1, cur_prob)
            
def main(s, pad_s, max_probs):
    rec(s, pad_s, [0]*len(s), 0, 1.0)
    
def main_2(seq, poss_sizes, Pyc, max_length):
    rec_2(seq, poss_sizes, Pyc, [0]*len(seq), 0, max_length)
    
def rec_2(seq, poss_sizes, Pyc, cur_sizes, i, max_length):
    cur_v = seq[i]
    for size in poss_sizes[cur_v]:
        cur_sizes[i] = size

        if i == len(seq) - 1:
            tuple_sizes = tuple(cur_sizes + (max_length - len(seq)) * [0])
            if tuple_sizes not in Pyc:
                Pyc[tuple_sizes] = []
                
            cndl_counts = defaultdict(int)

            for x, v in enumerate(seq):
                cndl_counts[(v, cur_sizes[x])] += 1
                
            Pyc[tuple_sizes].append(cndl_counts)
        
        else:
            rec_2(seq, poss_sizes, Pyc, cur_sizes, i+1, max_length)
            
def chunks(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]  

def rec_entropy(s, pad_s, cur_sizes, i, prev_prob, y_seq_counts, s_seq_counts, tgt_length):
    cur_v = s[i]
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == tgt_length - 1:
            final_s = tuple(cur_sizes)
            y_seq_counts[final_s] += cur_prob * s_seq_counts[tuple(s)]
        else:
            rec_entropy(s, pad_s, cur_sizes, i+1, cur_prob, y_seq_counts, s_seq_counts, tgt_length)
            
def main_entropy(s, pad_s, s_seq_counts, y_seq_counts, tgt_length):
    if len(s) >= tgt_length:
        rec_entropy(s, pad_s, [0]*tgt_length, 0, 1.0, y_seq_counts, s_seq_counts, tgt_length)
        
def rec_condl_entropy(s, pad_s, cur_sizes, i, prev_prob, trunc_sequence_probs):
    cur_v = s[i]
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == len(s) - 1:
            trunc_sequence_probs[tuple(s)].append(cur_prob)
                
        else:
            rec_condl_entropy(s, pad_s, cur_sizes, i+1, cur_prob, trunc_sequence_probs)
            
def main_condl_entropy(s, pad_s, trunc_sequence_probs):
    rec_condl_entropy(s, pad_s, [0]*len(s), 0, 1.0, trunc_sequence_probs)
    
#### H(Y|S) code 
def compute_h_y_s(sequences, s_seq_counts, tgt_length):
    trunc_s_seq_counts = defaultdict(float)
    trunc_sequences_set = set()
    
    for seq in sequences:
        if len(seq) >= tgt_length:
            trunc_s_seq_counts[tuple(seq[:tgt_length])] += s_seq_counts[tuple(seq)]
            trunc_sequences_set.add(tuple(seq[:tgt_length]))
            
    trunc_sequences = []
    for seq in trunc_sequences_set:
        trunc_sequences.append(list(seq))
        
    trunc_sequence_probs = defaultdict(list)
    
    for seq in trunc_sequences:
        main_condl_entropy(seq, pad_scheme, trunc_sequence_probs)
        
    condl_entropy_calc = 0.0
    
    for seq in trunc_sequences:
        prob_seq = trunc_s_seq_counts[tuple(seq)] / total_count
        
        interior_entropy_sum = 0.0
        
        for prob in trunc_sequence_probs[tuple(seq)]:
            if prob > 0:
                interior_entropy_sum += prob * math.log2(prob)
                
        condl_entropy_calc += prob_seq * interior_entropy_sum
    return condl_entropy_calc

def rec_l_div(dataset, s, pad_s, cur_sizes, i, prev_prob, y_seq_counts, tgt_length, weights):
    if dataset == "linode_from_index":
        cur_v = '~'.join(s[:i+1])       # SWITCH - only for linode
    else:
        cur_v = s[i]                    # Autocomplete, or wiki
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == tgt_length - 1:
            final_s = tuple(cur_sizes)
            #y_seq_counts[final_s].append(cur_prob * s_seq_counts[tuple(s)])
            y_seq_counts[final_s].append(cur_prob * weights[s[i]])
        else:
            rec_l_div(dataset, s, pad_s, cur_sizes, i+1, cur_prob, y_seq_counts, tgt_length, weights)
            
def main_l_div(dataset, s, pad_s, s_seq_counts, y_seq_counts, tgt_length, weights):
    if len(s) >= tgt_length:
        rec_l_div(dataset, s, pad_s, [0]*tgt_length, 0, 1.0, y_seq_counts, tgt_length, weights)


def rec_l_div_for_lp(s, pad_s, cur_sizes, i, prev_prob, tgt_length, y_seq_counts, s_seq_counts):
    cur_v = s[i]                    # Autocomplete
    
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size
        
        if i == tgt_length - 1:
            final_s = tuple(cur_sizes)
            y_seq_counts[final_s].append((cur_v,cur_prob * s_seq_counts[tuple(s)]))
            #y_seq_counts[final_s].append(cur_prob * weights[s[i]])
        else:
            rec_l_div_for_lp(s, pad_s, cur_sizes, i+1, cur_prob, tgt_length, y_seq_counts, s_seq_counts)


def main_l_div_for_lp(s, pad_s, s_seq_counts, y_seq_counts, tgt_length):
    if len(s) >= tgt_length:
        rec_l_div_for_lp(s, pad_s, [0]*tgt_length, 0, 1.0, tgt_length, y_seq_counts, s_seq_counts)


def elementWiseDiff(a, b):
    return [j - i if j - i > 1e-10 else 0 for i, j in zip(a, b)]