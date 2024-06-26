import numpy as np
import pandas as pd
import random
import statistics
import math
from collections import defaultdict

from load_dataset import load_dataset, load_sequence_counts
from utils import main_l_div, main_tgt_length_mvmd


# Assumes all sequences are equal length
def construct_VA_seq_linode(vertices, sequences):
    max_seq_len = max([len(s) for s in sequences])
    va_seq = [dict() for i in range(max_seq_len)]
    
    for seq in sequences:
        prefix = ''
        for i, v in enumerate(seq):
            if i == 0:
                prefix += v
            else:
                prefix += f'~{v}'
            va_seq[i][prefix] = vertices[v]

    return va_seq


def construct_VA_seq(vertices, words, max_words_len = None):
    if max_words_len is None:
        max_words_len = max([len(w) for w in words])
    va_seq = [dict() for i in range(max_words_len)]

    for word in words:
        prefix = ''
        for i, c in enumerate(word):
            prefix += c
            va_seq[i][prefix] = vertices[prefix]

        avg = int(sum([val for val in va_seq[i].values()]) / len(va_seq[i].values()))
        while i+1 < max_words_len:
            i+=1
            prefix += '_'
            va_seq[i][prefix] = avg

    return va_seq


def pad_scheme_from_partitions(partitions):
    pad_scheme = []
    for partition in partitions:
        pad = []
        for group in partition:
            if len(group) == 0: continue

            pad_size = max(group.values())
            pad.append({key: pad_size for key in group.keys()})
        pad_scheme.append(pad)

    return pad_scheme


def get_flat_pad_scheme(pad_scheme):
    pad_scheme_flat = [item for sublist in pad_scheme for item in sublist]
    # partition_flat = [item for sublist in partitions for item in sublist]

    return pad_scheme_flat


def pr(s, weights):
    return weights[s] / sum([w for w in weights.values()])


def weight_set_from_list(lst, all_weights):
    return { k: all_weights[k] for k in lst }

def generate_group_mapping(vertices, partition):
    groups_dict = {}
    for i, p in enumerate(partition):
        for v in p:
            groups_dict[v] = i
    return groups_dict


def flatten_va_seq(va_seq):
    va_seq_flat = {}
    for d in va_seq:
        va_seq_flat.update(d)
    return va_seq_flat

def sequence_end_vertex(seq):
    return seq.split('~')[-1]


def compute_action_suffix_wts(va_seq, weights):
    va_seq_flat = flatten_va_seq(va_seq)
    max_suffix_weights, sum_suffix_weights = {}, {}

    for v_action in va_seq_flat.keys():
        # for j in range(1, len(v_action)):
        prefix = v_action[:-1]

        # For nodes that never appear as a prefix in a seq
        if v_action not in max_suffix_weights:
            max_suffix_weights[v_action] = 0
            sum_suffix_weights[v_action] = 1e-4

        if len(prefix) > 0:
            if prefix not in max_suffix_weights:
                max_suffix_weights[prefix] = weights[v_action]
                sum_suffix_weights[prefix] = weights[v_action]
            else:
                max_suffix_weights[prefix] = max(max_suffix_weights[prefix], weights[v_action])
                sum_suffix_weights[prefix] = sum_suffix_weights[prefix] + weights[v_action]
    
    return max_suffix_weights, sum_suffix_weights


def compute_action_suffix_wts_wiki(va_seq, weights):
    va_seq_flat = flatten_va_seq(va_seq)
    max_suffix_weights, sum_suffix_weights = {}, {}

    for v_action in va_seq_flat.keys():
        v_action_seq = v_action.split('~')

        # For nodes that never appear as a prefix in a seq
        if v_action not in max_suffix_weights:
            max_suffix_weights[v_action] = 0
            sum_suffix_weights[v_action] = 1e-4

        if len(v_action_seq) > 1:
            prefix, cur_v = '~'.join(v_action_seq[:-1]), v_action_seq[-1]
            if prefix not in max_suffix_weights:
                max_suffix_weights[prefix] = weights[cur_v]
                sum_suffix_weights[prefix] = weights[cur_v]
            else:
                max_suffix_weights[prefix] = max(max_suffix_weights[prefix], weights[cur_v])
                sum_suffix_weights[prefix] = sum_suffix_weights[prefix] + weights[cur_v]

    return max_suffix_weights, sum_suffix_weights


def suffixLDiversity(subset, sum_suffix_weights, max_suffix_weights, l):
    if len(subset) == 0:
        return True

    total_wt_suffix_set = sum([sum_suffix_weights[v] for v in subset if v in sum_suffix_weights])
    max_wt_suffix_set = max([max_suffix_weights[v] for v in subset if v in max_suffix_weights ])
    # print(max_wt_suffix_set, total_wt_suffix_set)
    return max_wt_suffix_set / total_wt_suffix_set <= 1/l


def svsdDiversityMvmdModified(va, weights, l, max_suffix_weights, sum_suffix_weights):
    s_va_keys = sorted(va.keys(), key=lambda x: (weights[x], va[x]), reverse=True)
    S = list(s_va_keys.copy())

    if len(S) == 0:
        return [va]
    
    if len(va.keys()) < 2*l and pr(s_va_keys[0], weights) <= 1/l:
        return [va]
    
    if pr(s_va_keys[0], weights) > 1/l:
        print('cant satisfy l-diversity!', sequence_end_vertex(s_va_keys[0]), weights[s_va_keys[0]], str(sum([w for w in weights.values()])) )
        # print(weights)
        # return [S]
        return [va]

    partitions = []
    
    while len(S) > 0:
        for cur_alpha in range(l-1, len(S)):
            if pr(S[0], weight_set_from_list(S[:cur_alpha+1], weights)) <= 1/l and \
                (cur_alpha >= len(S)-1 or pr(S[cur_alpha+1], weight_set_from_list(S[cur_alpha+1:], weights)) <= 1/l):
                
                P_alpha_minus, P_alpha_plus = S[:cur_alpha+1], S[cur_alpha+1:]

                if not suffixLDiversity(P_alpha_minus, sum_suffix_weights, max_suffix_weights, l) or \
                      not suffixLDiversity(P_alpha_plus, sum_suffix_weights, max_suffix_weights, l):
                    # print(max_wt_suffix_set, total_wt_suffix_set)
                    if cur_alpha == len(S) - 1:
                        print("Can't achieve l-diversity. Returning full set.")
                        return [va]
                        # return [S]
                    continue

                partitions.append({ k: va[k] for k in P_alpha_minus })
                S = P_alpha_plus
                break
    
    return partitions


def mvmdDiversity(D, weights, l, dataset='autocomplete'):
    if dataset == 'autocomplete':
        max_suffix_weights, sum_suffix_weights = compute_action_suffix_wts(D, weights)
    else:
        max_suffix_weights, sum_suffix_weights = compute_action_suffix_wts_wiki(D, weights)

    P = [svsdDiversityMvmdModified(D[0], weights, l, max_suffix_weights, sum_suffix_weights)]
    G = generate_group_mapping(D[0].keys(), P[0])

    for i in range(1, len(D)):
        print(i)
        G_new = {}
        va = D[i]
        for action in va.keys():
            if dataset == "autocomplete":
                group_idx = G[action[:-1]] # --> SWITCH: for autcomplete dataset
            
            else:   # SWITCH: Next two lines only for linode dataset
                prev_seq = '~'.join(action.split('~')[:-1])
                group_idx = G[prev_seq]

            G_new[action] = group_idx
        
        cur_partition = []
        for group_idx in range(len(P[i-1])):
            G_w = [action for action in va if G_new[action] == group_idx]
            # G_w_pages = [sequence_end_vertex(action) for action in G_w]

            cur_partition += svsdDiversityMvmdModified(
                { key: va[key] for key in G_w },
                { key: weights[sequence_end_vertex(key)] for key in G_w },
                l, 
                max_suffix_weights, 
                sum_suffix_weights)

        P.append(cur_partition)
        G = generate_group_mapping(D[i].keys(), P[i])
    return P


def run_mvmd_autocomplete(dataset, l):
    df_nodes = pd.read_csv("data/autocomplete_dataset/search_results.csv")
    df_q = pd.read_csv("data/autocomplete_dataset/transition_matrix.csv", index_col=0)
    df_weights = pd.read_csv("data/autocomplete_dataset/suffix_search_results.csv", index_col=None, header=None)

    edges = list(zip(df_q['from'], df_q['to']))

    words = list(df_nodes[df_nodes['NumSearchResults'] > 0]['Query'])
    words_len = 7

    eligible_words = [word for word in words if len(word) == words_len]
    print(len(eligible_words))

    vertices = dict(zip(df_nodes['Query'], df_nodes['AutocompletePacketSize']))
    weights = dict(zip(df_weights[0], df_weights[1]))

    va_seq = construct_VA_seq(vertices, words)
    va_seq_flat = { k: v for obj in va_seq for k, v in obj.items() }

    for v in va_seq_flat:
        if v not in weights:
            word_end_indx = v.index('_')
            weights[v] =  weights[v[:word_end_indx]]

    weights.pop('root')

    assert len(va_seq_flat) == len(weights)
    partitions = mvmdDiversity(va_seq, weights, l, dataset=dataset)
    return va_seq, weights, partitions


def run_mvmd_wiki(dataset, vertices, sequences, l, seed=1):
    random.seed(seed)
    df_wiki_weights = pd.read_csv("data/wikipedia_dataset/weights.csv")
    df_wiki_weights.columns = ['v', 'w']

    wiki_weights = dict(zip(df_wiki_weights['v'], df_wiki_weights['w']))

    # fix weights for pages that couldn't be found
    min_wt, max_wt = min([w for w in wiki_weights.values()]), max([w for w in wiki_weights.values()])
    ct = 0
    for p, w in wiki_weights.items():
        if w == -1:
            wiki_weights[p] = random.randint(min_wt, max_wt)
            ct += 1
    
    va_seq = construct_VA_seq_linode(vertices, sequences)
    weights = wiki_weights
    partitions = mvmdDiversity(va_seq, weights, l, dataset=dataset)
    return va_seq, weights, partitions


def run_mvmd_linode_from_index(dataset, vertices, sequences, l, cap_length):
    test_sequences = []
    for seq in sequences:
        if len(seq) >= cap_length:
            test_sequences.append(seq)
    sequences = test_sequences

    va_seq = construct_VA_seq_linode(vertices, sequences)
    weights = { v: 1 for v in vertices }
    partitions = mvmdDiversity(va_seq, weights, l, dataset=dataset)
    return va_seq, weights, partitions


def run_mvmd(dataset, l=3, cap_sequences = False, cap_length = 3):
    vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges, Q = load_dataset(dataset, cap_sequences, cap_length)
    s_seq_counts = load_sequence_counts(dataset)

    if dataset == "autocomplete":
        va_seq, weights, partitions = run_mvmd_autocomplete(dataset, l)
    elif dataset == "wikipedia":
        va_seq, weights, partitions = run_mvmd_wiki(dataset, vertices, sequences, l)
    elif dataset == "linode_from_index":
        va_seq, weights, partitions = run_mvmd_linode_from_index(dataset, vertices, sequences, l, cap_length)
    else:
        print("Error! Dataset not valid for this algorithm.")
    
    max_c = 0
    pad_factors = {}
    
    va_seq_flat = { k: v for obj in va_seq for k, v in obj.items() }
    
    pad_scheme = {}
    sum_pad_fac, len_pad_fac = 0, 0
    for partition in partitions:
        for group in partition:
            if len(group) == 0:
                continue

            pad_size = max(group.values())
            
            for v in group.keys():
                pad_scheme[v] = [(pad_size,1.0)]
                if v != 'root':
                    pad_factor = pad_size / va_seq_flat[v]
                    max_c = max(max_c, pad_factor)
                    sum_pad_fac += pad_factor
                    len_pad_fac += 1
                    # max_c = max(max_c, pad_size/vertices[v])

    print(f'max pad factor: {max_c}')
    print(f"mean pad factor: {sum_pad_fac / len_pad_fac}")
    pad_factors[l] = (max_c, sum_pad_fac / len_pad_fac)
    # print(pad_scheme)
    
    min_l_div = []
    avg_l_div = []
    max_l_div = []
    
    visited = {}
    for tgt_length in range(1, max_length+1):
        y_seq_counts = defaultdict(list)
    
        for seq in sequences:
            idx = '~'.join(seq[:tgt_length])
            if idx in visited:
                continue
            main_l_div(dataset, seq, pad_scheme, s_seq_counts, y_seq_counts, tgt_length, weights)
            visited[idx] = 1
    
        all_l_div = []
        
        for y_seq, y_seq_list_of_counts in y_seq_counts.items():
            denom = sum(y_seq_list_of_counts)
            
            local_l_div = math.inf
            for count in y_seq_list_of_counts:
                inverse_prob = denom / count
                local_l_div = min(local_l_div, inverse_prob)
    
            all_l_div.append(local_l_div)

        min_l_div.append(min(all_l_div))
        avg_l_div.append(statistics.mean(all_l_div))
        max_l_div.append(max(all_l_div))
    
    i_inf_res = []
    for tgt_length in range(1, max_length + 1):
        max_probs = defaultdict(float)

        for seq in sequences:
            main_tgt_length_mvmd(seq, pad_scheme, max_probs, tgt_length, dataset)

        i_inf = math.log2(sum(max_probs.values()))

        i_inf_res.append(i_inf)
    
    return {
        'l_div': (min_l_div, max_l_div, avg_l_div), 
        'partitions': partitions, 
        'pad_scheme': pad_scheme,
        'i_inf': i_inf_res,
        'pad_factors': pad_factors[l]
    }


if __name__ == "__main__":
    # replace the dataset below if running as a script
    #   otherwise, you may import and call the run_mvmd function in another script/notebook 
    run_mvmd("linode_from_index")