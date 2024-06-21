import numpy as np
import math
import csv
from collections import defaultdict

from load_dataset import load_dataset, load_sequence_counts
from utils import main_entropy, main_tgt_length


# Construct the sources list used in algorithm
def constructSourcesDestinations(vertices, edges):
    sources, destinations = {}, {}
    
    for v in vertices.keys():
        sources[v] = []
        destinations[v] = []
    
    for edge in edges:
        src, dst = edge[0], edge[1]
        sources[dst].append(src)
        destinations[src].append(dst)
    
    return (sources, destinations)


def coarsestBisumulation(partition, vertices, edges, Q):
    idx = 0
    sets = {}
    B_tracker = {}

    sources, _ = constructSourcesDestinations(vertices, edges)

    for block in partition:
        sets[idx] = block

        for x_i in block:
            B_tracker[x_i] = idx

        idx += 1

    l = set(list(range(idx)))

    #iterCount = 0
    while len(l) > 0:
        s_id = l.pop()
        s = sets[s_id]
        l1, l2 = set(), set()
        sums = defaultdict(float)

        subsets = {}

        for x_j in s:
            for x_i in sources[x_j]:
                sums[x_i] += Q[(x_i,x_j)]
                l1.add(x_i)

        for x_i in l1:
            B_id = B_tracker[x_i]
            B = sets[B_id]

            B.remove(x_i)

            if B_id not in subsets:
                subsets[B_id] = {}

            if sums[x_i] not in subsets[B_id]:
                subsets[B_id][sums[x_i]] = set()

            subsets[B_id][sums[x_i]].add(x_i)

            l2.add(B_id)

        for B_id in l2:
            largest_size = len(sets[B_id])
            largest_block = B_id

            to_add = set()
            was_in_l = False

            if len(sets[B_id]) == 0:
                del sets[B_id]

                if B_id in l:
                    was_in_l = True
                    l.remove(B_id)

            else:
                to_add.add(B_id)


            for subset in subsets[B_id].values():
                sets[idx] = subset

                if len(subset) > largest_size:
                    largest_size = len(subset)
                    largest_block = idx

                for x_i in subset:
                    B_tracker[x_i] = idx

                to_add.add(idx)
                idx += 1

            #for ss in to_add:
                #print(sets[ss])

            if len(to_add) == 1:
                if was_in_l:
                    l = l.union(to_add)
                    #print("added a singleton")
            else:
                to_add.remove(largest_block)
                #print("removed largest")
                l = l.union(to_add)
        return (sets, B_tracker)
    
def bdk_step1(vertices, edges, Q):
    nodes = np.array(list(vertices.keys()))
    initial_partition = np.random.randint(0, 2, size=nodes.shape[0])

    initial_partition = [set(nodes[initial_partition == 0]), set(nodes[initial_partition == 1])]
    return coarsestBisumulation(initial_partition, vertices, edges, Q)

# Generates padding scheme based on the f_limit countermeasure
def bdk_step2(sets, vertices):
    pad_scheme = {}
    for block_idx in sets:
        vertices_in_block = sets[block_idx]
        max_block_size = 0
        for v in vertices_in_block:
            max_block_size = max(max_block_size, vertices[v])
        pad_scheme[block_idx] = max_block_size

    return pad_scheme


def constructQLinode(edges, destinations):
    Q = {}
    for edge in edges:
        src, dst = edge[0], edge[1]
        Q[(src, dst)] = 1./len(destinations[src])
    return Q


def constructQWikipedia(edges, destinations):
    Q = {}
    for edge in edges:
        src, dst = edge[0], edge[1]
        Q[(src, dst)] = 1./len(destinations[src])
    return Q


def run_bdk(dataset, num_trials=100, save_dir=None):
    vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges, Q = load_dataset(dataset, cap_sequences, cap_length)
    s_seq_counts = load_sequence_counts(dataset)

    max_c_list = []
    mean_pad_factors = []

    min_MI_envelope = [math.inf] * max_length
    max_MI_envelope = [-math.inf] * max_length

    for trial in range(num_trials):
        sum_pad_fac, len_pad_fac = 0, 0

        sets, B_tracker = bdk_step1(vertices, edges, Q)
        pad_scheme_blocks = bdk_step2(sets, vertices)

        max_c = 0

        pad_scheme = {}
        for v, block_idx in B_tracker.items():
            pad_scheme[v] = [(pad_scheme_blocks[block_idx],1.0)]
            if v != 'root':
                pad_factor = pad_scheme_blocks[block_idx]/vertices[v]
                max_c = max(max_c, pad_factor)
                sum_pad_fac += pad_factor
                len_pad_fac += 1
                
        max_c_list.append(max_c)
        mean_pad_factors.append(sum_pad_fac / len_pad_fac)
    
        for tgt_length in range(1, max_length+1):
            y_seq_counts = defaultdict(float)
        
            for seq in sequences:
                main_entropy(seq, pad_scheme, s_seq_counts, y_seq_counts, tgt_length)
        
            entropy_calc = 0.0
            total_count = sum(y_seq_counts.values())
        
            for y, seq_count in y_seq_counts.items():     
                prob = seq_count / total_count
                if prob > 0:
                    entropy_calc -= prob * math.log2(prob)

            min_MI_envelope[tgt_length - 1] = min(min_MI_envelope[tgt_length - 1], entropy_calc)
            max_MI_envelope[tgt_length - 1] = max(max_MI_envelope[tgt_length - 1], entropy_calc)
        
        print(f"Completed BDK Run {trial}")
        

    if save_dir is not None:
        with open(f'{save_dir}/bdk_{dataset}.csv', 'a') as f:
            spamwriter = csv.writer(f)
            spamwriter.writerow([f'backes_min_envelope', min_MI_envelope])
            spamwriter.writerow([f'backes_max_envelope', max_MI_envelope])
            spamwriter.writerow([f'backes_c_list', max_c_list])

    print(f'Completed {num_trials} runs of BDK.')
    
    i_inf_res = []
    for tgt_length in range(1, max_length+1):
        max_probs = defaultdict(float)

        for seq in sequences:
            main_tgt_length(seq, pad_scheme, max_probs, tgt_length)

        i_inf = math.log2(sum(max_probs.values()))

        i_inf_res.append(i_inf)

    return {
        'pad_scheme':pad_scheme, 
        'mutual_inf': (min_MI_envelope, max_MI_envelope), 
        'pad_factors': (max_c_list, mean_pad_factors),
        'i_inf': i_inf_res
    }

cap_sequences = False
cap_length = 4 # if cap_sequences is enabled, then this will be the truncated length

if __name__ == "__main__":
    dataset = 'wikipedia'
    run_bdk(dataset)