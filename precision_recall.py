import csv
import pandas as pd
import json
from random import sample
from load_dataset import load_sequence_counts_wiki, load_wiki_dataset, load_sequence_counts_linode_from_index
from collections import Counter


# ----- CONSTANTS --------
TAU_RANGE = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 0.8, 0.9, 1]
EXPT_COUNT = 10


# ---------- Helper Functions ----------

def load_autocomplete_dataset():
    df_nodes = pd.read_csv("data/autocomplete_dataset/search_results.csv")
    df_q = pd.read_csv("data/autocomplete_dataset/transition_matrix.csv", index_col=0)
    edges = list(zip(df_q['from'], df_q['to']))

    words = list(df_nodes[df_nodes['NumSearchResults'] > 0]['Query'])
    words_len = 7

    eligible_words = [word for word in words if len(word) == words_len]
    print(len(eligible_words))

    vertices = dict(zip(df_nodes['Query'], df_nodes['AutocompletePacketSize']))
    return (vertices, edges, words)

def get_padding_sequence(word, pad_scheme):
    pad_seq = []
    c = ''
    for i in range(len(word)):
        c += word[i]
        pad_seq.append(pad_scheme[c])
    return pad_seq


def compute_adversary_precision_recall(adversary, tau, target_set_pos, target_set_neg):
    tp, fp, tn, fn = 0, 0, 0, 0
    for y_seq, prob_in_target_set in adversary.items():
        if prob_in_target_set >= tau:
            # adversary predicts positive
            tp += target_set_pos[y_seq] if y_seq in target_set_pos else 0
            fp += target_set_neg[y_seq] if y_seq in target_set_neg else 0
            # print(f"tau: {tau}, tp: {tp}, fp: {fp}, prob_in_target_set: {prob_in_target_set}")
        else:
            # adversary predicts negative
            fn += target_set_pos[y_seq] if y_seq in target_set_pos else 0
            tn += target_set_neg[y_seq] if y_seq in target_set_neg else 0
            # print(f"tau: {tau}, tn: {tn}, fn: {fn}, prob_in_target_set: {prob_in_target_set}")
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    # print(f'tp {tp}, fp {fp}, fn {fn}, tn {tn}')

    return (precision, recall)


def rec_pr(s, pad_s, search_results, cur_sizes, i, prev_prob, total_probs, dataset="autocomplete", target_len=7):
    cur_v = s[i]
    for size, prob in pad_s[cur_v]:
        cur_prob = prev_prob * prob
        cur_sizes[i] = size

        padded_size_seq = tuple(cur_sizes)
        
        if i == target_len - 1 or i == len(s) - 1:
            # SWITCH:
            if dataset == "autocomplete":
                pr_s = search_results[cur_v]     # ---> Autocomplete
            else:
                pr_s = search_results[tuple(s)]    # ---> Wiki or LInode

            total_probs[padded_size_seq] = total_probs.get(padded_size_seq, 1) * (pr_s * cur_prob)
        else:
            rec_pr(s, pad_s, search_results, cur_sizes, i+1, cur_prob, total_probs, dataset, target_len)

            
def main(sequence, search_results, pad_s, total_probs, dataset, target_len=7):
    rec_pr(sequence, pad_s, search_results, [0]*len(sequence), 0, 1.0, total_probs, dataset, target_len)


def word_to_sequence(word):
    # words = list(df_nodes[df_nodes['NumSearchResults'] > 0]['Query'])

    # print(f"Number of words: {len(words)}")

    # sequence = []
    # for word in words:
    prefix = ''
    word_seq = []
    for c in word:
        prefix += c
        word_seq.append(prefix)
    # sequence.append(word_seq)

    return word_seq


def load_test_words(seq_len):
    _, _, words = load_autocomplete_dataset()
    target_set_size = int(len(words) * 0.05)
    target_words = sample(words, target_set_size)

    search_results = {}
    with open('data/autocomplete_dataset/suffix_search_results.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            search_results[row[0]] = int(row[1])

    test_words = [w[:seq_len] for w in words if len(w) >= seq_len]
    word_in_target_set = { w:1 if w in target_words else 0 for w in test_words }
    
    print(Counter([len(w) for w in test_words]))

    print(f"Target set size: {target_set_size}; test set size: {len(test_words)}")
    return (test_words, word_in_target_set, search_results)


# ---------- Precision Recall HELPER Functions ----------

def precision_recall(pad_scheme_flat, test_seqs, seq_in_target_set, p_s, dataset, method, seq_len=7):
    # test_seqs, seq_in_target_set, p_s = load_test_words(seq_len)
    target_words = set([ w for w, res in seq_in_target_set.items() if res == 1])

    test_seqs = list(set(test_seqs))
    print(f"Len test seqs: {len(test_seqs)}")

    p_y_s_times_p_s, totals_y = {}, {}
    observed_sequences = set()

    # for each y_seq, track probability of an input sequence to
    #     belong(pos) / not belong(neg) to the target set 
    target_set_pos, target_set_neg = {}, {}

    for test_seq in test_seqs:
        if dataset == 'autocomplete':
            y_seq = tuple(get_padding_sequence(test_seq, pad_scheme_flat))
        else:
            if "L-Diversity" in method:
                y_seq = pad_scheme_flat['~'.join(list(test_seq))] # --> use this for l-diversity
            else:
                y_seq = tuple([pad_scheme_flat[v] for v in test_seq]) # --> use this for other algorithms
        
        y_s = (test_seq, y_seq)

        observed_sequences.add(y_seq)
        p_y_s_times_p_s[y_s] = p_s[test_seq]
        
        if y_seq not in totals_y:
            totals_y[y_seq] = 0
        
        totals_y[y_seq] += p_y_s_times_p_s[y_s]

        if seq_in_target_set[test_seq] == 1:
            target_set_pos[y_seq] = target_set_pos.get(y_seq, 0) + (1 * p_y_s_times_p_s[y_s])
        else:
            target_set_neg[y_seq] = target_set_neg.get(y_seq, 0) + (1 * p_y_s_times_p_s[y_s])

    p_s_y = { s_y: p_y_s_times_p_s[s_y] / totals_y[s_y[1]] for s_y in p_y_s_times_p_s }

    adversary = {}

    # print(f"Len observed: {len(observed_sequences)}")
    for seq in observed_sequences:
        for target_word in target_words:
            if (target_word, seq) in p_s_y:
                if seq not in adversary:
                    adversary[seq] = 0
                adversary[seq] += p_s_y[(target_word, seq)]
    # print(f"Len adversary: {len(adversary)} {len(p_s_y)}")
    # print(sum(p_s_y.values()) / len(p_s_y.values()))
    # print(list(p_s_y.keys())[0])

    recall_precision_mp = {}
    for tau in TAU_RANGE:
        precision, recall = compute_adversary_precision_recall(adversary, tau, target_set_pos, target_set_neg)
        recall_precision_mp[tau] = (recall, precision)
        # recall_precision_mp[recall] = precision
        print(f'Tau: {tau}; precision={precision}, recall={recall}')

    return recall_precision_mp


def precision_recall_per_req(pad_scheme_flat, test_seqs, seq_in_target_set, p_s, dataset='autocomplete', seq_len=4):
    target_words = set([ w for w, res in seq_in_target_set.items() if res == 1])
    print(len(target_words))

    test_seqs = list(set(test_seqs))

    p_y_s_times_p_s, totals_y = {}, {}
    observed_sequences = set()

    # for each y_seq, track probability of an input sequence to
    #     belong(pos) / not belong(neg) to the target set 
    target_set_pos, target_set_neg = {}, {}

    for test_seq in test_seqs:
        total_probs = {}
        if dataset == "autocomplete":
            seq = word_to_sequence(test_seq)    # --> autocomplete
        else:
            seq = test_seq                        # --> Linode or Wiki
        
        main(seq, p_s, pad_scheme_flat, total_probs, dataset, target_len=seq_len)

        for y_seq in total_probs:
            # if dataset == 'autocomplete':
            #     y_seq = tuple(get_padding_sequence(test_seq, pad_scheme_flat))
            # else:
            #     y_seq = tuple([pad_scheme_flat[v] for v in test_seq])

            y_s = (test_seq, y_seq)

            observed_sequences.add(y_seq)
            p_y_s_times_p_s[y_s] = total_probs[y_seq]
        
            if y_seq not in totals_y:
                totals_y[y_seq] = 0
        
            totals_y[y_seq] += p_y_s_times_p_s[y_s]

            if seq_in_target_set[test_seq] == 1:
                target_set_pos[y_seq] = target_set_pos.get(y_seq, 0) + (1 * p_y_s_times_p_s[y_s])
            else:
                target_set_neg[y_seq] = target_set_neg.get(y_seq, 0) + (1 * p_y_s_times_p_s[y_s])

    p_s_y = { s_y: p_y_s_times_p_s[s_y] / totals_y[s_y[1]] for s_y in p_y_s_times_p_s }
    
    adversary = {}

    for seq in observed_sequences:
        for target_word in target_words:
            if (target_word, seq) in p_s_y:
                if seq not in adversary:
                    adversary[seq] = 0
                adversary[seq] += p_s_y[(target_word, seq)]

    recall_precision_mp = {}
    for tau in TAU_RANGE:
        precision, recall = compute_adversary_precision_recall(adversary, tau, target_set_pos, target_set_neg)
        # recall_precision_mp[recall] = precision
        recall_precision_mp[tau] = (recall, precision)
        print(f'Tau: {tau}; precision={precision}, recall={recall}')

    return recall_precision_mp

# ---------- Precision Recall EXPORTED (main) Functions ----------

def precision_recall_autcomplete(pad_scheme_flat: dict, seq_len: int = 7, method: str = "LP"):
    results = []
    for _ in range(EXPT_COUNT):
        test_words, word_in_target_set, search_results = load_test_words(seq_len)
        # print(f"number of words: {len(test_words)}")
        if 'lp' in method.lower():
            recall_precision_mp = precision_recall_per_req(
                pad_scheme_flat, test_words, word_in_target_set, search_results, dataset="autocomplete", seq_len=seq_len)
        else:
            recall_precision_mp =  precision_recall(
                pad_scheme_flat, test_words, word_in_target_set, search_results, "autocomplete", method, seq_len=seq_len)
        results.append(recall_precision_mp)
    
    return average_recall_precision(results, method, seq_len, "autocomplete")


def precision_recall_wiki(pad_scheme_flat: dict, seq_len: int = 7, method: str = "LP"):
    sequences, _ = load_wiki_dataset(past=False)
    test_seqs = list(set([tuple(seq[:seq_len]) for seq in sequences]))

    p_s = load_sequence_counts_wiki(test_sequences=test_seqs)

    results = []
    for _ in range(EXPT_COUNT):
        target_set = set(sample(test_seqs, len(test_seqs) // 20))
        seq_in_target_set = { seq: 1 if seq in target_set else 0 for seq in test_seqs }
        print(f"len test: {len(test_seqs)}; len of target: {len(target_set)}")
        # print(target_set)

        target_p_s = 0
        for s, res in seq_in_target_set.items():
            if res == 1:
                target_p_s += p_s[s]
        
        print(target_p_s)

        if 'lp' in method.lower():
            recall_precision_mp = precision_recall_per_req(pad_scheme_flat, test_seqs, seq_in_target_set, p_s, 'wiki', seq_len=seq_len)
        else:
            recall_precision_mp =  precision_recall(pad_scheme_flat, test_seqs, seq_in_target_set, p_s, 'wiki', method, seq_len=seq_len)
        
        results.append(recall_precision_mp)
    
    return average_recall_precision(results, method, seq_len, 'wiki')

    # with open(f'../experiments/results/recall_precision_seq{seq_len}_wiki.csv', 'a') as f:
    #     spamwriter = csv.writer(f)
    #     spamwriter.writerow([method, json.dumps(recall_precision_mp)])


def precision_recall_linode_from_index(pad_scheme_flat: dict, seq_len: int = 4, method: str = "LP"):
    vertices, vertices_subset, dataset_sequences, prefix_closed_sequences, max_length, edges = load_dataset(
        'linode_from_index', cap_sequences=True, cap_length=seq_len)
    
    test_seqs = list(set([tuple(seq[:seq_len]) for seq in dataset_sequences]))

    p_s = load_sequence_counts_linode_from_index()
    # p_s = load_sequence_counts_linode_from_index(test_sequences=test_seqs)

    results = []
    for _ in range(EXPT_COUNT):
        target_set = set(sample(test_seqs, len(test_seqs) // 10))
        seq_in_target_set = { seq: 1 if seq in target_set else 0 for seq in test_seqs }

        if 'lp' in method.lower():
            recall_precision_mp = precision_recall_per_req(pad_scheme_flat, test_seqs, seq_in_target_set, p_s, dataset='linode_from_index', seq_len=seq_len)
        else:
            recall_precision_mp =  precision_recall(pad_scheme_flat, test_seqs, seq_in_target_set, p_s, 'linode_from_index', method, seq_len=seq_len)
        
        results.append(recall_precision_mp)
    return average_recall_precision(results, method, seq_len, 'linode_from_index')

# ---------------------------------------
# ----     LINODE DATASET 
# ---------------------------------------


def load_dataset(dataset, cap_sequences, cap_length):
    # load Linode (from Index)
    if dataset == 'linode_from_index':
        vFile = 'data/linode/vertices_no_errors.csv'
        oFile = 'data/linode/object_lists_short.txt'
        eFile = 'data/linode/edges_no_errors.csv'
        seqFile = 'data/linode/linode_sequences_from_index.csv'
        
        edges = pd.read_csv(eFile).to_records(index=False)

        vertexList = (pd.read_csv(vFile))['URL'].tolist()
        with open(oFile) as f:
            objectLists = json.load(f)
    
        vertices = {}

        for url in vertexList:
            objList = objectLists[url]
            total = 0
            for obj in objList:
                total += int(obj[2])
        
            vertices[url] = total
    
        vertices = dict(sorted(vertices.items(), key=lambda item: item[1]))
    
        #sequences = pd.read_csv(eFile).to_records(index=False)
        
        sequences = []
        with open(seqFile, "r") as file:
            lines = file.read().splitlines()
            
            for line in lines:
                sequences.append(line.split(','))
                
        # sources, destinations = constructSourcesDestinations(vertices, edges)
        # Q = constructQLinode(edges, destinations)
    
    # load Wikipedia
    if dataset == 'wikipedia':
        vFile = 'data/wikipedia_dataset/vertices.csv'
        eFile = 'data/wikipedia_dataset/edges.csv'
        # seqFile = '../wikipedia_dataset/sequences_new.csv'
        seqFile = 'data/wikipedia_dataset/sequences_random_walks.csv'
        
        edges = pd.read_csv(eFile, header=None).to_records(index=False)

        vertices = {}
        with open(vFile) as csvfile:
            content = csv.reader(csvfile)
            for row in content:
                if row[0] == '':
                    continue
                vertices[row[0]] = int(row[1])
        
        vertices = dict(sorted(vertices.items(), key=lambda item: item[1]))

        #sequences = pd.read_csv(eFile,header=None).to_records(index=False)
        
        sequences = []
        with open(seqFile, "r") as file:
            lines = file.read().splitlines()
            
            for line in lines:
                seq = line.split(',')
                if len(seq) == 7:
                    sequences.append(seq)
                    
        # sources, destinations = constructSourcesDestinations(vertices, edges)
        # Q = constructQWikipedia(edges, destinations)    
        
    # this block truncates sequences if enabled    
    if cap_sequences:
        trunc_sequences = []
    
        for seq in sequences:
            trunc_sequences.append(seq[:cap_length])
            
        sequences = trunc_sequences
        
    # this tracks which vertices are actually included in the sequences
    # (not using it now, but we may need it)
    vertices_subset = {}
        
    for seq in sequences:
        for v in seq:
            vertices_subset[v] = vertices[v]  
                
    # max_length is needed inside the LP
    max_length = 0
    for seq in sequences:
        max_length = max(max_length, len(seq))
        
    # create the prefix-closed set of sequences    
    prefix_closed_set = set()
    
    for seq in sequences:
        for i in range(1,len(seq)+1):
            prefix_closed_set.add(tuple(seq[:i]))
            
    prefix_closed_sequences = []
    for seq in prefix_closed_set:
        prefix_closed_sequences.append(list(seq))
            
    return vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges

    
    

def average_recall_precision(results, method, seq_len, dataset):
    recall_precision_avg = {}
    for tau in TAU_RANGE:
        count = 0
        precision_sum, recall_sum = 0, 0
        for result in results:
            if result[tau][0] > 0 or result[tau][1] > 0:
                count += 1 
                recall_sum += result[tau][0]
                precision_sum += result[tau][1]
        
        if count > 0:
            recall_precision_avg[recall_sum / count] = precision_sum / count
    return recall_precision_avg


def compute_precision_recall_ldiv_pad_scheme(l, pad_scheme, seq_len, dataset, sequences = None):
    pad = { key: val[0][0] for key, val in pad_scheme.items()}
    print(pad)
    if dataset == 'autocomplete':
        return precision_recall_autcomplete(
            pad, seq_len=seq_len, method=f'L-Diversity(l={l})')
    if dataset == "linode_from_index":
        return precision_recall_linode_from_index(
            pad, seq_len=seq_len, method=f'L-Diversity(l={l})')
    if dataset == 'wikipedia':
        return precision_recall_wiki(pad, seq_len=seq_len, method=f"L-Diversity(l={l})")