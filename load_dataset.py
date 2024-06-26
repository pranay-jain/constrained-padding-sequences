import pandas as pd
import numpy as np
import csv
import json
from collections import defaultdict

def load_dataset(dataset, cap_sequences, cap_length):
    
    # load Autocomplete
    if dataset == 'autocomplete':
        df_nodes = pd.read_csv("data/autocomplete_dataset/search_results.csv")
        df_q = pd.read_csv("data/autocomplete_dataset/transition_matrix.csv", index_col=0)

        vertices = dict(zip(df_nodes['Query'], df_nodes['AutocompletePacketSize']))
        vertices['root'] = 0
        edges = list(zip(df_q['from'], df_q['to']))

        full_words = df_nodes.query('NumSearchResults > 0')['Query'].tolist()

        sequences = []
        for word in full_words:
            seq = []
            for i in range(1,len(word)+1):
                seq.append(word[:i])

            sequences.append(seq)
            
        Q = {(row['from'], row['to']): row['percentage'] for _, row in df_q.iterrows()}
                
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
                
        sources, destinations = constructSourcesDestinations(vertices, edges)
        Q = constructQLinode(edges, destinations)
    
    # load Wikipedia
    if dataset == 'wikipedia':
        vFile = 'data/wikipedia_dataset/vertices.csv'
        eFile = 'data/wikipedia_dataset/edges_sm.csv'
        # seqFile = 'data/wikipedia_dataset/sequences_new.csv'
        seqFile = 'data/wikipedia_dataset/sequences_random_walk.csv'
        
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
        
        counter = defaultdict(int)
        
        with open(seqFile, "r") as file:
            lines = file.read().splitlines()
            
            for line in lines:
                seq = line.split(',')
                counter[seq[0]] += 1
                if len(seq) == 7 and counter[seq[0]] <= 2:
                    sequences.append(seq)

        sources, destinations = constructSourcesDestinations(vertices, edges)
        Q = constructQWikipedia(edges, destinations)    
        
    # load Synthetic
    if dataset == 'synthetic':
        seq_length = 8
        
        vertices = {}
        sequences = []
        
        for row in range(2**seq_length):
            row_binary_string = f'{row:08b}'
            
            new_seq = [0] * seq_length
            
            for column in range(seq_length):
                vertex = str(row) + "-" + str(column)
                
                vertices[vertex] = int(row_binary_string[column]) + 1
                    
                new_seq[column] = vertex
                
            sequences.append(new_seq)
            
        row = 2**seq_length
        new_seq = [0] * seq_length
        for column in range(seq_length):
            vertex = str(row) + "-" + str(column)
            if column == 0:
                vertices[vertex] = 3
            else:
                vertices[vertex] = 1
            new_seq[column] = vertex
        sequences.append(new_seq)
        
        edges = []
        
        for sequence in sequences:
            for column in range(1, seq_length):
                edges.append((sequence[column-1], sequence[column]))
        
        sources, destinations = constructSourcesDestinations(vertices, edges)
        Q = None

    # load Synthetic2
    if dataset == 'synthetic2':
        seq_length = 8
        
        vertices = {}
        sequences = []
        
        row_idx = -1
        
        for row in range(2**seq_length):
            row_idx += 1
            
            row_binary_string = f'{row:08b}'
            
            new_seq = [0] * seq_length
            
            for column in range(seq_length):
                vertex = str(row_idx) + "-" + str(column)
                
                vertices[vertex] = int(row_binary_string[column]) + 1
                    
                new_seq[column] = vertex
                
            sequences.append(new_seq)
            
        row_idx += 1
        new_seq = [0] * seq_length
        for column in range(seq_length):
            vertex = str(row_idx) + "-" + str(column)
            
            vertices[vertex] = 3

            new_seq[column] = vertex
        sequences.append(new_seq)
        
        for row in range(2**seq_length):
            row_idx += 1
            
            row_binary_string = f'{row:08b}'
            
            new_seq = [0] * seq_length
            
            for column in range(seq_length):
                vertex = str(row_idx) + "-" + str(column)
                
                vertices[vertex] = int(row_binary_string[column]) + 20
                    
                new_seq[column] = vertex
                
            sequences.append(new_seq)
            
        row_idx += 1
        new_seq = [0] * seq_length
        for column in range(seq_length):
            vertex = str(row_idx) + "-" + str(column)
            
            vertices[vertex] = 10

            new_seq[column] = vertex
        sequences.append(new_seq)
        
        edges = []
        
        for sequence in sequences:
            for column in range(1, seq_length):
                edges.append((sequence[column-1], sequence[column]))
        
        sources, destinations = constructSourcesDestinations(vertices, edges)
        Q = None 
        
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

    return vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges, Q

def load_sequence_counts(dataset):
    if dataset == "autocomplete":
        return load_sequence_counts_autocomplete()
    elif dataset == 'wikipedia':
        return load_sequence_counts_wiki()
    elif dataset == "linode_from_index":
        return load_sequence_counts_linode_from_index()
    else: 
        return None


# -------- Helper Functions

def seq_to_idx(seq):
    return '~'.join(seq)


def load_sequence_counts_autocomplete():
    df_nodes = pd.read_csv("data/autocomplete_dataset/search_results.csv")

    df_full_word_counts = df_nodes.query('NumSearchResults > 0')[['Query','NumSearchResults']]
    
    s_seq_counts = {}
    
    for index, row in df_full_word_counts.iterrows():
        seq = []
        for i in range(1,len(row['Query'])+1):
            seq.append(row['Query'][:i])
            
        seq = tuple(seq)
        s_seq_counts[seq] = row['NumSearchResults']
        
    return s_seq_counts


def load_wiki_dataset(past: bool, cap_sequences: bool = False):
    # df_wiki_nodes = pd.read_csv(f"data/wikipedia_dataset/{'past_' if past else ''}vertices.csv")
    # df_wiki_nodes.columns=['page', 'size']

    # df_edges = pd.read_csv(f"data/wikipedia_dataset/{'past_' if past else ''}edges_sm.csv", header=None)
    # df_edges.columns = ['src', 'dst']

    # vertices = dict(zip(df_wiki_nodes['page'], df_wiki_nodes['size']))
    # edges = set(list(zip(df_edges['src'], df_edges['dst'])))
    # print(f"edges: {len(list(edges))}")
   
    df_wiki_weights = pd.read_csv("data/wikipedia_dataset/weights.csv")
    df_wiki_weights.columns = ['v', 'w']

    wiki_weights = dict(zip(df_wiki_weights['v'], df_wiki_weights['w']))

    sequences = []
    # with open('data/wikipedia_dataset/sequences_new.csv', "r") as file:
    with open('data/wikipedia_dataset/sequences_random_walk.csv', "r") as file:
        lines = file.read().splitlines()
        
        for line in lines:
            seq = line.split(',')
            if len(seq) == 7:          ## !!! leaving this here for now ... we can reconsider this though !!!
                sequences.append(seq)

    return sequences, wiki_weights
    # return (vertices, list(edges), sequences, wiki_weights)


def load_sequence_counts_wiki(test_sequences=None):
    sequences, page_views = load_wiki_dataset(past=False)
    if test_sequences is None:
        test_sequences = sequences
    
    for page, views in page_views.items():
        if views == -1:
            page_views[page] = 0

    s_seq_counts = {}  

    for seq in test_sequences:     
        sum_of_page_views = 0
        for page in seq:
            sum_of_page_views += page_views[page]
        s_seq_counts[tuple(seq)] = sum_of_page_views / len(seq) # using the average

    # return s_seq_counts
    
    # switch: equal seq weights
    s_seq_counts = {tuple(seq): 1 for seq in test_sequences}

    total_p_s = sum(list(s_seq_counts.values()))

    P_S = { k: v / total_p_s for k, v in s_seq_counts.items() }

    return P_S


def load_sequence_counts_linode_from_index():
    seqFile = 'data/linode/linode_sequences_from_index.csv'
        
    sequences = []
    with open(seqFile, "r") as file:
        lines = file.read().splitlines()
            
        for line in lines:
            sequences.append(line.split(','))
            
    s_seq_counts = {tuple(seq): 1 for seq in sequences}
    
    return s_seq_counts


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