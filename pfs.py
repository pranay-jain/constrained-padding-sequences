import gurobipy as gp
import numpy as np
import math
import time
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict

from load_dataset import load_dataset, load_sequence_counts
from utils import create_strides, main_tgt_length, main_2

cap_sequences = False
cap_length = 4 # if cap_sequences is enabled, then this will be the truncated length

thread_count = 16
gurobi_output = 0 # 0 = don't show; 1 = show

def run_pfs(dataset, c, k=2, prefix_closed=True):
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
    
    return pad_scheme, i_inf_res


def run_pfg(dataset, c, k=2):
    vertices, vertices_subset, sequences, prefix_closed_sequences, max_length, edges, Q = load_dataset(dataset, cap_sequences, cap_length)
    s_seq_counts = load_sequence_counts(dataset)

    edges = set()

    for seq in sequences:
        for i in range(len(seq)-1):
            edges.add((seq[i], seq[i+1]))
            edges.add((seq[i],))
            edges.add((seq[i+1],))

    edges = list(list(x) for x in edges)

    uniq_sizes, stride_uniq_sizes = create_strides(vertices, c, k)

    lp, valid_ranges = linear_program(c, vertices, edges, uniq_sizes, stride_uniq_sizes, max_length)
    
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
    
    return pad_scheme, i_inf_res


def linear_program(c, v_dict, e_list, u_sizes, stride_u_sizes, max_length):    
   
    # Create the model
    m = gp.Model("Sequences")
    m.setParam("OutputFlag", gurobi_output)
    m.setParam("Threads", thread_count)
      
    cndl_list = []
    poss_sizes = {v: [] for v in v_dict.keys()}

    for v in v_dict.keys():
        orig_size = v_dict[v]
        
        max_s = c * orig_size
        for s in stride_u_sizes:
            if (orig_size <= s) and (s <= max_s):
                cndl_list.append((v,s))
                poss_sizes[v].append(s)
            if s > max_s:
                break
                                               
    p_y_s = m.addVars(cndl_list, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="pys")
    m.addConstrs((p_y_s.sum(v,'*') == 1 for v in v_dict.keys()))
    
    Pi_y_constrs = {}
    
    for edge in e_list:
        main_2(edge, poss_sizes, Pi_y_constrs, max_length)
                                             
    Pi_y = m.addVars(Pi_y_constrs.keys(), lb=0.0, ub=1.0)
    
    matrix_index = {}
    i = 0
    
    for Pi_y_tuple in Pi_y_constrs.keys():
        matrix_index[Pi_y_tuple] = i
        i += 1
        
    for cndl in cndl_list:
        matrix_index[cndl] = i
        i += 1
        
    num_constrs = 0
    for constr_list in Pi_y_constrs.values():
        num_constrs += len(constr_list)
    
    rhs = np.zeros(num_constrs)
    
    l = [0] * len(matrix_index)
    
    for Pi_y_tuple in Pi_y_constrs.keys():
        l[matrix_index[Pi_y_tuple]] = Pi_y[Pi_y_tuple]
        
    for cndl in cndl_list:
        l[matrix_index[cndl]] = p_y_s[cndl]

    x = gp.MVar.fromlist(l)
    
    A = lil_matrix((num_constrs,len(matrix_index)), dtype=int)
    
    constr_index = 0
    
    for seq in Pi_y_constrs.keys():
        
        for cndl_counts in Pi_y_constrs[seq]:
            total = 0
            
            for cndl, count in cndl_counts.items():
                A[constr_index,matrix_index[cndl]] = -count 
                total += count
                
            A[constr_index,matrix_index[seq]] = total
                
            constr_index += 1

    A = csr_matrix(A)
    expr = A @ x
    m.addConstr(expr >= rhs)
            
    m.setObjective(Pi_y.sum(), gp.GRB.MINIMIZE)

    start_time = time.time()
    m.optimize()
    end_time = time.time()
            
    print("Gurobi's optimization method runtime (in seconds): " + str(end_time - start_time))
    
    return m, poss_sizes
