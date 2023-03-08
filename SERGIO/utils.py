import numpy as np
import scanpy as sc
import pandas as pd
import csv
import networkx as nx
#from Sergio txt to networkx structure
def convert_interaction_net_to_networkx(input_file_taregts):
    '''This function converts the input text file where interaction is stored into a networkx object'''
    G = nx.DiGraph()
    with open(input_file_taregts) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            unformatted = line[0].split(',')
            i = float(unformatted[0])
            in_degree = float(unformatted[1])
            if i.is_integer()&in_degree.is_integer():
                i = int(i)
                in_degree = int(in_degree)
            else:
                raise ValueError('1st and 2nd column of '+input_file_taregts+" should be integers, but they aren't")
            neighbours  = np.rint(np.array(unformatted[2:2+in_degree],dtype=float)).astype(int)
            interaction = np.array(unformatted[2+in_degree:2+2*in_degree],dtype = float)
            G.add_weighted_edges_from(list(zip(neighbours,[i]*in_degree,interaction)))#neighbours are the genes pointing to i
        return G

