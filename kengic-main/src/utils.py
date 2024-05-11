import os
import argparse
import json
import copy
import pandas as pd
import numpy as np
import time
import multiprocessing
import collections
import shutil
import math
import logging
import nlp
import pickle
import re
from nltk.translate.bleu_score import sentence_bleu

def convert_ngram_dict_to_df(ngrams):
    ngrams_dfs = {}

    for ni in range(2, 5):
        columns = [i for i in range(1,ni+1)]
        # print(columns)
        ngrams_df = pd.DataFrame(ngrams[ni], columns=columns)
        # print(ngrams_df.head())
        ngrams_df['count'] = 0
        # print(ngrams_df.head())
        ngrams_df = ngrams_df.groupby(columns).count().sort_values(['count'], ascending=False).reset_index()
        # print(ngrams_df.head())
        ngrams_df['probability'] = ngrams_df['count']/ngrams_df['count'].sum()
        # print(ngrams_df.head())
        ngrams_dfs[ni] = ngrams_df

    return ngrams_dfs

def ngrams(x, n):
    ngrams = []
    for idx, ix in enumerate(x[0:len(x)-n+1]):
        #print(idx, ix)
        gram = x[idx:idx+n]
        ngrams += [gram]
        
    return ngrams

class kengic ():
    def __init__(self, configs, out_folder,input_csv_file_path,top_n_captions=5):
        self.configs = configs
        self.out_folder = out_folder
        self.input_csv_file_path = input_csv_file_path
        self.top_n_captions = top_n_captions
        self.references = []
        input = pd.read_csv(self.input_csv_file_path)
        self.img_id = input['img_id'].values[0]
        self.keywords = input['keywords'].values[0]
        self.data,self.ngram_dfs = self.load_data()
    
    def load_data(self):
        data = pd.read_csv('./indiana_reports_cleaned2.csv')
        self.references= data[data['imgID'] == self.img_id]['captions'].values[0]
        ngrams_dic = {}
        try:
            print('Loading ngrams_dic.pkl...')
            with open('ngrams_dic.pkl','rb') as f:
                ngrams_dic = pickle.load(f)
        except:
            print('Failed to Load ngrams_dic.pkl...')

            ngrams_dic= self.create_ngrams(data) #bet return kol el tokens w ngrams_dic
            #save ngrams_dic bta3 kaza no3 ngram mn 1-9
        
            print('saving ngrams_dic.pkl...')
            with open('ngrams_dic.pkl', 'wb') as f:
                pickle.dump(ngrams_dic, f)  
        ngrams_dfs = convert_ngram_dict_to_df(ngrams_dic)
        return data,ngrams_dfs
    
    def create_ngrams(self, data):
        ngrams_dict = {}
        ngrams_2= []
        ngrams_3= []
        ngrams_4= []
        ngrams_5= []
        for idx,row in data.iterrows():
            row = eval(row["captions"])
            for caption in row:

                ngrams_2 += ngrams(['<t>'] + caption + ['</t>'], 2)
                ngrams_3 += ngrams(['<t>', '<t>'] + caption + ['</t>'], 3)
                ngrams_4 += ngrams(['<t>', '<t>', '<t>'] + caption + ['</t>'], 4)
                ngrams_5 += ngrams(['<t>', '<t>', '<t>', '<t>'] + caption + ['</t>'], 5)

        ngrams_dict[2] = ngrams_2
        ngrams_dict[3] = ngrams_3
        ngrams_dict[4] = ngrams_4
        ngrams_dict[5] = ngrams_5
        return ngrams_dict
    
    def get_parents(self,n,graph_node,parents):
        ngram_data =self.ngram_dfs[n]
        graph_node = graph_node.split()[0] #we get the first word of the graph node to increase the probability of finding the parent
        parents_data = ngram_data[ngram_data[n] == graph_node]
        parents_data['from_prob'] = parents_data['count']/float(parents_data['count'].sum()) #probability of phrase of being a parent to this graph node
        parents_data = parents_data.sort_values('from_prob', ascending=False)[0:parents]
        return parents_data

    def create_graph(self,keywords,n,parents,hops):
        queue = keywords
        hop = 0
        columns = [i for i in range(1, n+1)]
        graph = {}
        edges = {}

        while hop < hops and len(queue) > 0:
            current_parent = queue[0].split()
            
            if len(current_parent) == 1:
                current_node = current_parent[0]
            else:
                current_node = ' '.join(current_parent[0:-1])
            
            graph_node = queue[0]
            queue = queue[1:]
            if '{' in graph_node:
                idx_from = graph_node.index('{')
                idx_to = graph_node.index('}')
                current_hop = int(graph_node[idx_from+1:idx_to])
                if current_hop > hop:
                    hop = hop + 1
                continue
            
            parents_data = self.get_parents(n, graph_node, parents) #gab 23la 5 probabilities were en el kelma fe 25r el ngram   
            captions = parents_data[columns].apply(lambda x: ' '.join(x), axis=1).values #joining the n-gram into one sentence

            if current_node not in graph.keys():
                graph[current_node] = []

            for cap_idx, cap in enumerate(captions):
                parent_node = ' '.join(cap.split()[0:-1])
                from_prob = parents_data[cap_idx:cap_idx+1]['from_prob'].values[0]
                edges[parent_node + ':' + current_node] = from_prob

                graph[parent_node] = [current_node]

            queue += ['{' + str(hop+1) + '}'] + list(captions)
        
        return graph, edges

    def caption_generation(self):
        keywords = self.keywords
        if not (keywords == None or keywords == '[]'):
            keywords = eval(keywords)
            n = self.configs[0]
            parents = self.configs[1]
            hops = self.configs[2]
            nlogP = self.configs[3]
            optimiser = self.configs[4]
            max_iters = self.configs[5]

            graph, edges = self.create_graph(keywords, n, parents, hops)



            

        
