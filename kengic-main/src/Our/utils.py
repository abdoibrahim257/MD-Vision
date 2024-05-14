import os
import argparse
import json
import copy
import pandas as pd
import numpy as np
import time
import collections
import shutil
import math
import logging
import pickle
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

def convert_ngram_dict_to_df(ngrams,n):

    columns = [i for i in range(1,n+1)]
    ngrams_df = pd.DataFrame(ngrams, columns=columns)
    ngrams_df['count'] = 0
    ngrams_df = ngrams_df.groupby(columns).count()
    ngrams_df = ngrams_df.sort_values(['count'], ascending=False).reset_index()
    ngrams_df['probability'] = ngrams_df['count']/ngrams_df['count'].sum()
    return ngrams_df

def ngrams(x, n): #Algorithm 1
    ngrams = []
    for idx, ix in enumerate(x[0:len(x)-n+1]):
        gram = x[idx:idx+n]
        ngrams += [gram]
    return ngrams

def create_ngrams(data,n=3):
    ngrams_list= []
    for idx,row in data.iterrows():
        row = eval(row["captions"]) # eval converts this "[alo]" to ["alo"] 
        for caption in row:
            if (n==2):
                ngrams_list += ngrams(['<t>'] + caption + ['</t>'], 2)
            elif (n==3):
                ngrams_list += ngrams(['<t>', '<t>'] + caption + ['</t>'], 3)
            elif (n==4):
                ngrams_list += ngrams(['<t>', '<t>', '<t>'] + caption + ['</t>'], 4)
            elif (n==5):
                ngrams_list += ngrams(['<t>', '<t>', '<t>', '<t>'] + caption + ['</t>'], 5)

    return ngrams_list

def load_data(img_id,n=3):
        data = pd.read_csv('./indiana_reports_cleaned2.csv')
        references = data[data['imgID'] == img_id].values[0]
        try:
            print('Loading ngrams_dic.pkl...')
            with open('ngrams_dic.pkl','rb') as f:
                ngrams_dic = pickle.load(f)
        except:
            print('Failed to Load ngrams_dic.pkl...')

            ngrams_dic= create_ngrams(data,n) #bet return kol el tokens w ngrams_dic
            #save ngrams_dic bta3 kaza no3 ngram mn 1-9
        
            print('saving ngrams_dic.pkl...')
            with open('ngrams_dic.pkl', 'wb') as f:
                pickle.dump(ngrams_dic, f)  
        ngrams_df = convert_ngram_dict_to_df(ngrams_dic,n)
        return data,ngrams_df,references

def get_parents(n,ngram_df,graph_node,parents_no=5):

    ngram_data =ngram_df
    graph_node = graph_node.split()[0] #we get the first word of the graph node to increase the probability of finding the parent
    parents_data = ngram_data[ngram_data[n] == graph_node]
    parents_data['from_prob'] = parents_data['count']/float(parents_data['count'].sum()) #probability of phrase of being a parent to this graph node
    parents_data = parents_data.sort_values('from_prob', ascending=False)[0:parents_no]
    return parents_data

def create_graph(ngram_df,keywords,n,parents,hops):
    queue = keywords
    hop = 0
    columns = [i for i in range(1, n+1)]
    graph = {}
    edges = {}

    while hop < hops and len(queue) > 0:
        current_parent = queue[0].split()
        
        # This is done to get parents of parents in case of hops >0
        if len(current_parent) == 1:
            current_node = current_parent[0]
        else:
            current_node = ' '.join(current_parent[0:-1])
        
        graph_node = queue[0]
        queue = queue[1:]
        if graph_node.isdigit():
            current_hop = int(graph_node)
            if current_hop > hop:
                hop = hop + 1
            continue

        parents_data = get_parents(n,ngram_df,graph_node, parents) #gab 23la 5 probabilities were en el kelma fe 25r el ngram   
        captions = parents_data[columns].apply(lambda x: ' '.join(x), axis=1).values #joining the n-gram into one sentence

        if current_node not in graph.keys():
            graph[current_node] = []

        for cap_idx, cap in enumerate(captions):
            parent_node = ' '.join(cap.split()[0:-1])
            from_prob = parents_data[cap_idx:cap_idx+1]['from_prob'].values[0]
            edges[parent_node + ':' + current_node] = from_prob

            graph[parent_node] = [current_node]

        queue += [str(hop+1)] + list(captions)
    
    return graph, edges

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
    
    


    def check_connection(self, sentence):
        #clean sentence from any <t> </t> tokens
        sentence = sentence.replace('<t>','').replace('</t>','')
        ngrams_dfs = self.ngram_dfs
        sentence_tokens = sentence.split()
        
        n = len(sentence_tokens)

        #apply filtering process for extracting this exact sentence from the corpus
        if n == 1:
            ngrams = ngrams_dfs[1][ngrams_dfs[1][1] == sentence_tokens[0]]
        elif n == 2:
            ngrams = ngrams_dfs[2]
            ngrams = ngrams[ngrams[1] == sentence_tokens[0]]
            ngrams = ngrams[ngrams[2] == sentence_tokens[1]]
        elif n == 3:
            ngrams = ngrams_dfs[3]
            ngrams = ngrams[ngrams[1] == sentence_tokens[0]]
            ngrams = ngrams[ngrams[2] == sentence_tokens[1]]
            ngrams = ngrams[ngrams[3] == sentence_tokens[2]]
        elif n == 4:
            ngrams = ngrams_dfs[4]
            ngrams = ngrams[ngrams[1] == sentence_tokens[0]]
            ngrams = ngrams[ngrams[2] == sentence_tokens[1]]
            ngrams = ngrams[ngrams[3] == sentence_tokens[2]]
            ngrams = ngrams[ngrams[4] == sentence_tokens[3]]
        elif n == 5:
            ngrams = ngrams_dfs[5]
            ngrams = ngrams[ngrams[1] == sentence_tokens[0]]
            ngrams = ngrams[ngrams[2] == sentence_tokens[1]]
            ngrams = ngrams[ngrams[3] == sentence_tokens[2]]
            ngrams = ngrams[ngrams[4] == sentence_tokens[3]]
            ngrams = ngrams[ngrams[5] == sentence_tokens[4]]
        
        if len(ngrams) > 0:
            count = ngrams['count'].values[0] #get the count of the sentence in the corpus if the 
                                            #sentence is valid and exists in the corpus
        else:
            count = 0

        return count 
    


    def top_down_traversal(self, graph, keywords, e_f=2):
        print('Top down traversal for graph (size:' + str(len(graph.keys())) + ') keywords:' + str(keywords))

        for first_node in graph.keys():
            for second_node in graph.keys():
                # check if they are not the same node nor the second sentence is a start sentence
                if first_node != second_node and second_node.split()[0] != '<t>':
                    sentence = first_node + ' ' + second_node
                    connections = self.check_connection(sentence)

                    if connections >= e_f and second_node not in graph[first_node]:
                        graph[first_node].append(second_node)
        return graph
    
    def path_init(self, keywords):
        path = []
        for keyword in keywords:
            path.append([keyword])
            
        return path
    
    def all_start_tokens(self,ngrams):
        for ng in ngrams:
            if ng != '<t>':
                return False
        return True
    
    def get_gram(self, gram):
        n = len(gram)
        ngrams = self.ngrams_dfs[n]
        history = None
        for i, f in enumerate(gram):        
            ngrams = ngrams[ngrams[i+1] == f]
            
            if i+1 == len(gram) - 1:
                history = ngrams
        
        if history is not None:
            ngrams['count_history'] = history['count'].sum()
            ngrams['conditional_prob'] = ngrams['count']/history['count'].sum()
        
        return ngrams
    
    def get_conditional_prob(self, padded_path, nlogP):
        total = 0
        
        #ask amr about this the dude pads again what to do?

        ##extract every ngram from the path formed till now
        for i,_ in enumerate(padded_path[0:len(padded_path)-nlogP+1]):
            curr = padded_path[i:i+nlogP]
            conditional_prob = self.get_gram(curr)['conditional_prob'].values
            log_prob = np.log(conditional_prob)
            
            if len(log_prob) > 0:
                total += log_prob[0]
            elif self.all_start_tokens(curr):
                total += 1
            else:
                total += -np.inf
        
        return total
            
    def get_extra_nouns(caption_tokens, keywords):
        tagged_words = pos_tag(caption_tokens)
        nouns = [word for word, pos in tagged_words if pos.startswith('N')]
        extra_nouns = [noun for noun in nouns if noun not in keywords]
        return extra_nouns
    
    def get_log_prob(self, padded_path_tokens, nlogP, keywords, optimser):
        #get total conditional probabilities for current path
        conditional_prob = self.get_conditional_prob(padded_path_tokens, nlogP) #first part 
                                    #of the cost function log(ngram)
        
        #calulcate the cost function
        if optimser > 1:
            length = 0
            
            #get extra nouns from padded path tokens using pos tagging
            N = self.get_extra_nouns(padded_path_tokens, keywords)
            
            #get how many keywords actually used in the path
            H = len([word for word in padded_path_tokens if word in keywords])
            
            if H == 0:
                H = 1e-3
                
            L = len(padded_path_tokens)
            
            if optimser == 2: #logP_H
                total /= float(H)
            elif optimser == 3: #logP_HL
                total /= float(H*L)
            elif optimser == 4: #logP_HL_N
                total = (total * (1 + len(N)))/float(H*L)
            
            if L == 0:
                total = -np.inf
    
        return total
        
            
    
    def rank(self, paths, top_n, nlogP, keywords, optimiser):
        
        #get length of paths to compare current paths
        top_n = len(paths) if len(paths) < top_n else top_n
        
        top_captions = ['']*top_n
        tops_log_probs = np.ones(top_n)*-np.inf
        min_inf = 0
        
        for path_idx, path in enumerate(paths):
            if len(path) < nlogP:
                padded_path = ['<t>']*(nlogP - len(path)) + path
            
            #get conditional probabilty of the path now 
            log_prob = self.get_log_prob(padded_path, nlogP, keywords, optimiser)
            
            if log_prob == -np.inf:
                min_inf += 1
                
            min_highest_index = np.argmin(tops_log_probs)

            if log_prob > tops_log_probs[min_highest_index]:
                tops_log_probs[min_highest_index] = log_prob
                top_captions[min_highest_index] = path
        
        order = np.argsort(tops_log_probs*-1)
        top_captions = np.array(top_captions, dtype=list)[order].tolist()
        tops_log_probs = tops_log_probs[order].tolist()

        if '' in top_captions:
            empty_caps_idx = [i for i, j in enumerate(top_captions) if j != '']
            top_captions = np.array(top_captions, dtype=list)[empty_caps_idx].tolist()
            tops_log_probs = np.array(tops_log_probs, dtype=list)[empty_caps_idx].tolist()
        
        return top_captions, tops_log_probs

    def set_global_paths(self, global_paths, global_log_probs, top_log_probs, top_paths, top_n):
            if global_paths == []:
                global_paths = top_paths
                global_log_probs = top_log_probs

            else:
                for i, p in enumerate(top_paths):
                    if p not in global_paths:
                        global_paths += [p]
                        global_log_probs += [top_log_probs[i]]

                # global_paths += top_paths
                # global_log_probs += top_log_probs

                global_log_probs_ = np.array(global_log_probs)
                global_paths_ = np.array(global_paths, dtype=list)

                max_idx = np.argsort(global_log_probs_*-1)
                global_log_probs = global_log_probs_[max_idx].tolist()[0:top_n]
                global_paths = global_paths_[max_idx].tolist()[0:top_n]

            return global_paths, global_log_probs
    
    def keywords_in_path(self, path, keywords):
        found = 0
        for ci in keywords:
            if ci in path:
                found += 1
        return found
    
    def remove_path(self, paths, path):
        filtered_paths = []
        for p in paths:
            if p != path:
                filtered_paths += [p]
        return filtered_paths
    
    def traverse(self, graph, max_iters, keywords, nlogP, optimiser):
        captions_generated = []
        paths = []
        
        global_best_path = []
        global_log_prob = []
        
        iteration = 0
        top_n = 5
        
        print(' - Traversing graph with nlogP:' + str(nlogP) + ' max iterations:' + str(max_iters) + ' top_phrases:' + str(top_n) + ' keywords:' + str(keywords))
        
        #initialise the paths
        paths = self.path_init(keywords)
        
        while (len(paths) > 0) and iteration < max_iters:
            # Q ← rank(Q,fni, op) . Top n ranked paths found in Q based on f n
            top_captions, top_log_probs = self.rank(paths, top_n, nlogP, keywords, optimiser) #get top n captions from current paths
            
            #we need to add/update the top captions to the best globally found till now
            global_best_path, global_log_prob = self.set_global_paths(global_best_path, global_log_prob, top_log_probs, top_captions, top_n)
    
            if len(top_captions) > 0:
                paths = top_captions
            else:
                break
            
            q = paths[0]
            t = q[-1]
            children = graph[t]
            
            children_in_q = self.keywords_in_path(q, children)
            
            if children_in_q == len(children):
                paths = self.remove_path(paths, q)
            else:
                for child in children:
                    if child not in q:
                        if self.keywords_in_path(q, keywords) < len(keywords):
                            paths += [q + [child]]
                        else:
                            caption = ' '.join(q)
                            
                            if caption not in captions_generated:
                                captions_generated += [caption]
                        
                        paths = self.remove_path(paths, q)
                        
            iteration += 1
            print( str(iteration) + '/' + str(max_iters) + ' ' + str(keywords) + ' paths:' + str(len(paths)) + ' captions:' + str(len(captions_generated)))
        
        #now we neek to get best captions from the global paths
        
            
                
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
            
            graph = self.top_down_traversal(graph, keywords)
            
            captions, log_probs, num_graph_nodes, iterations = self.traverse(graph, max_iters, keywords, nlogP, optimiser)



            

        
