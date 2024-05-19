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

def convert_ngram_dict_to_df(ngrams):
    ngram_dfs={}
    for j in range(1,6):
        columns = [i for i in range(1,j+1)]
        ngrams_df = pd.DataFrame(ngrams[j], columns=columns)
        ngrams_df['count'] = 0
        ngrams_df = ngrams_df.groupby(columns).count()
        ngrams_df = ngrams_df.sort_values(['count'], ascending=False).reset_index()
        ngrams_df['probability'] = ngrams_df['count']/ngrams_df['count'].sum()
        ngram_dfs[j] = ngrams_df
    return ngram_dfs

def ngrams(x, n): #Algorithm 1
    ngrams = []
    for idx, ix in enumerate(x[0:len(x)-n+1]):
        gram = x[idx:idx+n]
        ngrams += [gram]
    return ngrams

def create_ngrams(data):
    unigram_list = []
    bigram_list = []
    trigram_list = []
    fourgram_list = []
    fivegram_list = []
    
    for idx,row in data.iterrows():
        row = eval(row["captions"]) # eval converts this "[alo]" to ["alo"] 
        for caption in row:
            unigram_list += ngrams(['<t>'] + caption + ['</t>'], 1)
            bigram_list += ngrams(['<t>'] + caption + ['</t>'], 2)
            trigram_list += ngrams(['<t>', '<t>'] + caption + ['</t>'], 3)
            fourgram_list += ngrams(['<t>', '<t>', '<t>'] + caption + ['</t>'], 4)
            fivegram_list += ngrams(['<t>', '<t>', '<t>', '<t>'] + caption + ['</t>'], 5)
    
    ngrams_dict = {1:unigram_list,2:bigram_list, 3:trigram_list, 4:fourgram_list, 5:fivegram_list}
    return ngrams_dict

def load_data(img_id):
        data = pd.read_csv('./indiana_reports_cleaned2.csv')
        references = data[data['imgID'] == img_id].values[0]
        try:
            print('Loading ngrams_dic.pkl...')
            with open('ngrams_dic.pkl','rb') as f:
                ngrams_dic = pickle.load(f)
        except:
            print('Failed to Load ngrams_dic.pkl...')

            ngrams_dic= create_ngrams(data) #bet return kol el tokens w ngrams_dic
            #save ngrams_dic bta3 kaza no3 ngram mn 1-9
        
            print('saving ngrams_dic.pkl...')
            with open('ngrams_dic.pkl', 'wb') as f:
                pickle.dump(ngrams_dic, f)  
        ngrams_dfs = convert_ngram_dict_to_df(ngrams_dic) #convert ngrams_dic to dictionary of dataframes
        return ngrams_dfs,references

def get_parents(n,ngram_data,graph_node,parents_no=5): #DOESNT REMOVE THE KEYWORD FROM THE PARENTS
 
    graph_node = graph_node.split()[0] #we get the first word of the graph node to increase the probability of finding the parent
    parents_data = ngram_data[ngram_data[n] == graph_node]
    parents_data.loc[:, 'from_prob'] = parents_data['count']/float(parents_data['count'].sum()) #probability of phrase of being a parent to this graph node
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
        
        # This is done to get parents of parents in case of hops > 0 (gn0 in paper)
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

def check_connection(sentence,ngrams_dfs):
    #clean sentence from any <t> </t> tokens
    sentence = sentence.replace('<t>','').replace('</t>','')
    # ngrams_dfs = self.ngram_dfs
    sentence_tokens = sentence.split()
    
    n = len(sentence_tokens)
    ngrams = []
    #apply filtering process for extracting this exact sentence from the corpus
    if n == 1:
        ngrams = ngrams_dfs[1]
        ngrams = ngrams[ngrams[1] == sentence_tokens[0]]
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

def top_down_traversal(graph, keywords, ngram_dfs,e_f=2):
    print('Top down traversal for graph (size:' + str(len(graph.keys())) + ') keywords:' + str(keywords))
    for first_node in graph.keys():
        for second_node in graph.keys():
            # check if they are not the same node nor the second sentence is a start sentence
            if first_node != second_node and second_node.split()[0] != '<t>':
                sentence = first_node + ' ' + second_node
                connections = check_connection(sentence, ngram_dfs)

                if connections >= e_f and second_node not in graph[first_node]:
                    graph[first_node] += [second_node]
    return graph

def all_start_tokens(ngrams):
    for ng in ngrams:
        if ng != '<t>':
            return False
    
    return True

def calc_Conditional(padded_path, ngram_dfs):
    n = len(padded_path)
    ngram = ngram_dfs[n]
    history = None
    for j,word in enumerate(padded_path):
        ngram = ngram[ngram[j+1] == word]
        if j+1 == n-1:
            history = ngram
            
    if history is not None:
        ngram.loc[:, 'conditional_prob'] = ngram['count'] / (history['count'].sum())
    return ngram

def get_conditional_prob(path,n2,ngram_dfs):
    prob =0
    ngram = ngram_dfs[n2]
    
    padded_path = path
    if len(path) < n2:
        padded_path = ['<t>']*(n2-len(path)) + path
    for i in range(len(padded_path)-n2+1):
        curr = padded_path[i:i+n2]
        
        logProb =  np.log(calc_Conditional(curr, ngram_dfs)['conditional_prob']).values
        
        if len(logProb) > 0:
            prob += logProb[0]
        elif (all_start_tokens(curr)):
            prob += 1
        else:
            prob += -1*np.inf
  
    return prob

def get_extra_nouns(caption_tokens, keywords):
    tagged_words = pos_tag(caption_tokens)
    nouns = [word for word, pos in tagged_words if pos.startswith('N')]
    extra_nouns = [noun for noun in nouns if noun not in keywords]
    return extra_nouns

def get_cost(path,keywords,n2,cost_func,ngram_dfs):
    cost =0
    M = len([word for word in path if word in keywords])
    
    # splitted_path = ' '.join(path).split()
    cost = get_conditional_prob(path,n2, ngram_dfs)

    
    if cost_func ==1:
        return cost
    elif cost_func ==2:
        return cost/float(M)
    elif cost_func ==3:
        return cost/float((M*len(path)))
    elif cost_func ==4:
        N = get_extra_nouns(splitted_path,keywords)
        return cost*(len(N))/float((M*len(path)))


def rank(Q,keywords,top_n,n2,Cost_func,ngram_dfs):
    top_n = len(Q) if len(Q) < top_n else top_n
    best_captions = ['']*top_n
    best_costs = np.ones(top_n)*(-1*np.inf)
    
    for i, q in enumerate(Q):
        padded_path = q
        if len(q) < n2:
            padded_path = ['<t>']*(n2-len(q)) + q
        
        splitted_path = ' '.join(padded_path).split()
        cost = get_cost(splitted_path,keywords,n2,Cost_func,ngram_dfs)
        
        min_highest_index = np.argmin(best_costs)
        if cost > best_costs[min_highest_index]:
            best_costs[min_highest_index] = cost
            best_captions[min_highest_index] = q

    
    order = np.argsort(best_costs*-1)
    best_captions = np.array(best_captions, dtype=list)[order].tolist()
    best_costs = best_costs[order].tolist()

    if '' in best_captions:
        empty_caps_idx = [i for i, j in enumerate(best_captions) if j != '']
        best_captions = np.array(best_captions, dtype=list)[empty_caps_idx].tolist()
        best_costs = np.array(best_costs, dtype=list)[empty_caps_idx].tolist()
        
    
    return best_captions,best_costs

    

def traverse(graph, max_iters, keywords, n2, optimiser,ngram_dfs):
    S = set()
    Q = [[keyword] for keyword in keywords]
    qi=0
    top_n = 5
    while len(Q) > 0 and qi < max_iters:
        topCaptions,topCosts = rank(Q,keywords,top_n,n2,optimiser,ngram_dfs)
        
        if len(topCaptions) > 0:
                Q = topCaptions
        else:
            break
        
        q = topCaptions[0]
        t = q[-1]
        children = graph[t]
        children_in_q = len([child for child in children if child in q])
        if children_in_q != len(children):
            for child in children:
                if child not in q:
                    if len([word for word in q if word in keywords]) < len(keywords):
                        Q.append(q + [child])
                    else:
                        caption = ' '.join(q)
                        if caption not in S:
                            S.add(caption)
        Q.remove(q)
        qi+=1
    
    #rank the captions in S
    print("Ranking the captions in S")
    S_splitted = [caption.split() for caption in S]
    top_S, top_S_cost = rank(S_splitted,keywords,top_n,n2,optimiser,ngram_dfs)
    
    tops_S = [' '.join(caption) for caption in top_S]

    return tops_S,top_S_cost, len(graph.keys()), qi

# Now first_key and first_value hold the key and value of the first item in the dictionary



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

        ##extract every ngram from the path formed till now (if padded path is bigger than nlogP then we need to extract every nlogP ngram from the path) 
        #Example of above case: path = ["dog","riding","a","skateboard"] nlogP = 3 ==> ngrams = ["dog","riding","a"] , ["riding","a","skateboard"]
         
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
        total = self.get_conditional_prob(padded_path_tokens, nlogP) #first part 
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
                total = (total * (len(N)))/float(H*L)
            
            if L == 0:
                total = -np.inf
    
        return total
        
            
    
    def rank(self, paths, top_n, nlogP, keywords, optimiser):
        
        #get length of paths to compare current paths
        top_n = len(paths) if len(paths) < top_n else top_n
        
        top_captions = ['']*top_n
        tops_log_probs = np.ones(top_n)*-np.inf
        min_inf = 0
        
        for path_idx, path in enumerate(paths):#["a","b","c"]
            if len(path) < nlogP:
                padded_path = ['<t>']*(nlogP - len(path)) + path
            else:
                padded_path=path
            #get cost of padded path 
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
            
            q = paths[0]  # [["a","b","c"],["a","b","d"],["a","b","e"]] / 
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



            

        
