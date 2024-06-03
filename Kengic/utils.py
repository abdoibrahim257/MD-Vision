import os
import json
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import pickle
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from pycocoevalcap.cider.cider import Cider
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
import tempfile

def ngrams(x, n): #Algorithm 1
    ngrams = []
    for idx, ix in enumerate(x[0:len(x)-n+1]):
        gram = x[idx:idx+n]
        ngrams += [gram]
    return ngrams

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
        data = pd.read_csv('./indiana_reports_cleaned3.csv')
        references = eval(data[data['imgID'] == img_id].values[0][1])
        try:
            print('Loading ngrams_dic.pkl...')
            with open('ngrams_dic2.pkl','rb') as f:
                ngrams_dic = pickle.load(f)
        except:
            print('Failed to Load ngrams_dic.pkl...')

            ngrams_dic= create_ngrams(data) #bet return kol el tokens w ngrams_dic
            #save ngrams_dic bta3 kaza no3 ngram mn 1-9
        
            print('saving ngrams_dic.pkl...')
            with open('ngrams_dic2.pkl', 'wb') as f:
                pickle.dump(ngrams_dic, f)  
        ngrams_dfs = convert_ngram_dict_to_df(ngrams_dic) #convert ngrams_dic to dictionary of dataframes
        return ngrams_dfs, references

def get_parents(n,ngram_data,graph_node,parents_no=5): #DOESNT REMOVE THE KEYWORD FROM THE PARENTS
 
    graph_node = graph_node.split()[0] #we get the first word of the graph node to increase the probability of finding the parent
    parents_data = ngram_data[ngram_data[n] == graph_node]
    parents_data.loc[:, 'from_prob'] = parents_data['count']/float(parents_data['count'].sum()) #probability of phrase of being a parent to this graph node
    parents_data = parents_data.sort_values('from_prob', ascending=False)[0:parents_no]
    return parents_data


def check_connection(sentence,ngrams_dfs):
    #clean sentence from any <t> </t> tokens
    sentence = sentence.replace('<t>','').replace('</t>','')
    # ngrams_dfs = self.ngram_dfs
    sentence_tokens = sentence.split()
    
    n = len(sentence_tokens)
    ngrams = []
    #apply filtering process for extracting this exact sentence from the corpus
    ngrams = ngrams_dfs[n]
    
    for i in range(1, n+1):
        ngrams = ngrams[ngrams[i] == sentence_tokens[i-1]]
    
    if len(ngrams) > 0:
        count = ngrams['count'].values[0] #get the count of the sentence in the corpus if the 
                                        #sentence is valid and exists in the corpus
    else:
        count = 0

    return count 


def get_conditional_prob(path,n2,ngram_dfs):
    prob =0
    
    padded_path = path
    if len(path) < n2:
        padded_path = ['<t>']*(n2-len(path)) + path
    for i in range(len(padded_path)-n2+1):
        curr = padded_path[i:i+n2]
        
        n = len(curr)
        ngram = ngram_dfs[n]
        history = None
        for j,word in enumerate(curr):
            ngram = ngram[ngram[j+1] == word]
            if j+1 == n-1:
                history = ngram
                
        if history is not None:
            ngram['conditional_prob'] = ngram['count'] / (history['count'].sum())
        
        logProb =  np.log(ngram['conditional_prob']).values
        
        if len(logProb) > 0:
            prob += logProb[0]
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
    
    cost = get_conditional_prob(path,n2, ngram_dfs)

    
    if cost_func ==1:
        return cost
    elif cost_func ==2:
        return cost/float(M)
    elif cost_func ==3:
        return cost/float((M*len(path)))
    elif cost_func ==4:
        N = get_extra_nouns(path,keywords)
        return cost*(len(N))/float((M*len(path)))


def rank(Q,keywords,top_n,n2,Cost_func,ngram_dfs):
    top_n = len(Q) if len(Q) < top_n else top_n
    best_captions = ['']*top_n
    best_costs = np.ones(top_n)*(-1*np.inf)
    
    for i, q in enumerate(Q):
        
        splitted_path = ' '.join(q).split()
        cost = get_cost(splitted_path,keywords,n2,Cost_func,ngram_dfs)
        
        min_highest_index = np.argmin(best_costs)
        if cost > best_costs[min_highest_index]:
            best_costs[min_highest_index] = cost
            best_captions[min_highest_index] = q

    # Pair up best_captions and best_costs
    pairs = list(zip(best_captions, best_costs)) # zip bygma3 el 2 lists 3la ba3d w byrag3 iterator
    pairs.sort(key=lambda x: x[1], reverse=True)

    for i in range(len(pairs)):
        best_captions[i] = pairs[i][0]
        best_costs[i] = pairs[i][1]
    
    # best_captions, best_costs = zip(*pairs) # 3lashan nrag3 el 7aga zy ma kanet

    # Convert back to list if necessary
    best_captions = list(best_captions)
    best_costs = list(best_costs)

    temp_captions=[]
    temp_costs=[]
    for i in range(best_captions):
        if best_captions[i]!='':
            temp_captions.append(best_captions[i])
            temp_costs.append(best_costs[i])

    best_captions=temp_captions
    best_costs=temp_costs
        
    return best_captions,best_costs

def handle_global(global_captions,global_costs,top_captions,top_costs, top_n):
    
    for i in range(len(top_captions)):
        temp = ' '.join(top_captions[i])
        if temp not in global_captions:
            global_captions.append(temp)
            global_costs.append(top_costs[i])
    
    pairs = list(zip(global_captions, global_costs)) # zip bygma3 el 2 lists 3la ba3d w byrag3 iterator
    pairs.sort(key=lambda x: x[1], reverse=True)

    global_captions, global_costs = zip(*pairs) # 3lashan nrag3 el 7aga zy ma kanet (unzip the tuples)

    # Convert back to list if necessary
    global_captions = list(global_captions)[0:top_n]
    global_costs = list(global_costs)[0:top_n]
    
    return global_captions,global_costs
    

def get_bleu(topCaptions, reference):
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for caption in topCaptions:
        bleu_score = sentence_bleu(reference, caption.split(), smoothing_function=smoothie)
        bleu_scores.append(bleu_score)
    return bleu_scores



# def get_CIDEr(topCaptions, references):
#     meteric = Cider()
#     #gts is the reference captions
#     #res is the generated captions
#     gts = {}
#     res = {}
#     gts[0] = references
#     for i in range(len(topCaptions)):
#         res[0] =[topCaptions[i].split()]
        
#         score, scores = meteric.compute_score(gts, res)

#         print('CIDEr Score:', score)