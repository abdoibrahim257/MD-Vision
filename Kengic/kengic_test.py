from kengic import *
import pandas as pd

n = 2 #ngrams default = 3
parents = 5 #parents default = 5
hops = 2
nlogP = n #n used in loss function
optimiser = 3 #which cost fucntion to use
max_iters = 30 #max iterations default = 150
out_folder = 'output_captions'  


def main_kengic(keywords,unique_refs):
    data = pd.read_csv('./indiana_reports_cleaned3.csv')
    with open('ngrams_dic2.pkl','rb') as f:
        ngrams_dic = pickle.load(f)
    ngrams_dfs = convert_ngram_dict_to_df(ngrams_dic)
    graph,_ = create_graph(ngrams_dfs[n],keywords , n, parents, hops)
        #graph generation top down
    graph = top_down_traversal(graph, keywords, ngrams_dfs)
        
        #graph traversal 
    top_captions, top_costs,_,_ = traverse(graph, max_iters, keywords, nlogP, optimiser, ngrams_dfs)
        
        #check the size of the captions if greater than 3 then take the top 3
    if len(top_captions) > 3:
        top_captions = top_captions[:3]
        top_costs = top_costs[:3]
    bleu_scores = get_bleu(top_captions, unique_refs)
    #get average of the bleu scores
    avg_bleu = np.mean(bleu_scores)
    return avg_bleu