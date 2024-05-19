from utils import *

n = 3 #ngrams default = 3
parents = 5 #parents default = 5
hops = 1
nlogP = n #n used in loss function
optimiser = 2 #which cost fucntion to use
max_iters = 150 #max iterations default = 150
out_folder = 'output_captions'  

def main():
    print('AMERICA YA, HALOOO')
    ngram_dfs, keywords, references= initialize()
    
    #graph generation bottom up
    graph,edges = create_graph(ngram_dfs[n],keywords , n, parents, hops)
    
    #graph generation top down
    graph = top_down_traversal(graph, keywords, ngram_dfs)
    
    #graph traversal 
    top_captions, top_costs, V, iterations = traverse(graph, max_iters, keywords, nlogP, optimiser, ngram_dfs)
    
    for i in range(len(top_captions)):
        print('Caption:', top_captions[i], 'Cost:', top_costs[i],'\n')

def initialize():
    # configs = [n, parents, hops, nlogP, optimiser, max_iters]
    # start_end_tokens = ['<t>', '</t>']
    input = pd.read_csv('./input.csv') #CNN output will be here
    img_id = input['img_id'].values[0]
    ngram_dfs,references = load_data(img_id)
    keywords = eval(input['keywords'].values[0])

    return ngram_dfs,keywords,references
    # caption_gen = kengic(configs, out_folder)
    # return caption_gen

main()
    