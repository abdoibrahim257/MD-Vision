from utils import *

n = 3 #ngrams default = 3
parents = 5 #parents default = 5
hops = 2
nlogP = n #n used in loss function
optimiser = 4 #which cost fucntion to use
max_iters = 150 #max iterations default = 150
out_folder = 'output_captions'  

def main():
    print('AMERICA YA, HALOOO')
    ngram_dfs,keywords,references= initialize()
    graph,edges = create_graph(ngram_dfs[n],keywords , n, parents, hops)
    # caption_gen.caption_generation()
    

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
    