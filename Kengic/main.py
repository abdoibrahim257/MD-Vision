from kengic import *
from predict import *

n = 2 #ngrams default = 3
parents = 5 #parents default = 5
hops = 2
nlogP = n #n used in loss function
optimiser = 3 #which cost fucntion to use
max_iters = 150 #max iterations default = 150
out_folder = 'output_captions'  

def main():
    
    image = 'Data\Images\CXR1383_IM-0245-1002.png' #get the image ID after CXR
    
    link = image.split('\\')[-1]
    getid = link.split('_')[0] #===> getid = 'CXR2'
    id = getid[3:]
    
    #remove the cxr getid
    image = Image.open(image)
    mlc, visual_extractor, vocab = initialize_models()
    topics = predict(image, mlc, visual_extractor, vocab)
    
    #make multi-word topics in a seperate list
    normal = [[topic] for topic in topics if topic == 'normal']    
    single = [topic for topic in topics if len(topic.split()) == 1 and topic != 'others' and topic != 'normal' and topic != 'atelectases']
    multi = [topic.split() for topic in topics if len(topic.split()) > 1]
    if len(normal) >=1:
        multi = [['no'] + topic.split() for topic in topics if len(topic.split()) > 1]
        single = ['no'] + single
        
    total = [single] +  multi
    ngram_dfs, _, references= initialize()
    global_caption = {}
    
    k = 0
    for keywords in total:
        graph,edges = create_graph(ngram_dfs[n],keywords , n, parents, hops)
        
        #graph generation top down
        graph = top_down_traversal(graph, keywords, ngram_dfs)
        
        #graph traversal 
        top_captions, top_costs, V, iterations = traverse(graph, max_iters, keywords, nlogP, optimiser, ngram_dfs)
        
        #check the size of the captions if greater than 3 then take the top 3
        if len(top_captions) > 3:
            top_captions = top_captions[:3]
            top_costs = top_costs[:3]
            
        global_caption[k] = [(np.round(top_costs[i],2),top_captions[i]) for i in range(len(top_captions))]
        k+=1
        
        bleuScores = get_metrics(top_captions, eval(references[1]))
        
        for i in range(len(top_captions)):
            print('\n', 'Caption:', top_captions[i], 'Cost:', top_costs[i])
            
    print(global_caption)

def initialize():
    input = pd.read_csv('./input.csv') #CNN output will be here
    img_id = input['img_id'].values[0]
    ngram_dfs,references = load_data(img_id)
    keywords = eval(input['keywords'].values[0])
    return ngram_dfs,keywords,references

main()
    