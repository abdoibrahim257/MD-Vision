from kengic import *
from predict import *

n = 2 #ngrams default = 3
parents = 5 #parents default = 5
hops = 2
nlogP = n #n used in loss function
optimiser = 3 #which cost fucntion to use
max_iters = 30 #max iterations default = 150
out_folder = 'output_captions'  
    

   
def punctuate_and_capitalize(sentences):
    punctuated_sentences = []
    for sentence in sentences:
        # Strip leading and trailing whitespace
        sentence = sentence.strip()
        # Capitalize the first letter
        sentence = sentence.capitalize()
        # Ensure the sentence ends with a period (or appropriate punctuation)
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'
        punctuated_sentences.append(sentence)
        
    # join the sentences
    punctuated_sentences = ' '.join(punctuated_sentences)
    return punctuated_sentences
    

def initialize_kengic():

    ngram_dfs,_ = load_data(-1)
    
    return ngram_dfs


def kengic_main(image, mlc, visual_extractor, ngram_dfs):
    
    # ngram_dfs, image_refs= initialize(image_path)
    # ngram_dfs, _= initialize(image_path)
    
    # mlc, visual_extractor, vocab = initialize_models()
    
    topics = kengic_predict(image, mlc, visual_extractor)
    
    #remove commas from the topics
    topics = [str(topic).replace(',', '') for topic in topics]
    
    deseases = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "nodule", "pneumonia", "pneumothorax", "edema", "emphysema", "fibrosis", "pleural", "thickening", "hernia"]
        
    #make multi-word topics in a seperate list
    normal = [[topic] for topic in topics if topic == 'normal']
    if len(normal) > 0:
        #I need to negate every single topic
        single = [topic for topic in topics if len(topic.split())==1 and topic != 'normal'] 
        single_no = []
        single_temp = []
        
        for i in range(len(single)):
            if single[i] in deseases:
                single_no.append(['no', single[i]])
            else:
                single_temp.append(single[i])
        
        total = single_no + [single_temp]
        
        multi = [topic.split() for topic in topics if len(topic.split()) > 1 and topic != 'normal']
        
        for i in range(len(multi)):
            #check if any of he words in the multi-word topic is in the deseases list
            if any(word in deseases for word in multi[i]):
                multi_no = ['no'] + multi[i]
                total.append(multi_no) 
            else:
                total.append(multi[i])
                
        #remove any list less than 1
        total = [x for x in total if len(x) > 1]
        
    else:
        single = [topic for topic in topics if len(topic.split())==1] 
        multi = [topic.split() for topic in topics if len(topic.split()) > 1]
        total = [single]+  multi
        
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
            
    best_caption_generated = []
    
    for key in global_caption.keys():
        best_caption_generated.append(global_caption[key][0][1])
    
    return punctuate_and_capitalize(best_caption_generated) 


#MAIN

# caption = main('Data\\nlm_images\\CXR2867_IM-1274-1001.png')
# print(caption)