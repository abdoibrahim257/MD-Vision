from kengic import *
from predict import *

n = 2 #ngrams default = 3
parents = 5 #parents default = 5
hops = 2
nlogP = n #n used in loss function
optimiser = 3 #which cost fucntion to use
max_iters = 30 #max iterations default = 150
out_folder = 'output_captions'  
    

def main(image_path):
    
    ngram_dfs, image_refs,image_name= initialize(image_path)
    
    #remove the cxr getid
    image = Image.open(image_path)
    mlc, visual_extractor, vocab = initialize_models()
    topics = predict(image, mlc, visual_extractor, vocab)
    
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
        graph,_ = create_graph(ngram_dfs[n],keywords , n, parents, hops)
        
        #graph generation top down
        graph = top_down_traversal(graph, keywords, ngram_dfs)
        
        #graph traversal 
        top_captions, top_costs,_,_ = traverse(graph, max_iters, keywords, nlogP, optimiser, ngram_dfs)
        
        #check the size of the captions if greater than 3 then take the top 3
        if len(top_captions) > 3:
            top_captions = top_captions[:3]
            top_costs = top_costs[:3]
            
        global_caption[k] = [(np.round(top_costs[i],2),top_captions[i]) for i in range(len(top_captions))]
        k+=1
            
    # print(global_caption)
    #for each key get the best caption make it as a list of captions to run bleu scre
    best_caption_generated = []
    
    for key in global_caption.keys():
        best_caption_generated.append(global_caption[key][0][1])
    
    #get the bleu score
    bleuScores = get_bleu(best_caption_generated, image_refs)
    print('\n', 'Average Bleu Score:', np.round(np.mean(bleuScores),3))
    
    #Punctuate each sentence and Capitalize the first letter
    # print('\n', 'Generated Captions:', '\n')
    
    return punctuate_and_capitalize(best_caption_generated) 
    
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
    

def initialize(image_path):
    # input = pd.read_csv('./input.csv') #CNN output will be here
    
    # image = 'Data\Images\CXR1383_IM-0245-1002.png' #get the image ID after CXR
    link = image_path.split('\\')[-1]
    getid = link.split('_')[0] #===> getid = 'CXR2'
    img_id = eval(getid[3:])
    
    # img_id = input['img_id'].values[0]
    ngram_dfs, refs = load_data(img_id)
    # keywords = eval(input['keywords'].values[0])
    return ngram_dfs,refs,link


#MAIN

caption = main('Test_results\CXR793_IM-2330-1001.png')
print(caption)