from kengic import *
from predict import *
import os

n = 2 #ngrams default = 3
parents = 5 #parents default = 5
hops = 2
nlogP = n #n used in loss function
optimiser = 3 #which cost fucntion to use
max_iters = 30 #max iterations default = 150
out_folder = 'output_captions'  
    

def main(image_path):
    
    ngram_dfs, image_refs= initialize(image_path)
    
    if ngram_dfs == None and image_refs == None:
        return None
    #remove the cxr getid
    image = Image.open(image_path)
    mlc, visual_extractor, _, _, _ = initialize_models('resnet152')
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
    # total = [['thoracic', 'vertebrae']]
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
        
        #loop on each top captions if length of splitted caption is less than keywords then remove it
        temp1 = []
        temp2 = []
        for i in range(len(top_captions)):
            if len(top_captions[i].split()) >= len(keywords):
                temp1.append(top_captions[i])
                temp2.append(top_costs[i])
        top_captions = temp1
        top_costs = temp2
        
        global_caption[k] = [(np.round(top_costs[l],2),top_captions[l]) for l in range(len(top_captions))]
        k+=1
            
    # print(global_caption)
    #for each key get the best caption make it as a list of captions to run bleu scre
    best_caption_generated = []
    
    for key in global_caption.keys():
        if len(global_caption[key]) > 1:
            best_caption_generated.append(global_caption[key][0][1])
    
    bleu_1 = 0
    bleu_2 = 0
    bleu_3 = 0
    bleu_4 = 0
    meteor_score = 0
         
    if (len(best_caption_generated) > 0):
    #get the bleu score
        bleu_1,bleu_2,bleu_3,bleu_4 = get_bleu_all(best_caption_generated, image_refs)
        print('\n', 'Average Bleu Score:', np.round(bleu_1,3), np.round(bleu_2,3),np.round(bleu_3,3),np.round(bleu_4,3))
        
        meteor_score = get_meteor_all(best_caption_generated, image_refs)
        print('\n', 'Meteor Score:', np.round(meteor_score,3))
        
    return np.round(bleu_1,3), np.round(bleu_2,3),np.round(bleu_3,3),np.round(bleu_4,3), np.round(meteor_score,3), punctuate_and_capitalize(best_caption_generated) 
    
def punctuate_and_capitalize(sentences):
    punctuated_sentences = []
    if(len(sentences) > 0):
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
    # image = 'Data\Images\CXR1383_IM-0245-1002.png' #get the image ID after CXR
    link = image_path.split('\\')[-1]
    getid = link.split('_')[0] #===> getid = 'CXR2'
    img_id = eval(getid[3:])
    
    print('Image ID:', img_id)
    ngram_dfs, refs = load_data(img_id)
    
    return ngram_dfs,refs


#TESTING
#loop on images in Test_results run main using the images path and write in file each image and its bleu
# imgDir = 'Test_results\\test_imgs3'
# results = []

# # saving to excel file
# for img in os.listdir(imgDir):
#     # with open('Test_results/BLAA.txt', 'a') as f:
#     #     f.write("Image: " + img + '\n')
#     result = main(os.path.join(imgDir,img))
#     if result:
#         bleu_1, bleu_2, bleu_3, bleu_4, meteor, caption = result
#     results.append([img, caption, bleu_1, bleu_2, bleu_3, bleu_4, meteor])

# # Create a DataFrame from the results
# df = pd.DataFrame(results, columns=['Image', 'Caption', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'Meteor'])
# # Write the DataFrame to an Excel file
# df.to_excel('Test_results/caption_genn5.xlsx', index=False)
# print('Done!')

  
    
#MAIN
_,_,_,_,_,caption= main('Images\CXR3110_IM-1460-1001.png')
print(caption)