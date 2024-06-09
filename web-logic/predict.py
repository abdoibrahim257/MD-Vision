from PIL import Image
import torch
import joblib
from torchvision import transforms
from models import VisualExtractor, SentenceLSTM, WordLSTM , MLC
import spacy
from build_vocab import Vocabulary
import cv2
import numpy as np
def initialize_models (model_name = 'hog_pca') :
    """
    Function to initialize the models and return them.
    
    Returns:
        model : Model for captioning the images.
    """
    model = model_name
    path = f'./Trials/{model}/models'
    
    
    # Initialize the model
    mlc = MLC(vis_features_dim= 1024 , embed_dim= 512 , classes = 210 , k=10)
    mlc.load_state_dict(torch.load(f'{path}/mlc.pth'))
    mlc.eval()
    
    # Initialize the visual extractor
    visual_extractor = VisualExtractor( model_name= model,output_embed_size=1024, pretrained=True)
    if model != 'hog_pca':
        visual_extractor.load_state_dict(torch.load(f'{path}/visual_extractor.pth'))
    visual_extractor.eval()
    
    # Initialize the sentence lstm
    sentence_lstm = SentenceLSTM(vis_features_dim= 1024 , sem_features_dim=512 , hidden_dim=512 , att_dim= 256 , sent_input_dim= 1024 , word_input_dim= 512 , stop_dim= 256 , device='cpu')
    sentence_lstm.load_state_dict(torch.load(f'{path}/sentence_lstm.pth'))
    sentence_lstm.eval()
    
    # Initialize the word lstm
    word_lstm = WordLSTM(word_hidden_dim=512, word_input_dim= 512, vocab_size= 1974, num_layers= 1 )
    word_lstm.load_state_dict(torch.load(f'{path}/word_lstm.pth'))
    word_lstm.eval()
    
    # Load the vocab
    vocab = joblib.load('./Trials/vocab.pkl')
    
    return mlc, visual_extractor, sentence_lstm, word_lstm , vocab

def improve_punctuation(text, nlp):
    # Process the text with spaCy
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Capitalize the first letter of each sentence and strip extra spaces
    sentences = [sent.strip().capitalize() for sent in sentences]
    
    # Join the sentences with proper punctuation
    punctuated_text = ' '.join(sentences)
    return punctuated_text

def predict (image, mlc, visual_extractor, sentence_lstm, word_lstm , vocab) :
    """
    Function to predict the caption for the image provided.
    
    Args:
        image : Image for which caption is to be predicted.
    """
    
    # Initialize the models
    # mlc, visual_extractor, sentence_lstm, word_lstm , vocab = initialize_models()
    
    # Transform the image to tensor
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to the specified size
        transforms.CenterCrop(224),  # Perform a center crop of the specified size
        transforms.ToTensor(), #Convert the image to a PyTorch tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize the image
    
    # Define the maximum number of sentences and words
    s_max = 8
    w_max = 40
    device = torch.device('cpu')
    


    if visual_extractor.model_name == 'hog_pca':
        # Calculate the center of the image once
        center_x = 256 // 2 - 224 // 2
        center_y = 256 // 2 - 224 // 2
        # Convert from PIL to OpenCV
        image = np.array(image)
        image = cv2.resize(image, (256, 256)) # Resize the image to 256x256 pixels
        # Center crop
        image = image[center_x:center_x+224, center_y:center_y+224]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # add a batch dimension
        image = np.expand_dims(image, axis=0)
    else:
        # Transform the image
        image = transform(image)
        # Add a batch dimension
        image = image.unsqueeze(0)

    # Get the features from the visual extractor
    visual_features = visual_extractor(image)
    
    # Get the tags
    _ , semantic_features = mlc(visual_features)
    
    # Get the topics and stop probabilities from the sentence LSTM model
    topics, ps = sentence_lstm(visual_features, semantic_features , s_max, device)
    
    # Initialize a tensor to store the predicted words
    pred_words = torch.zeros((1, s_max, w_max))
    
    # start tokens
    start_tokens = torch.zeros((1, 1)).to(device)
    start_tokens[0, 0] = vocab.word2idx['<start>']
    
    # Initialize a list to store the predicted caption
    pred_caption = []
    
    for j in range(s_max):
        # Generate word outputs using the word LSTM model
        word_outputs = word_lstm.forward_test(topics[:, j, :], start_tokens, w_max)
        
        pred_words[:, j, :] = word_outputs
        
    for k in range(s_max):
        if ps [0 , k , 1] > 0.5:
            # Convert the predicted words to a list
            words_x = pred_words[0, k, :].tolist()
            # Convert the word IDs to words, join them into a sentence, and append to the predicted captions
            p = " ".join([vocab.id2word[w] for w in words_x if w not in {vocab.word2idx['<pad>'], vocab.word2idx['<start>'], vocab.word2idx['<end>']}]) + "."
            # remove < num > from the sentence
            p = p.replace('<', '')
            p = p.replace('>', '')
            p = p.replace('num', '')
            # remove extra spaces
            p = ' '.join(p.split())
            pred_caption.append(p)

    pred_caption = ' '.join(pred_caption)
    nlp = spacy.load('en_core_web_sm')
    pred_caption = improve_punctuation(pred_caption, nlp)
    return pred_caption


def kengic_predict(image , mlc , visual_extractor) :
    """
    Function to predict the caption for the image provided.
    
    Args:
        image : Image for which caption is to be predicted.
    """
    
    # Initialize the models
    # mlc, visual_extractor, sentence_lstm, word_lstm , vocab = initialize_models()
    
    # Transform the image to tensor
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to the specified size
        transforms.CenterCrop(224),  # Perform a center crop of the specified size
        transforms.ToTensor(), #Convert the image to a PyTorch tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize the image
    
    # Define the maximum number of sentences and words
    s_max = 8
    w_max = 40
    device = torch.device('cpu')
    


    if visual_extractor.model_name == 'hog_pca':
        # Calculate the center of the image once
        center_x = 256 // 2 - 224 // 2
        center_y = 256 // 2 - 224 // 2
        # Convert from PIL to OpenCV
        image = np.array(image)
        image = cv2.resize(image, (256, 256)) # Resize the image to 256x256 pixels
        # Center crop
        image = image[center_x:center_x+224, center_y:center_y+224]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # add a batch dimension
        image = np.expand_dims(image, axis=0)
    else:
        # Transform the image
        image = transform(image)
        # Add a batch dimension
        image = image.unsqueeze(0)

    # Get the features from the visual extractor
    visual_features = visual_extractor(image)
    
    # Get the tags
    top_K_classes, _ = mlc(visual_features)
    # tags = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', 'foreign body', 'dish', 'aorta, thoracic', 'atherosclerosis', 'histoplasmosis', 'hypoventilation', 'catheterization, central venous', 'pleural effusions', 'pleural effusion', 'callus', 'sternotomy', 'lymph nodes', 'tortuous aorta', 'stent', 'interstitial pulmonary edema', 'cholecystectomies', 'neoplasm', 'central venous catheter', 'pneumothorax', 'metastatic disease', 'vena cava, superior', 'cholecystectomy', 'scoliosis', 'subcutaneous emphysema', 'thoracolumbar scoliosis', 'spinal osteophytosis', 'pulmonary fibroses', 'rib fractures', 'sarcoidosis', 'eventration', 'fibrosis', 'spine', 'obstructive lung disease', 'pneumonitis', 'osteopenia', 'air trapping', 'demineralization', 'mass lesion', 'pulmonary hypertension', 'pleural diseases', 'pleural thickening', 'calcifications of the aorta', 'calcinosis', 'cystic fibrosis', 'empyema', 'catheter', 'lymph', 'pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease', 'chronic obstructive pulmonary disease', 'rib', 'clip', 'aortic ectasia', 'shoulder', 'scarring', 'scleroses', 'adenopathy', 'emphysemas', 'pneumonectomy', 'infection', 'aspiration', 'bilateral pleural effusion',                                                                                                                                                  'bulla', 'lumbar vertebrae', 'lung neoplasms', 'lymphadenopathy', 'hyperexpansion', 'ectasia', 'bronchiectasis', 'nodule', 'pneumonia', 'right-sided pleural effusion', 'osteoarthritis', 'thoracic spondylosis', 'picc', 'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters', 'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary', 'pneumoperitoneum', 'scar', 'atheroscleroses', 'aortic calcifications', 'volume overload', 'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', 'copd', 'kyphoses', 'spinal fractures', 'fracture', 'clavicle', 'focal atelectasis', 'collapse', 'thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema', 'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', 'humerus', 'heart failure', 'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases, interstitial', 'pulmonary disease, chronic obstructive', 'opacity', 'deformity', 'chronic disease', 'pleura', 'aorta', 'tuberculoses', 'hiatal hernia', 'scolioses', 'pleural fluid', 'malignancy', 'kyphosis', 'bronchiectases', 'congestion', 'discoid atelectasis',                                                              'nipple', 'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes', 'pulmonary edema', 'vascular calcification', 'sclerotic', 'central venous catheters', 'catheterization', 'hydropneumothorax', 'aortic valve', 'hyperinflation', 'prostheses', 'pacemaker, artificial', 'bypass grafts', 'pulmonary fibrosis', 'multiple myeloma', 'postoperative period',                                                            'cabg', 'right lower lobe pneumonia', 'granuloma', 'degenerative change, spine', 'atelectasis', 'inflammation', 'effusion', 'cicatrix', 'tracheostomy', 'aortic diseases', 'sarcoidoses', 'granulomas', 'interstitial lung disease', 'infiltrates', 'displaced fractures', 'chronic lung disease', 'picc line', 'intubation, gastrointestinal', 'lung diseases', 'multiple pulmonary nodules', 'intervertebral disc degeneration', 'pulmonary emphysema', 'spine curvature', 'fibroses', 'chronic granulomatous disease', 'degenerative disease', 'atelectases', 'ribs', 'pulmonary arterial hypertension', 'edemas', 'pectus excavatum', 'lung granuloma', 'plate-like atelectasis', 'enlarged heart', 'hilar calcification', 'heart valve prosthesis', 'tuberculosis', 'old injury', 'patchy atelectasis', 'histoplasmoses', 'exostoses', 'mastectomies', 'right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm', 'lobectomy', 'spinal fusion', 'spondylosis', 'ascending aorta', 'granulomatous infection', 'fractures, bone', 'calcified granuloma', 'degenerative joint disease', 'intubation, intratracheal', 'others']
    tags = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', '<t> foreign body',  '<t> dish',  'aorta, thoracic', '<t> atherosclerosis',  '<t> histoplasmosis', '<t> hypoventilation', '<t> catheterization, central venous',  '<t> pleural effusions',  'pleural effusion', '<t> callus',  'sternotomy',  'lymph nodes',  'tortuous aorta', '<t> stent',  'interstitial pulmonary edema', '<t> cholecystectomies',  '<t> neoplasm',  '<t> central venous catheter',  'pneumothorax',  'metastatic disease',  '<t> vena cava, superior',  '<t> cholecystectomy',  'scoliosis', '<t> subcutaneous emphysema',  '<t> thoracolumbar scoliosis',  '<t> spinal osteophytosis',  '<t> pulmonary fibroses',  '<t> rib fractures',  '<t> sarcoidosis',  '<t> eventration',  '<t> fibrosis',  'spine', 'obstructive lung disease', 'pneumonitis',  'osteopenia',  '<t> air trapping',  '<t> demineralization',  '<t> mass lesion',  'pulmonary hypertension',  'pleural diseases', 'pleural thickening', '<t> calcifications of the aorta',  '<t> calcinosis', 'cystic fibrosis', '<t> empyema', '<t> catheter', '<t> lymph', '<t> pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease', 'chronic obstructive pulmonary disease', 'rib', '<t> clip', 'aortic ectasia', '<t> shoulder', 'scarring', '<t> scleroses', 'adenopathy', 'emphysemas', '<t> pneumonectomy', 'infection', 'aspiration', 'bilateral pleural effusion', '<t> bulla', 'lumbar vertebrae', 'lung neoplasms', 'lymphadenopathy', 'hyperexpansion', '<t> ectasia', '<t> bronchiectasis', 'nodule', 'pneumonia', '<t> right-sided pleural effusion', '<t> osteoarthritis', 'thoracic spondylosis', '<t> picc', 'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters', 'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary', '<t> pneumoperitoneum', '<t> scar', 'atheroscleroses', 'aortic calcifications', 'volume overload', 'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', '<t> copd', '<t> kyphoses', 'spinal fractures', 'fracture', '<t> clavicle', 'focal atelectasis', '<t> collapse', '<t> thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema', 'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', '<t> humerus', 'heart failure', 'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases,interstitial', 'pulmonary disease,chronic obstructive', 'opacity', '<t> deformity', 'chronic disease', '<t> pleura', '<t> aorta', '<t> tuberculoses', '<t> hiatal hernia', 'scolioses', 'pleural fluid', '<t> malignancy', '<t> kyphosis', '<t> bronchiectases', '<t> congestion', '<t> discoid atelectasis', '<t> nipple', 'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes', 'pulmonary edema', 'vascular calcification', '<t> sclerotic', '<t> central venous catheters', '<t> catheterization', '<t> hydropneumothorax', '<t> aortic valve', 'hyperinflation', 'prostheses', '<t> pacemaker,artificial', '<t> bypass grafts', 'pulmonary fibrosis', '<t> multiple myeloma', '<t> postoperative period', '<t> cabg', '<t> right lower lobe pneumonia', 'granuloma', 'degenerative change, spine', 'atelectasis', 'inflammation', 'effusion', '<t> cicatrix', '<t> tracheostomy', 'aortic diseases', '<t> sarcoidoses', 'granulomas', 'interstitial lung disease', 'infiltrates', 'displaced fractures', 'chronic lung disease', '<t> picc line', 'intubation,gastrointestinal', 'lung diseases', 'multiple pulmonary nodules', '<t> intervertebral disc degeneration', 'pulmonary emphysema', '<t> spine curvature', '<t> fibroses', 'chronic granulomatous disease', 'degenerative disease', '<t> atelectases', 'ribs', 'pulmonary arterial hypertension', 'edemas', '<t> pectus excavatum', 'lung granuloma', '<t> plate-like atelectasis', 'enlarged heart', '<t> hilar calcification', '<t> heart valve prosthesis', 'tuberculosis', '<t> old injury', 'patchy atelectasis', '<t> histoplasmoses', '<t> exostoses', '<t> mastectomies', '<t> right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm', '<t> lobectomy', 'spinal fusion', '<t> spondylosis', '<t> ascending aorta', 'granulomatous infection', '<t> fractures, bone', 'calcified granuloma', 'degenerative joint disease', 'intubation, intratracheal', '<t> others']
    # print(top_K_classes)
    top_K_classes = top_K_classes[0, :10]
    top_K_classes = top_K_classes.tolist()
    top_K_classes = [tags[w] for w in top_K_classes]
    top_K_classes = [accept for accept in top_K_classes if '<t>' not in accept]
    
    return top_K_classes


# if __name__ == '__main__' :
    # image = 'Data/Images/CXR1_1_IM-0001-4001.png'
    # image = Image.open(image)
    # captions = predict(image)
    # print(captions)
