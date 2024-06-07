from PIL import Image
import torch
import joblib
from torchvision import transforms
from models_new1 import VisualExtractor, SentenceLSTM, WordLSTM , MLC
import spacy
from build_vocab import Vocabulary

def initialize_models () :
    """
    Function to initialize the models and return them.
    
    Returns:
        model : Model for captioning the images.
    """
    
    # Initialize the model
    mlc = MLC(vis_features_dim= 1024 , embed_dim= 512 , classes = 210 , k=10)
    mlc.load_state_dict(torch.load('./Data/models/mlc.pth'))
    mlc.eval()
    
    # Initialize the visual extractor
    visual_extractor = VisualExtractor( model_name='resnet152',output_embed_size=1024, pretrained=True)
    visual_extractor.load_state_dict(torch.load('./Data/models/visual_extractor.pth'))
    visual_extractor.eval()
    
    # Load the vocab
    vocab = joblib.load('./Data/vocab.pkl')
    
    return mlc, visual_extractor, vocab

def improve_punctuation(text, nlp):
    # Process the text with spaCy
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Capitalize the first letter of each sentence and strip extra spaces
    sentences = [sent.strip().capitalize() for sent in sentences]
    
    # Join the sentences with proper punctuation
    punctuated_text = ' '.join(sentences)
    return punctuated_text


def predict (image , mlc , visual_extractor , vocab) :
    """
    Function to predict the caption for the image provided.
    
    Args:
        image : Image for which caption is to be predicted.
    """
    
    # Transform the image to tensor
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to the specified size
        transforms.CenterCrop(224),  # Perform a center crop of the specified size
        transforms.ToTensor(), #Convert the image to a PyTorch tensor
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize the image
    
    # Define the maximum number of sentences and words
    s_max = 8
    w_max = 40
    device = torch.device('cpu')
    
    # Transform the image
    image = transform(image)
    
    # Add a batch dimension
    image = image.unsqueeze(0)
    
    # Get the features from the visual extractor
    visual_features = visual_extractor(image)
    
    # Get the tags
    top_K_classes, semantic_features = mlc(visual_features)
    # tags = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', 'foreign body', 'dish', 'aorta, thoracic', 'atherosclerosis', 'histoplasmosis', 'hypoventilation', 'catheterization, central venous', 'pleural effusions', 'pleural effusion', 'callus', 'sternotomy', 'lymph nodes', 'tortuous aorta', 'stent', 'interstitial pulmonary edema', 'cholecystectomies', 'neoplasm', 'central venous catheter', 'pneumothorax', 'metastatic disease', 'vena cava, superior', 'cholecystectomy', 'scoliosis', 'subcutaneous emphysema', 'thoracolumbar scoliosis', 'spinal osteophytosis', 'pulmonary fibroses', 'rib fractures', 'sarcoidosis', 'eventration', 'fibrosis', 'spine', 'obstructive lung disease', 'pneumonitis', 'osteopenia', 'air trapping', 'demineralization', 'mass lesion', 'pulmonary hypertension', 'pleural diseases', 'pleural thickening', 'calcifications of the aorta', 'calcinosis', 'cystic fibrosis', 'empyema', 'catheter', 'lymph', 'pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease', 'chronic obstructive pulmonary disease', 'rib', 'clip', 'aortic ectasia', 'shoulder', 'scarring', 'scleroses', 'adenopathy', 'emphysemas', 'pneumonectomy', 'infection', 'aspiration', 'bilateral pleural effusion',                                                                                                                                                  'bulla', 'lumbar vertebrae', 'lung neoplasms', 'lymphadenopathy', 'hyperexpansion', 'ectasia', 'bronchiectasis', 'nodule', 'pneumonia', 'right-sided pleural effusion', 'osteoarthritis', 'thoracic spondylosis', 'picc', 'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters', 'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary', 'pneumoperitoneum', 'scar', 'atheroscleroses', 'aortic calcifications', 'volume overload', 'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', 'copd', 'kyphoses', 'spinal fractures', 'fracture', 'clavicle', 'focal atelectasis', 'collapse', 'thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema', 'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', 'humerus', 'heart failure', 'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases, interstitial', 'pulmonary disease, chronic obstructive', 'opacity', 'deformity', 'chronic disease', 'pleura', 'aorta', 'tuberculoses', 'hiatal hernia', 'scolioses', 'pleural fluid', 'malignancy', 'kyphosis', 'bronchiectases', 'congestion', 'discoid atelectasis',                                                              'nipple', 'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes', 'pulmonary edema', 'vascular calcification', 'sclerotic', 'central venous catheters', 'catheterization', 'hydropneumothorax', 'aortic valve', 'hyperinflation', 'prostheses', 'pacemaker, artificial', 'bypass grafts', 'pulmonary fibrosis', 'multiple myeloma', 'postoperative period',                                                            'cabg', 'right lower lobe pneumonia', 'granuloma', 'degenerative change, spine', 'atelectasis', 'inflammation', 'effusion', 'cicatrix', 'tracheostomy', 'aortic diseases', 'sarcoidoses', 'granulomas', 'interstitial lung disease', 'infiltrates', 'displaced fractures', 'chronic lung disease', 'picc line', 'intubation, gastrointestinal', 'lung diseases', 'multiple pulmonary nodules', 'intervertebral disc degeneration', 'pulmonary emphysema', 'spine curvature', 'fibroses', 'chronic granulomatous disease', 'degenerative disease', 'atelectases', 'ribs', 'pulmonary arterial hypertension', 'edemas', 'pectus excavatum', 'lung granuloma', 'plate-like atelectasis', 'enlarged heart', 'hilar calcification', 'heart valve prosthesis', 'tuberculosis', 'old injury', 'patchy atelectasis', 'histoplasmoses', 'exostoses', 'mastectomies', 'right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm', 'lobectomy', 'spinal fusion', 'spondylosis', 'ascending aorta', 'granulomatous infection', 'fractures, bone', 'calcified granuloma', 'degenerative joint disease', 'intubation, intratracheal', 'others']
    tags = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', '<t> foreign body',  '<t> dish',  'aorta, thoracic', '<t> atherosclerosis',  '<t> histoplasmosis', '<t> hypoventilation', '<t> catheterization, central venous',  '<t> pleural effusions',  'pleural effusion', '<t> callus',  'sternotomy',  'lymph nodes',  'tortuous aorta', '<t> stent',  'interstitial pulmonary edema', '<t> cholecystectomies',  '<t> neoplasm',  '<t> central venous catheter',  'pneumothorax',  'metastatic disease',  '<t> vena cava, superior',  '<t> cholecystectomy',  'scoliosis', '<t> subcutaneous emphysema',  '<t> thoracolumbar scoliosis',  '<t> spinal osteophytosis',  '<t> pulmonary fibroses',  '<t> rib fractures',  '<t> sarcoidosis',  '<t> eventration',  '<t> fibrosis',  'spine', 'obstructive lung disease', 'pneumonitis',  'osteopenia',  '<t> air trapping',  '<t> demineralization',  '<t> mass lesion',  'pulmonary hypertension',  'pleural diseases', 'pleural thickening', '<t> calcifications of the aorta',  '<t> calcinosis', 'cystic fibrosis', '<t> empyema', '<t> catheter', '<t> lymph', '<t> pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease', 'chronic obstructive pulmonary disease', 'rib', '<t> clip', 'aortic ectasia', '<t> shoulder', 'scarring', '<t> scleroses', 'adenopathy', 'emphysemas', '<t> pneumonectomy', 'infection', 'aspiration', 'bilateral pleural effusion', '<t> bulla', 'lumbar vertebrae', 'lung neoplasms', 'lymphadenopathy', 'hyperexpansion', '<t> ectasia', '<t> bronchiectasis', 'nodule', 'pneumonia', '<t> right-sided pleural effusion', '<t> osteoarthritis', 'thoracic spondylosis', '<t> picc', 'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters', 'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary', '<t> pneumoperitoneum', '<t> scar', 'atheroscleroses', 'aortic calcifications', 'volume overload', 'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', '<t> copd', '<t> kyphoses', 'spinal fractures', 'fracture', '<t> clavicle', 'focal atelectasis', '<t> collapse', '<t> thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema', 'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', '<t> humerus', 'heart failure', 'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases,interstitial', 'pulmonary disease,chronic obstructive', 'opacity', '<t> deformity', 'chronic disease', '<t> pleura', '<t> aorta', '<t> tuberculoses', '<t> hiatal hernia', 'scolioses', 'pleural fluid', '<t> malignancy', '<t> kyphosis', '<t> bronchiectases', '<t> congestion', '<t> discoid atelectasis', '<t> nipple', 'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes', 'pulmonary edema', 'vascular calcification', '<t> sclerotic', '<t> central venous catheters', '<t> catheterization', '<t> hydropneumothorax', '<t> aortic valve', 'hyperinflation', 'prostheses', '<t> pacemaker,artificial', '<t> bypass grafts', 'pulmonary fibrosis', '<t> multiple myeloma', '<t> postoperative period', '<t> cabg', '<t> right lower lobe pneumonia', 'granuloma', 'degenerative change, spine', 'atelectasis', 'inflammation', 'effusion', '<t> cicatrix', '<t> tracheostomy', 'aortic diseases', '<t> sarcoidoses', 'granulomas', 'interstitial lung disease', 'infiltrates', 'displaced fractures', 'chronic lung disease', '<t> picc line', 'intubation,gastrointestinal', 'lung diseases', 'multiple pulmonary nodules', '<t> intervertebral disc degeneration', 'pulmonary emphysema', '<t> spine curvature', '<t> fibroses', 'chronic granulomatous disease', 'degenerative disease', '<t> atelectases', 'ribs', 'pulmonary arterial hypertension', 'edemas', '<t> pectus excavatum', 'lung granuloma', '<t> plate-like atelectasis', 'enlarged heart', '<t> hilar calcification', '<t> heart valve prosthesis', 'tuberculosis', '<t> old injury', 'patchy atelectasis', '<t> histoplasmoses', '<t> exostoses', '<t> mastectomies', '<t> right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm', '<t> lobectomy', 'spinal fusion', '<t> spondylosis', '<t> ascending aorta', 'granulomatous infection', '<t> fractures, bone', 'calcified granuloma', 'degenerative joint disease', 'intubation, intratracheal', '<t> others']

    # using the top 10 tags return the top 5 tags from the vocab   
    top_K_classes = top_K_classes[0, :10]
    top_K_classes = top_K_classes.tolist()
    top_K_classes = [tags[w] for w in top_K_classes]
    top_K_classes = [accept for accept in top_K_classes if '<t>' not in accept]
    
    return top_K_classes