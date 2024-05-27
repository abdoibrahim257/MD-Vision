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
    
    # Initialize the sentence lstm
    sentence_lstm = SentenceLSTM(vis_features_dim= 1024 , sem_features_dim=512 , hidden_dim=512 , att_dim= 256 , sent_input_dim= 1024 , word_input_dim= 512 , stop_dim= 256 , device='cpu')
    sentence_lstm.load_state_dict(torch.load('./Data/models/sentence_lstm.pth'))
    sentence_lstm.eval()
    
    # Initialize the word lstm
    word_lstm = WordLSTM(word_hidden_dim=512, word_input_dim= 512, vocab_size= 1977, num_layers= 1 )
    word_lstm.load_state_dict(torch.load('./Data/models/word_lstm.pth'))
    word_lstm.eval()
    
    # Load the vocab
    vocab = joblib.load('./Data/vocab.pkl')
    
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

def predict (image) :
    """
    Function to predict the caption for the image provided.
    
    Args:
        image : Image for which caption is to be predicted.
    """
    
    # Initialize the models
    mlc, visual_extractor, sentence_lstm, word_lstm , vocab = initialize_models()
    
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



if __name__ == '__main__' :
    image = 'Data/Images/CXR2_IM-0652-1001.png'
    image = Image.open(image)
    captions = predict(image)
    print(captions)
