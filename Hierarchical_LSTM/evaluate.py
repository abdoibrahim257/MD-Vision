import torch
import argparse
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.gleu_score import sentence_gleu
import json
from models import SentenceLSTM, WordLSTM , MLC
from nltk.tokenize import word_tokenize
from utils import train_val_split, save_json
from build_vocab import Vocabulary

def evaluate(args):
    
    # Get the training and validation data loaders, vocabulary, and visual extractor from the train_val_split function
    train_loader, val_loader, vocab, visual_extractor = train_val_split(args)

    # Calculate the size of the vocabulary
    vocab_size = len(vocab)

    # Initializing with the pretrained models 
    # Initialize the MLC model with the given parameters and move it to the specified device
    mlc = MLC(args.visual_features_dim, args.sem_features_dim, args.classes, args.k).to(args.device)

    # Initialize the SentenceLSTM model with the given parameters and move it to the specified device
    sentLSTM = SentenceLSTM( vis_features_dim= args.visual_features_dim ,sem_features_dim = args.sem_features_dim , hidden_dim = args.sent_hidden_dim, att_dim = args.att_dim, sent_input_dim = args.sent_input_dim, word_input_dim = args.word_input_dim, stop_dim = args.stop_dim, device= args.device).to(args.device)

    # Initialize the WordLSTM model with the given parameters and move it to the specified device
    wordLSTM = WordLSTM(args.word_input_dim, args.word_hidden_dim, vocab_size, args.num_layers).to(args.device)

    # Load the pretrained models
    # visual_extractor.load_state_dict(torch.load(args.visual_extractor_path))
    mlc.load_state_dict(torch.load(args.mlc_path))
    sentLSTM.load_state_dict(torch.load(args.sentence_lstm_path))
    wordLSTM.load_state_dict(torch.load(args.word_lstm_path))
    
    # Set the models to evaluation mode
    # This will disable features like dropout that are used during training
    visual_extractor.eval()
    mlc.eval()
    sentLSTM.eval()
    wordLSTM.eval()

    # Initialize dictionaries to store the predicted and real sentences
    pred_sentences = {}
    real_sentences = {}
        
    for i, (images,images_id, _ , captions, prob) in enumerate(val_loader):
        
        # Move the images, captions, and prob to the device (CPU or GPU)
        images = images.to(args.device)
        captions = captions.to(args.device)
        prob = prob.to(args.device)

        # Extract visual features from the images using the visual_extractor model
        if args.visual_model_name == 'hog_pca':
            visual_features = images
        else:
            visual_features = visual_extractor(images) # Shape (batch_size, visual_output_embed_dim)
        
        # Get the semantic features from the multi-label classifier (mlc) model
        _ , semantic_features = mlc(visual_features)
        
        # Get the topics and stop probabilities from the sentence LSTM model
        topics, ps = sentLSTM(visual_features, semantic_features , args.s_max, args.device)

        # Initialize a tensor to store the predicted words
        pred_words = torch.zeros((captions.shape[0], args.s_max, args.w_max)).to(args.device)
        
        # start tokens
        start_tokens = torch.zeros((captions.shape[0], 1)).to(args.device)
        start_tokens[:, 0] = vocab.word2idx['<start>']
        
        for j in range(args.s_max):
            # Generate word outputs using the word LSTM model
            word_outputs = wordLSTM.forward_test(topics[:, j, :], start_tokens)
            # Store the word outputs in the pred_words tensor
            pred_words[:, j, :] = word_outputs

        
        # Iterate over each caption
        for j in range(captions.shape[0]):
            # Initialize lists to hold the predicted and target captions
            pred_caption = []
            target_caption = []

            # Iterate over each word in the caption
            for k in range(args.s_max):
                # If the probability of the second class is greater than 0.5
                if ps[j, k, 1] > 0.5:
                    # Convert the predicted words to a list
                    words_x = pred_words[j, k, :].tolist()

                    # Convert the word IDs to words, join them into a sentence, and append to the predicted captions
                    p = " ".join([vocab.id2word[w] for w in words_x if w not in {vocab.word2idx['<pad>'], vocab.word2idx['<start>'], vocab.word2idx['<end>']}]) + "."
                    # remove < num > from the sentence
                    p = p.replace('<', '')
                    p = p.replace('>', '')
                    p = p.replace('num', '')
                    # remove extra spaces
                    p = ' '.join(p.split())
                    pred_caption.append(p)

                # If the probability of the target class is 1
                if prob[j, k] == 1:
                    # Convert the target words to a list
                    words_y = captions[j, k, :].tolist()

                    # Convert the word IDs to words, join them into a sentence, and append to the target captions
                    target_caption.append(" ".join([vocab.id2word[w] for w in words_y if w not in {vocab.word2idx['<pad>'], vocab.word2idx['<start>'], vocab.word2idx['<end>']}]) + ".")
                    target_caption = [t.replace('< num >', '') for t in target_caption]

            pred_sentences[images_id[j]] = pred_caption
            real_sentences[images_id[j]] = target_caption

    assert len(pred_sentences) == len(real_sentences)
    
    # Convert the values of real_sentences and pred_sentences dictionaries to lists
    real_sentences_list = list(real_sentences.values())
    pred_sentences_list = list(pred_sentences.values())
    
    # Join the words in each sentence into a string
    real_sentences_list = [' '.join(sentence) for sentence in real_sentences_list]
    pred_sentences_list = [' '.join(sentence) for sentence in pred_sentences_list]
    
    # Initialize lists to store the scores
    bleu_scores = []
    meteor_scores = []
    gleu_scores = []
    
    smoothing = SmoothingFunction()
    bleu_weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    # Iterate over each pair of real and predicted sentences
    for real_sentence, pred_sentence in zip(real_sentences_list, pred_sentences_list):
        # Tokenize the sentences
        real_sentence = word_tokenize(real_sentence)
        pred_sentence = word_tokenize(pred_sentence)
    
        # Only calculate scores if both sentences are not empty
        if real_sentence and pred_sentence:
            # Calculate BLEU scores with different weights and append to bleu_scores
            bleu_scores.append([sentence_bleu([real_sentence], pred_sentence, weights=weights, smoothing_function=smoothing.method1) for weights in bleu_weights])
        
            # Calculate METEOR score and append to meteor_scores
            meteor_scores.append(single_meteor_score(real_sentence, pred_sentence))
        
            # Calculate GLEU score and append to gleu_scores
            gleu_scores.append(sentence_gleu([real_sentence], pred_sentence))
    
    # Calculate average scores
    avg_bleu = [sum(score[i] for score in bleu_scores) / len(bleu_scores) for i in range(4)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_gleu = sum(gleu_scores) / len(gleu_scores)

    # Save scores in a json file
    with open(args.result_path + 'scores.json', 'w') as f:
        scores = {'Average BLEU-1': avg_bleu[0], 'Average BLEU-2': avg_bleu[1], 'Average BLEU-3': avg_bleu[2], 'Average BLEU-4': avg_bleu[3], 'Average METEOR': avg_meteor, 'Average GLEU': avg_gleu}
        json.dump(scores, f, indent=4)


    results = {}
    # Save the hypotheses and references in the same json file
    for i,(images,images_id,_, captions, prob) in enumerate(val_loader):
        for images_id in images_id:
            results[images_id] = {'Predicted Sentences': pred_sentences[images_id], 'Real Sentences': real_sentences[images_id]}
    save_json(results, args.result_path)
    
    return avg_bleu, avg_meteor, avg_gleu


if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', type = bool, default = True, help = 'Train from scratch or use pretrained model')

    # Image processing parameters
    parser.add_argument('--img_size', type = int, default = 256, help = 'size to which image is to be resized')
    parser.add_argument('--crop_size', type = int, default = 224, help = 'size to which the image is to be cropped')

    # Device selection
    parser.add_argument('--device', type = str, default = 'cuda:0', help = 'device to train the model on')

    # Data paths
    parser.add_argument('--images_path', type = str, default = 'Data/images_filenames.pkl', help = 'path to the images pickle file')
    parser.add_argument('--captions_path', type = str, default = 'Data/captions.pkl', help = 'path to the captions pickle file')
    parser.add_argument('--vocab_path', type = str, default = 'Data/vocab.pkl', help = 'path to the vocabulary object')
    parser.add_argument('--tags_path', type = str, default = 'Data/tags.pkl', help = 'path to the tags object')

    # Data Loader Parameters
    parser.add_argument('--shuffle', type = bool, default = True, help = 'shuffle the data')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of workers for the dataloader')
    parser.add_argument('--s_max', type = int, default = 8, help = 'maximum number of sentences in a report')
    parser.add_argument('--w_max', type = int, default = 40, help = 'maximum number of words in a sentence')

    # Visual Model Parameters
    parser.add_argument('--visual_model_name', type = str, default = 'resnet152', help = 'name of the visual model')
    parser.add_argument('--visual_features_dim', type = int, default = 1024, help = 'dimension of the visual features')
    parser.add_argument('--visual_momentum', type = float, default = 0.1, help = 'momentum for the visual model')
    parser.add_argument('--visual_extractor_path', type = str, default = 'Data/models/visual_extractor.pth', help = 'path to the visual extractor model')


    # Multi-label classification parameters
    parser.add_argument('--classes', type = int, default = 210, help = 'number of classes in the dataset')
    parser.add_argument('--k', type = int, default = 10, help = 'number of tags to predict')
    parser.add_argument('--mlc_path', type = str, default = 'Data/models/mlc.pth', help = 'path to the MLC model')
    parser.add_argument('--sem_features_dim', type = int, default = 512, help = 'dimension of semantic features')

    # Sentence LSTM parameters
    parser.add_argument('--stop_dim', type = int, default = 256, help = 'intermediate state dimension of stop vector network')
    parser.add_argument('--sent_hidden_dim', type = int, default = 512, help = 'hidden state dimension of sentence LSTM')
    parser.add_argument('--sent_input_dim', type = int, default = 1024, help = 'dimension of input to sentence LSTM')
    parser.add_argument('--sentence_lstm_path', type = str, default = 'Data/models/sentence_lstm.pth', help = 'path to the sentence LSTM model')

    # Word LSTM parameters
    parser.add_argument('--word_hidden_dim', type = int, default = 512, help = 'hidden state dimension of word LSTM')
    parser.add_argument('--word_input_dim', type = int, default = 512, help = 'dimension of input to word LSTM')
    parser.add_argument('--att_dim', type = int, default = 256, help = 'dimension of intermediate state in co-attention network')
    parser.add_argument('--num_layers', type = int, default = 1, help = 'number of layers in word LSTM')
    parser.add_argument('--word_lstm_path', type = str, default = 'Data/models/word_lstm.pth', help = 'path to the word LSTM model')

    # Loss weights
    parser.add_argument('--lambda_sent', type = int, default = 1, help = 'weight for cross-entropy loss of stop vectors from sentence LSTM')    
    parser.add_argument('--lambda_word', type = int, default = 1, help = 'weight for cross-entropy loss of words predicted from word LSTM with target words')
    parser.add_argument('--lambda_tag', type = int, default = 1, help = 'weight for cross-entropy loss of tags predicted from word LSTM with target tags')

    # Training parameters
    parser.add_argument('--batch_size', type = int, default = 128, help = 'size of the batch')
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs to train the model')

    # Learning rates
    parser.add_argument('--learning_rate_mlc', type = int, default = 1e-3, help = 'learning rate for the model')
    parser.add_argument('--learning_rate_cnn', type = int, default = 1e-5, help = 'learning rate for CNN Encoder')
    parser.add_argument('--learning_rate_lstm', type = int, default = 5e-4, help = 'learning rate for LSTM Decoder')

    # Logging parameters
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_epoch', type=int , default=5, help='step size for saving trained models')

    # Save results
    parser.add_argument('--result_path', type=str, default='Data/results/', help='path to save the results')

    args = parser.parse_args('')
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    evaluate(args)