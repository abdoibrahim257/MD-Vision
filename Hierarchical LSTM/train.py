import torch
import argparse
from torch import nn
import numpy as np
from tqdm import tqdm
from models import VisualExtractor, SentenceLSTM, WordLSTM , MLC
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts,OneCycleLR
from build_vocab import Vocabulary
from utils import train_val_split, save_model, generate_loss_plot
from evaluate import evaluate


def script(args):
    
        # Get the train_loader, val_loader, vocab and visual_extractor
    # train_val_split is a function that splits the dataset into training and validation sets
    train_loader, val_loader, vocab , visual_extractor = train_val_split(args)

    # Get the size of the vocabulary
    vocab_size = len(vocab) 

    print ('Start training the model')
    # The Visual Extractor model is initialized with the visual model name, visual features dimension, and momentum
    visual_extractor = visual_extractor
    
    # The MLC (Multi-Label Classification) model is initialized with the visual features dimension, semantic features dimension, number of classes, and a boolean indicating whether to use dropout
    mlc = MLC(args.visual_features_dim, args.sem_features_dim, args.classes , args.k).to(args.device)

    # The Sentence LSTM model is initialized with the visual features dimension, semantic features dimension, hidden dimension, attention dimension, sentence input dimension, word input dimension, internal stop dimension, and the device to run on
    sentLSTM = SentenceLSTM( vis_features_dim= args.visual_features_dim ,sem_features_dim = args.sem_features_dim , hidden_dim = args.sent_hidden_dim, att_dim = args.att_dim, sent_input_dim = args.sent_input_dim, word_input_dim = args.word_input_dim, stop_dim = args.stop_dim, device= args.device).to(args.device)

    # The Word LSTM model is initialized with the word input dimension, word hidden dimension, vocabulary size, and number of layers
    wordLSTM = WordLSTM(args.word_input_dim, args.word_hidden_dim, vocab_size, args.num_layers).to(args.device)

    # Define the loss functions for the tag, stop, and words
    criterion_tag = nn.BCELoss().to(args.device)
    criterion_stop = nn.CrossEntropyLoss().to(args.device)
    criterion_words = nn.CrossEntropyLoss().to(args.device)

    # Get the parameters of the CNN, LSTM, and MLC models
    params_cnn = list(visual_extractor.parameters())
    params_lstm = list(sentLSTM.parameters()) + list(wordLSTM.parameters())
    params_mlc = list(mlc.parameters())
        
    # Define the optimizers for the CNN, LSTM, and MLC models
    if (args.visual_model_name != 'hog_pca'):
        optim_cnn = torch.optim.Adam(params = params_cnn, lr = args.learning_rate_cnn)
    optim_lstm = torch.optim.Adam(params = params_lstm, lr = args.learning_rate_lstm)
    optim_mlc = torch.optim.Adam(params = params_mlc, lr = args.learning_rate_mlc)

    # Define the learning rate schedulers for the CNN, LSTM, and MLC models
    # The schedulers use the CosineAnnealingWarmRestarts strategy, which periodically resets the learning rate to a high value and decreases it according to a cosine schedule
    if (args.visual_model_name != 'hog_pca'):
        scheduler_cnn = CosineAnnealingWarmRestarts(optim_cnn, T_0 = 10, T_mult = 1, eta_min = 1e-6)
    scheduler_lstm = CosineAnnealingWarmRestarts(optim_lstm, T_0 = 10, T_mult = 1, eta_min = 5e-5)
    scheduler_mlc = CosineAnnealingWarmRestarts(optim_mlc, T_0 = 10, T_mult = 1, eta_min = 1e-4)

    # Get the total number of steps in the training set
    total_step = len(train_loader)
    
    if (args.visual_model_name != 'hog_pca'):
        assert next(visual_extractor.parameters()).is_cuda, "Visual Extractor model is not on CUDA"
    assert next(mlc.parameters()).is_cuda, "MLC model is not on CUDA"
    assert next(sentLSTM.parameters()).is_cuda, "Sentence LSTM is not on CUDA"
    assert next(wordLSTM.parameters()).is_cuda, "Word LSTM is not on CUDA"
    
    print ('The models are on CUDA')
    
    initial_loss = np.inf
    best_score = 0
    
    # # # Load the pretrained models
    # visual_extractor.load_state_dict(torch.load("./Data/models/visual_extractor.pth"))
    # mlc.load_state_dict(torch.load("./Data/models/mlc.pth"))
    # sentLSTM.load_state_dict(torch.load("./Data/models/sentence_lstm.pth"))
    # wordLSTM.load_state_dict(torch.load("./Data/models/word_lstm.pth"))
    
    # bleu , meteor , gleu = evaluate(args,visual_extractor, mlc, sentLSTM, wordLSTM)
    # best_score = 0.15 * (bleu[0] + bleu[1] + bleu[2] + bleu[3]) + 0.2 * meteor + 0.2 * gleu

    # Loop over the epochs use tqdm to display a progress bar
    with tqdm(range(args.epochs), desc="Epochs") as epoch_bar:
        for epoch in epoch_bar:
            # Set the models to training mode
            visual_extractor.train()
            mlc.train()
            sentLSTM.train()
            wordLSTM.train()

            # Loop over the training set
            for i, (images, _ , tags , captions, prob) in enumerate(train_loader):
                # Zero the gradients of the optimizers
                if (args.visual_model_name != 'hog_pca'):
                    optim_cnn.zero_grad()
                optim_lstm.zero_grad()
                optim_mlc.zero_grad()

                # Move the images, tags, captions, and prob to the device (CPU or GPU)
                images = images.to(args.device)
                tags = tags.to(args.device)
                captions = captions.to(args.device)
                prob = prob.to(args.device)

                # Extract visual features from the images using the visual_extractor model
                if args.visual_model_name == 'hog_pca':
                    visual_features = images
                else:
                    visual_features = visual_extractor(images) # Shape (batch_size, visual_output_embed_dim)

                # Get the semantic tags and features from the multi-label classifier (mlc) model
                semantic_tags , semantic_features = mlc(visual_features)

                # Calculate the loss for the tags using the criterion_tag loss function
                # This measures how well the model predicted the tags
                loss_tag = criterion_tag(semantic_tags, tags)

                # Get the topics and stop probabilities from the sentence LSTM model
                topics, ps = sentLSTM(visual_features, semantic_features , args.s_max, args.device)

                # Calculate the sentence loss using the criterion_stop loss function
                # This measures how well the model predicted when to stop generating a sentence
                # loss_sent = criterion_stop(ps.view(-1, 2), prob.view(-1))
                loss_sent = criterion_stop(ps.reshape(-1, 2), prob.reshape(-1))

                # Initialize the word loss to 0
                loss_word = torch.tensor([0.0]).to(args.device)

                # Loop over each word in the captions
                for j in range(captions.shape[1]):
                    # Generate word outputs using the word LSTM model
                    # The model takes in the current topic and the current word in the caption
                    word_outputs = wordLSTM(topics[:, j, :], captions[:, j, :])
                
                    # Calculate the word loss using the criterion_words loss function
                    # This measures how well the model predicted each word in the caption
                    # The outputs and targets are reshaped to be 1D tensors
                    loss_word += criterion_words(word_outputs.reshape(-1, vocab_size), captions[:, j, :].reshape(-1))
                
                # Calculate the total loss as a weighted sum of the sentence loss, word loss, and tag loss
                # The weights are given by the lambda_sent, lambda_word, and lambda_tag parameters
                loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word + args.lambda_tag * loss_tag
                
                # Backpropagate the loss
                # This computes the gradient of the loss with respect to the model parameters
                loss.backward()

                # Update the parameters of the CNN, LSTM, and MLC models
                # This applies the computed gradients to the model parameters
                if (args.visual_model_name != 'hog_pca'):
                    optim_cnn.step()
                optim_lstm.step()
                optim_mlc.step()
                
                # Get the current learning rates from the schedulers
                lr_cnn = 0
                if (args.visual_model_name != 'hog_pca'):
                    lr_cnn = scheduler_cnn.get_last_lr()[0]
                lr_lstm = scheduler_lstm.get_last_lr()[0]  
                lr_mlc = scheduler_mlc.get_last_lr()[0]

                # Write the losses to a CSV file in the result path
                with open(args.result_path + 'loss.csv', 'a') as f:
                    # # In the first epoch, write the headers to the file
                    if epoch == 0 and i == 0:
                        f.write('Epoch,Loss,Sentence Loss,Word Loss,Tag Loss,Score,Learning Rate CNN,Learning Rate LSTM,Learning Rate MLC\n')
                    # Every log_step iterations, write the current losses to the file
                    if i % args.log_step == 0:
                        f.write(f'{epoch},{loss.item()},{loss_sent.item()},{loss_word.item()},{loss_tag.item()},{best_score},{lr_cnn},{lr_lstm},{lr_mlc}\n')

                # Update the description and postfix of the progress bar
                # This prints the current epoch number and losses to the console
                epoch_bar.set_description(f'Epoch {epoch}')
                epoch_bar.set_postfix(Epoch_Loss=loss.item(), Sentence_LSTM_Loss=loss_sent.item(), Word_LSTM_Loss=loss_word.item(), Tag_Loss=loss_tag.item(), Step=i, Total_Step=total_step, Best_Score = best_score)
            
            # Step the learning rate schedulers for the CNN, LSTM, and MLC models
            # This updates the learning rate based on the current epoch
            if (args.visual_model_name != 'hog_pca'):
                scheduler_cnn.step()
            scheduler_lstm.step()
            scheduler_mlc.step()

            # Every save_epoch epochs, save the current state of the models
            # This allows for resuming training later from this state
            # I want only to save if the loss is less than the previous loss
            if epoch % args.save_epoch == 0:
                if loss.item() < initial_loss:
                    initial_loss = loss.item()
                    # Save the models
                    save_model(visual_extractor, mlc, sentLSTM, wordLSTM, args)
            
            # bleu , meteor , gleu = evaluate(args,visual_extractor, mlc, sentLSTM, wordLSTM)
            # bleu_weight = 0.1
            # meteor_weight = 0.3
            # gleu_weight = 0.3
            
            # combined_score = bleu_weight * (bleu[0] + bleu[1] + bleu[2] + bleu[3]) + meteor_weight * meteor + gleu_weight * gleu
            # if combined_score > best_score:
            #     best_score = combined_score
            #     save_model(visual_extractor, mlc, sentLSTM, wordLSTM, args)

    print('Training completed successfully')
    print('Models saved successfully')
    
    # Generate the loss plot
    generate_loss_plot(args)
    
    # Evaluate the model using the evaluate function
    print ('Start evaluating the model')
    evaluate(args)
    print ('Evaluation completed successfully')
    
    return args, val_loader, visual_extractor, mlc , sentLSTM, wordLSTM, vocab

if __name__ == "__main__":

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

    script(args)