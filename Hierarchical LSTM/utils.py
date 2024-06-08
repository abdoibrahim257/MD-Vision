import torch
import joblib
import os
from dataloader import get_loader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from models import VisualExtractor
import cv2

def save_model(visual_extractor, mlc, sentLSTM, wordLSTM, args):
    """ 
    Function to save the models to the specified paths.

    Args:
    visual_extractor (object): The visual extractor model.
    mlc (object): The MLC model.
    sentLSTM (object): The SentenceLSTM model.
    wordLSTM (object): The WordLSTM model.
    args (object): An object with the necessary arguments including the paths to save the models.
    """
    torch.save(visual_extractor.state_dict(), args.visual_extractor_path)
    torch.save(mlc.state_dict(), args.mlc_path)
    torch.save(sentLSTM.state_dict(), args.sentence_lstm_path)
    torch.save(wordLSTM.state_dict(), args.word_lstm_path)

def split_data(images , tags , captions , test_size = 0.2 , random_state = 42):
    """ 
    Function to split the data into training and validation sets.

    Args:
    images (list): A list of image names.
    tags (list): A list of tags.
    captions (list): A list of captions.
    test_size (float): The proportion of the dataset to include in the validation set.
    random_state (int): The seed used by the random number generator.

    Returns:
    images_train (list): A list of image names for the training set.
    images_val (list): A list of image names for the validation set.
    tags_train (list): A list of tags for the training set.
    tags_val (list): A list of tags for the validation set.
    captions_train (list): A list of captions for the training set.
    captions_val (list): A list of captions for the validation set.
    """
    images_train, images_val, tags_train, tags_val, captions_train, captions_val = train_test_split(images, tags, captions, test_size = test_size, random_state = random_state)
    return images_train, images_val, tags_train, tags_val, captions_train, captions_val

# def separate_images_tags(data):
    
#     return zip(*data)

def save_json(result, result_path):
    """ 
    Function to save the result as a JSON file.

    Args:
    result (dict): The result to be saved.
    result_path (str): The path to save the result.
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    filename = 'result.json'  # change this to your preferred filename
    with open(os.path.join(result_path, filename), 'w') as f:
        json.dump(result, f , indent =4)

def generate_loss_plot(args):
    """
    Function to generate a loss plot from a CSV file.

    Args:
    args (object): An object with the necessary arguments including the path to the CSV file.
    """

    # Print a message to indicate that the loss plot is being generated
    print('Generating loss plot')

    # Read the loss data from the CSV file
    loss_csv = pd.read_csv(args.result_path + 'loss.csv')

    # Plot the total loss, sentence loss, word loss, and tag loss
    plt.plot(loss_csv['Loss'])
    plt.plot(loss_csv['Sentence Loss'])
    plt.plot(loss_csv['Word Loss'])
    plt.plot(loss_csv['Tag Loss'])

    # Set the title, labels, and legend for the plot
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Sentence Loss', 'Word Loss', 'Tag Loss'], loc='upper left')

    # Save the plot as a PNG file
    plt.savefig(args.result_path + 'loss.png')

def train_val_split(args):
    """ 
    Function to split the data into training and validation sets and create DataLoaders for the sets.

    Args:
    args (object): An object with the necessary arguments including the paths to the data files.

    Returns:
    train_loader (object): A DataLoader for the training set.
    val_loader (object): A DataLoader for the validation set.
    vocab (object): The vocabulary object.
    visual_extractor (object): The visual extractor model.
    """

    # Define the transformations to be applied on the images
    # Resize the image to the given size, perform a random crop and then convert the image to a tensor
    transform_train = transforms.Compose([ 
            transforms.Resize(args.img_size),  # Resize the image to the specified size
            transforms.RandomCrop(args.crop_size),  # Perform a random crop of the specified size
            transforms.ToTensor(), #Convert the image to a PyTorch tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize the image
    
    transform_val = transforms.Compose([
            transforms.Resize(args.img_size),  # Resize the image to the specified size
            transforms.CenterCrop(args.crop_size),  # Perform a center crop of the specified size
            transforms.ToTensor(), #Convert the image to a PyTorch tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize the image
    
    # Load the data from the pkl files
    images = joblib.load(args.images_path)
    captions = joblib.load(args.captions_path)
    tags = joblib.load(args.tags_path)
    vocab = joblib.load(args.vocab_path)
    
    # caption is a dictionary with image names as keys and corresponding captions as values
    # tags is a dictionary with image names as keys and corresponding tags as values
    tags = list(tags.values())
    captions = list(captions.values())
    
    # Use split_data to split the dataset into training and validation sets
    images_train, images_val, tags_train, tags_val, captions_train, captions_val = split_data(images, tags, captions, test_size = 0.15, random_state = 42)
    
    # Initialize train_features and val_features as empty tensors
    train_features = None
    val_features = None
    # Initialize the visual extractor model
    visual_extractor = VisualExtractor(args.visual_model_name , args.visual_features_dim ,args.visual_momentum, args.pretrained).to(args.device)
    if args.visual_model_name == 'hog_pca':
        images_itself_train = np.load('./Data/images_train.npy')
        images_itself_val = np.load('./Data/images_val.npy')    
        train_features  = visual_extractor(images_itself_train)
        val_features  = visual_extractor(images_itself_val)

    # Create a DataLoader for the training set
    train_loader = get_loader(images = images_train, tags = tags_train, captions = captions_train, vocab = vocab, transform = transform_train, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = args.num_workers, pca_features = train_features, s_max = args.s_max, w_max = args.w_max)
    
    # Create a DataLoader for the validation set
    val_loader = get_loader(images = images_val, tags = tags_val ,captions = captions_val, vocab = vocab, transform = transform_val, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pca_features = val_features, s_max = args.s_max, w_max = args.w_max)

    
    return train_loader, val_loader, vocab , visual_extractor