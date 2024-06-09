import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import nltk
from pathlib import Path
from torchvision import transforms
import joblib
import numpy as np
import pickle
from build_vocab import Vocabulary

curr_path = Path(os.getcwd())
Indiana_path =  os.path.join(str(curr_path), 'Data')

class IU_Xray(Dataset):
    def __init__(self, images = None , tags = None , captions = None , vocab = None, transform = None , pca_features = None ,s_max=8, w_max=40):
        
        self.images = images # Got the images after processing the findings
        self.tags = tags # Got the tags after processing the findings
        self.captions = captions # Got the captions after processing the findings
        self.vocab = vocab
        self.transform = transform
        self.tags_dict = {} # Dictionary to store the tags
        self.captions_dict = {} # Dictionary to store the captions
        self.pca_features = pca_features
        self.s_max = s_max # Maximum number of sentences
        self.w_max = w_max # Maximum number of words in a sentence
        
        # Create a dictionary to store the tags
        for image , tag in zip(self.images , self.tags):
            self.tags_dict[image] = tag
        
        # Create a dictionary to store the captions
        for image , caption in zip(self.images , self.captions):
            self.captions_dict[image] = caption

    def __len__(self):
        # Number of images
        return len(self.images)

    def __getitem__(self, idx):
        if self.pca_features is not None:
            image_feature = self.pca_features[idx, :]
        # Get the image name and open the image file
        img_name = self.images[idx]
        image = Image.open(f'{Indiana_path}/Images/{img_name}').convert('RGB')
        
        # If a transform function is provided, apply it to the image
        if self.transform is not None:
            image = self.transform(image)
            
        # Get the corresponding caption for the image
        caption = self.captions_dict[img_name]
        
        # Get the corresponding tags for the image in a tensor
        tags = self.tags_dict[img_name]
        
        target = list()
        max_word_num = 0
        counter = 0
        for sentence in caption:
            # Break the loop if the maximum number of sentences is reached
            if counter >= self.s_max:
                break
        
            # Tokenize the sentence
            tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
            if len(tokens) == 0 or len(tokens) == 1 or len(tokens) > self.w_max:
                continue
        
            # Convert the tokens to a list of indices
            tokens = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        
            # Increment the counter
            counter += 1
        
        if self.pca_features is not None:
            image = image_feature

        # Return the image, the tensor of sentences, the number of sentences, the length of the longest sentence, and the tags
        return image, img_name, tags, target, len(target), max_word_num


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, tags, captions, prob)"""
    
    # Unpack the data into separate variables
    images, image_id, label, captions, sentence_num, max_word_num = zip(*data)
    
    # Convert the images to a torch tensor
    images = torch.stack(images, 0)

    # Get the maximum number of sentences and words in a sentence
    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)
    
    # Initialize the targets and prob tensors
    targets = np.zeros((len(captions), max_sentence_num , max_word_num))
    prob = np.zeros((len(captions), max_sentence_num ))

    # For loop to iterate over the captions and fill the targets and prob tensors
    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            # print ('The sentence is:', sentence)
            # print ('The length of the sentence is:', len(sentence))
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0
            
    # Convert the targets to a tensor long
    targets = torch.LongTensor(targets)
    # Convert tags to tensor
    tags = torch.Tensor(label)
    # Convert the prob to a tensor long
    prob = torch.LongTensor(prob)

    # Return the images, tags, targets, and prob tensors
    return images, image_id, tags, targets, prob

def get_loader(images , tags , captions , vocab , transform , batch_size , shuffle , num_workers, pca_features = None, s_max=8, w_max=40):
    """
    Returns torch.utils.data.DataLoader for IU_Xray dataset.
    """
    
    # Create an instance of the IU_Xray dataset class
    # This class should handle the loading and preprocessing of the data
    data = IU_Xray(images = images , tags = tags , captions = captions , vocab = vocab , transform = transform , pca_features = pca_features, s_max=s_max, w_max=w_max)
    
    # Create a DataLoader object
    # The DataLoader is responsible for creating batches of data and
    # handling the shuffling and loading of data in parallel
    # The collate_fn argument is a function that the DataLoader uses to collate the data samples into batches
    data_loader = DataLoader(dataset = data, 
                                              batch_size = batch_size,
                                              shuffle = shuffle,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    # Return the DataLoader object and the vocabulary
    # The DataLoader can be iterated over to get batches of data
    return data_loader


# Test the data set

if __name__ == '__main__':
    
    # Define the parameters
    resize = 256
    crop_size = 224
    batch_size = 6
    
    # Test the data loader
    curr_path = Path(os.getcwd())
    Indiana_path =  os.path.join(str(curr_path), 'Data')
    transform_train = transforms.Compose([ 
            transforms.Resize(resize), # Resize the image to 256x256 pixels
            transforms.RandomCrop(crop_size), # Crop the image to 224x224 pixels
            transforms.ToTensor(), #Convert the image to a PyTorch tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize the image
    
    # Load the data from the pkl files
    images = joblib.load(f'{Indiana_path}/images_filenames.pkl')
    captions = joblib.load(f'{Indiana_path}/captions.pkl')
    tags = joblib.load(f'{Indiana_path}/tags.pkl')
    vocab = joblib.load(f'{Indiana_path}/vocab.pkl')
    
    tags = list(tags.values())
    captions = list(captions.values())
    
    
    data_loader = get_loader(images , tags , captions , vocab , transform_train , batch_size = 6, shuffle = True, num_workers = 8, pca_features = None, s_max=8, w_max=40)

    for i, (image, image_id, label, target, prob) in enumerate(data_loader):
        print('Image ID:', image_id) 
        print('Label:', label)
        print ('Label shape:', label.shape) # (batch_size, number_of_tags) (6, 210)
        print('Target:', target)
        print('Target shape:', target.shape) # (batch_size, max_sentence_num, max_word_num) (6, 8, 40)
        print('Prob:', prob)
        print('Prob shape:', prob.shape) # (batch_size, max_sentence_num) (6, 8)
        break
    