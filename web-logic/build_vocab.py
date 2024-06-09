import argparse
from tqdm import tqdm
from collections import Counter
import pickle
import nltk
import json
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

def build_vocab(caption_list, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    # Initialize NLTK tokenizer
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # Use tqdm for progress bar
    for sentence in tqdm(caption_list , desc='Building Vocabulary'):
        tokens = tokenizer.tokenize(sentence.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab

def main(caption_path, vocab_path, threshold):
    # Load the JSON file
    with open(caption_path, 'r') as f:
        data = json.load(f)

    # Extract all sentences
    sentences = []
    for key in data:
        sentences.extend(data[key])

    vocab = build_vocab(sentences, threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, default='Data/captions.json', help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./Data/vocab.pkl', help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default= 1, help='minimum word count threshold')
    args = parser.parse_args()
    main(args.caption_path, args.vocab_path, args.threshold)      
