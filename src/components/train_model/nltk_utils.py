import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

import sys
from src.exception import CustomException
from src.logger import logging

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    try:
        logging.info('Tokenize the words of the given sentence')
        return nltk.word_tokenize(sentence)
    except Exception as e:
            raise CustomException(e,sys)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    try:
        logging.info("Returning the list of words")
        return stemmer.stem(word.lower())
    except Exception as e:
            raise CustomException(e,sys)

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    try:
        # stem each word
        sentence_words = [stem(word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words: 
                bag[idx] = 1
        logging.info("Return list in the form of binary")
        return bag
    except Exception as e:
            raise CustomException(e,sys)