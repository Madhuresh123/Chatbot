#import torch
import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer  #package
stemmer = PorterStemmer() 

def tokenize(sentence):  
    return nltk.word_tokenize(sentence)  #code to tokenize

def stem(word):
    return stemmer.stem(word.lower())  #code to stem and all in lower case

def bag_of_words(tokenized_sentence, all_words):  #will use in train file
    """
    sentence = ['hello','how','are','you']   # compare owrds of both sentence
    words =    ['hi','hello','i','you','bey','thank','cool']
    bag =      [ 0,    1,     0,   1,    0,     0,      0  ]
"""
    sentence_words = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype =np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

"""
sentence =  ['hello','how','are','you'] 
words =    ['hi','hello','i','you','bey','thank','cool']
bag =bag_of_words(sentence,words)

print(bag)
"""