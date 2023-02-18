import nltk
# nltk.download("punkt")
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())
"""
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
def bag_of_words(tokenized_sentence, all_words):
	tokenized_sentence = [stem(w) for w in tokenized_sentence]
	bag = np.zeros(len(all_words), dtype = np.float32)
	for idx, w in enumerate(all_words):
		if w in tokenized_sentence:
			bag[idx] = 1.0
	return bag
    
## Bag_of_words Testing
sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bog = bag_of_words(sentence, words)
# print(bog)

## Tokenization_Testing
a = "How are you bitch??"
a = tokenize(a)
# print(a)

## Stemming_Testing 
words = ["universe", "university", "universal"]
stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

