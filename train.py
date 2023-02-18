import json

# import nlp standard functions
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('contents.json', 'r') as f:
	contents = json.load(f)

# Empty List
all_words = []
# trained tag
tags = []
xy = []

## classification of tag, patterns
for content in contents['contents']:
	# only one element, use append
	tag = content['tag']
	tags.append(tag)
	for pattern in content['patterns']:
		# which is an array, use extend
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w, tag))

# ignore punctuation marks
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# exclude repeated data
all_words = sorted(set(all_words))
print(all_words)


x_Train = []
y_Train = []
## create a bag of word
for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence, all_words)
	x_Train.append(bag)

	label = tags.index(tag)
	y_Train.append(label)

x_Train = np.array(x_Train)
y_Train = np.array(y_Train)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

batch_size = 8
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
