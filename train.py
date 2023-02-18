import json

# import nlp standard functions
from nltk_utils import tokenize, stem, bag_of_words

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
