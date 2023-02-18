import json

with open('contents.json', 'r') as f:
	contents = json.load(f)

# Empty List
all_words = []
# trained tag
tags = []
xy = []

for content in contents['contents']:
	tag = content['tag']
	tags.append(tag)
