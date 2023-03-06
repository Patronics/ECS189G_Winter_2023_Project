import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')


os.chdir('../../data')
if not os.path.exists("./generated_stage_4_data"):
	os.makedirs("./generated_stage_4_data")

# load data
filename = 'data'
outfilepath = os.getcwd() + '/generated_stage_4_data/joke_data_clean'
print ("---- Processing /stage_4_data/text_generation/" + filename + " ----")
file = open(os.getcwd() + "/stage_4_data/text_generation/" + filename, 'rt')
outfile = open(outfilepath, 'w')
text = file.read()
file.close()

clean_lines = []

# read lines
for line in text.splitlines()[1:]:
  line = line.split('"', 1)[1]

  # convert to lower case
  line = line.lower()
  
  #lines that start with subreddit names are not jokes
  if line.startswith('/r/'):
    continue

  # remove lines mentioning reddit
  if 'reddit' in line:
    continue

  # remove links
  if 'http' in line:
    #remove  links with [link formatting](http://example.com)
    line = re.sub(r'\[(.+?)\]\((https?:\/\/[a-zA-Z0-9\/.(&\?=\-;_]+?)\)', r'\1', line)
    #remove raw links without formatting http://example.com
    line = re.sub(r'\(?https?:\/\/[a-zA-Z0-9\/.(&\?=\-;_]+\)?', '', line)
    #print(line)

  # remove annotations/citations of format [annotation] 
  line = re.sub(r'\[(.+?)\]', '', line)     
  
  #remove mention of cross-posts
  if any(s in line for s in ['x-post','x post','xpost']):
    # print(line)
    line = re.sub('\(?x.?post from [\w\/ ]+\)?:?', '', line)

  #mentions of subreddit names
  #if 'r/' in line:
  #  print (line)
  # split into words
  
  tokens = word_tokenize(line)

  # remove all punctuation from each word except for . ? -
  allowed_punctuation = ['?', '.', '-']
  
  table = str.maketrans('', '', string.punctuation.replace('.', '').replace('?', '').replace('-', ''))

  stripped = [w.translate(table) for w in tokens]

  # remove remaining tokens that are not alphabetic
  # words = [word for word in stripped if word.isalpha() ]

  # # filter out stop words
  # stop_words = set(stopwords.words('english'))
  # words = [w for w in words if not w in stop_words]

  # Remove empty strings
  clean_lines.append([token for token in stripped if len(token)]) 

# print(clean_lines)

#endOfJokeToken = ''
endOfJokeToken = ' <EOJ>'

out = str(endOfJokeToken+'\n').join([' '.join(line) for line in clean_lines])
# out = clean_lines

print(out, file=outfile)
print ("---- saved output to " + outfilepath+"  ----")
# with open('./data') as f:
#   for line in f:
#     if 'http' in line:
#       print(line)
#       

