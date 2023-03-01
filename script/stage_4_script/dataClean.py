import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

os.chdir('../../data')
if not os.path.exists("./generated_stage_4_data"):
	os.makedirs("./generated_stage_4_data")

# load data
filename = 'data'
file = open(os.getcwd() + "/stage_4_data/text_generation/" + filename, 'rt')
text = file.read()
file.close()

clean_lines = []

# read lines
for line in text.splitlines()[1:]:
  line = line.split('"', 1)[1]

  # convert to lower case
  line = line.lower()

  # remove links
  if 'http' in line:
    fixed = re.sub(r'/\[(.+?)\]\((https?:\/\/[a-zA-Z0-9\/.(]+?)\)/g', r'\1', line)

  # print(line)

  if 'x-post' in line:
    # print(line)
    line = re.sub('\(?x-post from [\w\/ ]+\)?:?', '', line)
    # print(line)

    # print('.......')

  # split into words
  tokens = word_tokenize(line)

  # remove punctuation from each word
  table = str.maketrans('', '', string.punctuation.replace('?','').replace('.',''))
  stripped = [w.translate(table) for w in tokens]

  # remove remaining tokens that are not alphabetic
  # words = [word for word in stripped if word.isalpha() ]

  # # filter out stop words
  # stop_words = set(stopwords.words('english'))
  # words = [w for w in words if not w in stop_words]

  clean_lines.append(stripped)
  
out = '\n'.join([' '.join(line) for line in clean_lines])

print(out, file=open('data_clean', 'w'))

# with open('./data') as f:
#   for line in f:
#     if 'http' in line:
#       print(line)
#       

