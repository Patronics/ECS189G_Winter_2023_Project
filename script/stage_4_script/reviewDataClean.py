import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('punkt')

os.chdir('../../data')
if not os.path.exists("./generated_stage_4_data"):
	os.makedirs("./generated_stage_4_data")

# load data
folder = 'train'

outfilepath = os.getcwd() + '/generated_stage_4_data/review_data_clean_' + folder
outfile = open(outfilepath, 'w')
print ("---- Processing /stage_4_data/text_classification/" + folder + " ----")

HTML_REMOVER = re.compile('<.*?>') 

def read_files(path, label):
  output = []

  for filename in os.listdir(path):
      full_filename = os.path.join(path, filename)

      if os.path.isfile(full_filename):
          with open(full_filename) as f:
            line = f.read()

            # remove html 
            line = re.sub(HTML_REMOVER, '', line)

            # convert to lower case
            line = line.lower()

            # remove links
            if 'http' in line:
              #remove raw links without formatting http://example.com
              line = re.sub(r'\(?https?:\/\/[a-zA-Z0-9\/.(&\?=\-;_]+\)?', '', line)
            
            tokens = word_tokenize(line)

            # remove all punctuation from each word except for . ? -
            table = str.maketrans('', '', string.punctuation.replace('.', '').replace('?', '').replace('-', ''))
            tokens = [w.translate(table) for w in tokens]

            # remove remaining tokens that are not alphabetic
            tokens = [w for w in tokens if w.isalpha() ]

            # filter out stop words
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]

            # stem words
            porter = PorterStemmer()
            stemmed = [porter.stem(word) for word in tokens]

            # Join line back together, removing empty tokens
            clean_line = ' '.join([token for token in stemmed if len(token)])

            output.append((clean_line, label))
  
  return output

dataFolder = os.getcwd() + "/stage_4_data/text_classification/" + folder

neg = read_files(dataFolder + '/neg', 0)
pos = read_files(dataFolder + '/pos', 1)

out = 'review,label\n' # 0 = negative, 1 = positive
out += '\n'.join([review+ ',' + str(label) for review, label in neg + pos])

print(out, file=outfile)
print ("---- saved output to " + outfilepath+"  ----")
# with open('./data') as f:
#   for line in f:
#     if 'http' in line:
#       print(line)
#       

