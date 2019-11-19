import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *

def parseData(fname):
  for l in open(fname):
    yield eval(l)

def parseDataFromURL(fname):
  for l in urlopen(fname):
    yield eval(l)

### Just the first 5000 reviews

print("Reading data...")
# http://cseweb.ucsd.edu/classes/fa19/cse258-a/data/beer_50000.json
data = list(parseData("beer_50000.json"))[:5000]
print("done")

### How many unique words are there?

wordCount = defaultdict(int)
for d in data:
  for w in d['review/text'].split():
    wordCount[w] += 1

print(len(wordCount))

### Ignore capitalization and remove punctuation

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

print(len(wordCount))

### With stemming

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stemmer = PorterStemmer()
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    w = stemmer.stem(w)
    wordCount[w] += 1

### Just take the most popular words...

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]

### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
