import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm
from sklearn import linear_model

def parseDataFromURL(fname):
  for l in urlopen(fname):
    yield eval(l)

def parseData(fname):
  for l in open(fname):
    yield eval(l)

print("Reading data...")
# Download from http://jmcauley.ucsd.edu/cse258/data/amazon/book_descriptions_50000.json
data = list(parseData("data/amazon/book_descriptions_50000.json"))
print("done")

### Naive bayes to determine p(childrens book | mentions wizards and mentions witches) ###

# p(childrens book)
prior = ["Children's Books" in b['categories'] for b in data]
prior = sum(prior) * 1.0 / len(prior)

# p(isn't children's book)
prior_neg = 1 - prior

# p(mentions wizards | is childrens)
p1 = ['wizard' in b['description'] for b in data if "Children's Books" in b['categories']]
p1 = sum(p1) * 1.0 / len(p1)

# p(mentions wizards | isn't childrens)
p1_neg = ['wizard' in b['description'] for b in data if not ("Children's Books" in b['categories'])]
p1_neg = sum(p1_neg) * 1.0 / len(p1_neg)

# p(mentions witches | is childrens)
p2 = ['witch' in b['description'] for b in data if "Children's Books" in b['categories']]
p2 = sum(p2) * 1.0 / len(p2)

# p(mentions witches | isn't childrens)
p2_neg = ['witch' in b['description'] for b in data if not ("Children's Books" in b['categories'])]
p2_neg = sum(p2_neg) * 1.0 / len(p2_neg)

# Prediction

score = prior * p1 * p2
score_neg = prior_neg * p1_neg * p2_neg

# Actual ('non-naive') probability

p = ["Children's Books" in b['categories'] for b in data if 'witch' in b['description'] and 'wizard' in b['description']]
p = sum(p) * 1.0 / len(p)

### Logistic Regression -- "Judging a book by its cover"

print("Reading data...")
# Download from http://jmcauley.ucsd.edu/cse255/data/amazon/book_images_5000.json
data = list(parseData("data/amazon/book_images_5000.json"))
print("done")

X = [b['image_feature'] for b in data]
y = ["Children's Books" in b['categories'] for b in data]

X_train = X[:2500]
y_train = y[:2500]

X_test = X[2500:]
y_test = y[2500:]

# Create a support vector classifier object, with regularization parameter C = 1000
# clf = svm.SVC(C=1000, kernel='linear')
# clf.fit(X_train, y_train)

# train_predictions = clf.predict(X_train)
# test_predictions = clf.predict(X_test)

# Logistic regression classifier
mod = linear_model.LogisticRegression(C=1.0)
mod.fit(X_train, y_train)

train_predictions = mod.predict(X_train)
test_predictions = mod.predict(X_test)


### Diagnostics

# From https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
f = open("5year.arff", 'r')

# Reading in data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

# Data setup
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# Fit model
mod = linear_model.LogisticRegression(C=1.0)
mod.fit(X,y)

pred = mod.predict(X)

# How many positive predictions?
sum(pred)

# Balanced model
mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)

# How many positive predictions?
sum(pred)

# Train/validation/test splits

# Shuffle the data
Xy = list(zip(X,y))
random.shuffle(Xy)

X = [d[0] for d in Xy]
y = [d[1] for d in Xy]

N = len(y)

Ntrain = 1000
Nvalid = 1000
Ntest = 1031

Xtrain = X[:Ntrain]
Xvalid = X[Ntrain:Ntrain+Nvalid]
Xtest = X[Ntrain+Nvalid:]

ytrain = y[:Ntrain]
yvalid = y[Ntrain:Ntrain+Nvalid]
ytest = y[Ntrain+Nvalid:]

mod.fit(Xtrain, ytrain)

pred = mod.predict(Xtest)

correct = pred == ytest

# True positives, false positives, etc.

TP_ = numpy.logical_and(pred, ytest)
FP_ = numpy.logical_and(pred, numpy.logical_not(ytest))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(ytest))
FN_ = numpy.logical_and(numpy.logical_not(pred), ytest)

TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)

# accuracy
sum(correct) / len(correct)
(TP + TN) / (TP + FP + TN + FN)

# BER
1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP))

# Ranking

scores = mod.decision_function(Xtest)

scores_labels = list(zip(scores, ytest))
scores_labels.sort(reverse = True)

sortedlabels = [x[1] for x in scores_labels]

# precision / recall
retrieved = sum(pred)
relevant = sum(ytest)
intersection = sum([y and p for y,p in zip(ytest,pred)])

precision = intersection / retrieved
recall = intersection / relevant

# precision at 10
sum(sortedlabels[:10]) / 10
