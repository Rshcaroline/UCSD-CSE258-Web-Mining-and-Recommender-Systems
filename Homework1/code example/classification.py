import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_descriptions_50000.json"))
print "done"

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

### SVM -- "Judging a book by its cover"

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_images_5000.json"))
print "done"

X = [b['image_feature'] for b in data]
y = ["Children's Books" in b['categories'] for b in data]

X_train = X[:2500]
y_train = y[:2500]

X_test = X[2500:]
y_test = y[2500:]

# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=1000, kernel='linear')
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
