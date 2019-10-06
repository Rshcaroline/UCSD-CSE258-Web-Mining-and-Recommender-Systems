import numpy
import urllib
import scipy.optimize
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))
print "done"

def feature(datum):
  feat = [1]
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### Convince ourselves that basic linear algebra operations yield the same answer ###

X = numpy.matrix(X)
y = numpy.matrix(y)
numpy.linalg.inv(X.T * X) * X.T * y.T

### Do older people rate beer more highly? ###

data2 = [d for d in data if d.has_key('user/ageInSeconds')]

def feature(datum):
  feat = [1]
  #feat.append(datum['user/ageInSeconds'])
  feat.append(datum['beer/ABV'])
  return feat

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### How much do women prefer beer over men? ###

data2 = [d for d in data if d.has_key('user/gender')]

def feature(datum):
  feat = [1]
  if datum['user/gender'] == "Male":
    feat.append(0)
  else:
    feat.append(1)
  return feat

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### Gradient descent ###

# Objective
def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  print "offset =", diffSqReg.flatten().tolist()
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  print "gradient =", numpy.array(res.flatten().tolist()[0])
  return numpy.array(res.flatten().tolist()[0])

scipy.optimize.fmin_l_bfgs_b(f, [0,0], fprime, args = (X, y, 0.1))

### Random features ###

def feature(datum):
  return [random.random() for x in range(30)]

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
