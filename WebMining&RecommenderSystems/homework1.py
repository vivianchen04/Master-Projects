import numpy          # linear algebra
import urllib         # load data from the web
import scipy.optimize # optimization routines
import random         # random number generation
import ast
import csv
#import dateutil.parser # for parsing plain-text dates
#import demjson # json (loose formatting)
import json
import math
#import matplotlib.pyplot as plt
import numpy
import sklearn
from collections import defaultdict
from sklearn import linear_model

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

def parseDataFromFile(fname):
  for l in open(fname):
    #yield eval(l)
    yield ast.literal_eval(l)

data = list(parseDataFromFile("C:/Users/vivia/Desktop/young_adult_10000.json"))

# Q1 
def feature(d):
  feat = d['review_text'].count('!')
  return [1] + [feat]

X = [feature(x) for x in data]
y = [d['rating'] for d in data]

#theta, residuals, rank, s = numpy.linalg.lstsq(X,y)
#theta
#array([3.68853304, 0.07109019])

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_


y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
#mse 1.5231747404538243

#Q2
def feature(d):
    feat = [1] # Constant feature
    feat.append(len(d['review_text'])) # Character Length
    feat.append(d['review_text'].count('!')) # Number of !
    return feat
X = [feature(x) for x in data]
y = [d['rating'] for d in data]
#theta, residuals, rank, s = numpy.linalg.lstsq(X,y)
#theta
#array([ 3.71751281e+00, -4.12150653e-05,  7.52759173e-02])


model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
# 1.521402924616585


#Q3
# 1
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    return feat

X = [feature(x) for x in data]
y = [d['rating'] for d in data]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
# 1.5231747404538243

# 1 - 2
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    return feat

X = [feature(x) for x in data]
y = [d['rating'] for d in data]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
# 1.504668610625097


# 1 - 3
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    feat.append((d['review_text'].count('!'))**3)
    return feat
X = [feature(x) for x in data]
y = [d['rating'] for d in data]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
# 1.4966845515179565

# 1- 4
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    feat.append((d['review_text'].count('!'))**3)
    feat.append((d['review_text'].count('!'))**4)
    return feat

X = [feature(x) for x in data]
y = [d['rating'] for d in data]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
#1.4904477302230832


# 1 - 5
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    feat.append((d['review_text'].count('!'))**3)
    feat.append((d['review_text'].count('!'))**4)
    feat.append((d['review_text'].count('!'))**5)
    return feat

X = [feature(x) for x in data]
y = [d['rating'] for d in data]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
#1.4896106953960724


# Q4
# split 50%/50% train/test fractions
#training = data[0:5000]
#testing = data[5000:]
#X_train = X[:len(X)//2]
#y_train = y[:len(X)//2]

#X_test = X[len(X)//2:]
#y_test = y[len(X)//2:]

# Q1 testing
def feature(d):
  feat = d['review_text'].count('!')
  return [1] + [feat]

#X = [feature(x) for x in testing]
#y = [d['rating'] for d in testing]
X = [feature(x) for x in data]
y = [d['rating'] for d in data]

X_train = X[:len(X)//2]
y_train = y[:len(X)//2]

X_test = X[len(X)//2:]
y_test = y[len(X)//2:]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X_test, y_test)
y_pred = model.predict(X_test)
sse = sum([x**2 for x in (y_test - y_pred)])
mse = sse / len(y_test)
#mse 1.50944896079672


# Q2 testing
def feature(d):
    feat = [1] # Constant feature
    feat.append(len(d['review_text'])) # Character Length
    feat.append(d['review_text'].count('!')) # Number of !
    return feat

X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
# mse 1.496956555076229


# Q3 testing

# 1
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    return feat

X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse1 = sse / len(y)
# 1.50944896079672

# 1 - 2
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    return feat

X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse2 = sse / len(y)
# 1.4859502154048487


# 1 - 3
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    feat.append((d['review_text'].count('!'))**3)
    return feat
X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse3 = sse / len(y)
# 1.4739087039506664

# 1- 4
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    feat.append((d['review_text'].count('!'))**3)
    feat.append((d['review_text'].count('!'))**4)
    return feat

X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse4 = sse / len(y)
#1.4610972293669993


# 1 - 5
def feature(d):
    feat = [1]
    feat.append(d['review_text'].count('!'))
    feat.append((d['review_text'].count('!'))**2) # Quadratic term
    feat.append((d['review_text'].count('!'))**3)
    feat.append((d['review_text'].count('!'))**4)
    feat.append((d['review_text'].count('!'))**5)
    return feat

X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([x**2 for x in (y - y_pred)])
mse5 = sse / len(y)
#1.4579954828854338

# Q5
# MAE
def feature(d):
    feat = [1]
    return feat

X = [feature(x) for x in testing]
y = [d['rating'] for d in testing]

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta = model.coef_

y_pred = model.predict(X)
sse = sum([abs(x) for x in (y - theta[0])])
mae = sse / len(y)
# mae 0.98172024


# Classification
 
data = list(parseDataFromFile("C:/Users/vivia/Desktop/beer_50000.json"))
data = [d for d in data if 'user/gender' in d]

# Q6
X = [[1, d['review/text'].count('!')] for d in data]
y = [d['user/gender'] == 'Female' for d in data]

mod = sklearn.linear_model.LogisticRegression()
mod.fit(X,y)
predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)
output = [TP, TN, FP, FN, BER]
# [0, 20095, 0, 308, 0.5]

# Q7
X = [[1, d['review/text'].count('!')] for d in data]
y = [d['user/gender'] == 'Female' for d in data]

mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)
output = [TP, TN, FP, FN, BER]
# [88, 16332, 3763, 220, 0.4507731134255145]

# Q8
scores = mod.decision_function(X)
scoreslabels = list(zip(scores, y))
scoreslabels.sort(reverse=True)
sortedlabels = [x[1] for x in scoreslabels]
res = []
K = [1,10,100,1000,10000]
for k in K: 
  res.append(sum(sortedlabels[:k]) / k)
# res
#[0.0, 0.0, 0.03, 0.033, 0.0308]
