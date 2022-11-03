#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
import numpy
import string
import random
from sklearn import linear_model
import sklearn


# In[2]:


# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


answers = {}


# In[5]:


f = open("spoilers.json.gz", 'r')


# In[6]:


dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)


# In[7]:


f.close()


# In[8]:


# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])
    
# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])


# In[9]:


# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[10]:


### 1


# In[11]:


def MSE(y, ypred):
    return sum([(a-b)**2 for (a,b) in zip(y,ypred)]) / len(y)


# In[12]:


# (a)
y = []
y_pred = []
for u in reviewsPerUser:
    cur = []
    reviews = reviewsPerUser[u]
    for i in range(0, len(reviews) - 1):
        cur.append(reviews[i]['rating'])
    if len(cur) == 0:
        continue
    y_pred.append(sum(cur)/len(cur))
    y.append(reviews[-1]['rating'])
answers['Q1a'] = MSE(y, y_pred)
assertFloat(answers['Q1a'])


# In[13]:


# (b)
y = []
y_pred = []
for u in reviewsPerItem:
    cur = []
    reviews = reviewsPerItem[u]
    for i in range(0, len(reviews) - 1):
        cur.append(reviews[i]['rating'])
    if len(cur) == 0:
        continue
    y_pred.append(sum(cur)/len(cur))
    y.append(reviews[-1]['rating'])
answers['Q1b'] = MSE(y, y_pred)
assertFloat(answers['Q1b'])


# In[14]:


### 2
answers['Q2'] = []

for N in [1,2,3]:
    y = []
    y_pred = []
    for u in reviewsPerUser:
        cur = []
        reviews = reviewsPerUser[u]
        for i in range(0, len(reviews) - 1):
            cur.append(reviews[i]['rating'])
        if len(cur) == 0:
            continue
        
        if len(cur) < N:
            cur_new = cur
        
        if len(cur) >= N:
            cur_new = cur[-N:]
        
        y_pred.append(sum(cur_new)/len(cur_new))
        y.append(reviews[-1]['rating'])
            
    answers['Q2'].append(MSE(y,y_pred))


# In[15]:


assertFloatList(answers['Q2'], 3)


# In[16]:


answers


# In[17]:


### 3a


# In[18]:


def feature3(N, u): # For a user u and a window size of N
    
    cur = []
    reviews = reviewsPerUser[u]
    for i in range(0, len(reviews) - 1):
        cur.append(reviews[i]['rating'])
    
    feat = [1]
    for n in range(1, N + 1):
        feat.append(cur[-n])

    return feat
    


# In[19]:


answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]


# In[20]:


assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4


# In[21]:


### 3b
answers['Q3b'] = []
def feat(N, u):
    feat = [1]
    data = reviewsPerUser[u]
    for d in data[-N-1:-1]:
        feat.append(d['rating'])
    return feat

for N in [1,2,3]:
    X = []
    y = []
    for u,data in reviewsPerUser.items():
        if len(data) <= N:
            continue
        else:
            X.append(feat(N,u))
            y.append(data[-1]['rating'])
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = MSE(y, y_pred)
    answers['Q3b'].append(mse)
assertFloatList(answers['Q3b'], 3)
answers


# In[22]:


### 4a
globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)

def featureMeanValue(N, u): # For a user u and a window size of N
    feat = [1]
    data = reviewsPerUser[u]
    if len(data) < N + 1:
        if len(data) < 2:
            for j in range(N):
                feat.append(globalAverage)
        elif len(data) >= 2:
            rate = [review['rating'] for review in data[:-1]]
            avg = sum(rate)/len(rate)
            for i in range(len(data)-1):
                feat.append(data[-i-2]['rating'])
            for i in range(N-len(data)+1):
                feat.append(avg)
    else:
        for i in range(N):
            feat.append(data[-i-2]['rating'])  
    return feat

def featureMissingValue(N, u):
    feat = [1]
    data = reviewsPerUser[u]

    if len(data) < N + 1:
        if len(data) < 2:
            for j in range(N):
                feat.append(1)
                feat.append(0)
        elif len(data) >= 2:
            for i in range(len(data)-1):
                feat.append(0)
                feat.append(data[- i - 2]['rating'])
            for i in range(N + 1-len(data)):
                feat.append(1)
                feat.append(0)
    else:
        for i in range(N):
            feat.append(0)
            feat.append(data[-i-2]['rating'])  
    return feat

answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]

answers


# In[23]:


answers['Q4b'] = []

for featFunc in [featureMeanValue, featureMissingValue]:
    X = []
    y = []
    for user,rating in reviewsPerUser.items():
        if len(rating) < 1:
            continue
        else:
            X.append(featFunc(10,user))
            y.append(rating[-1]['rating'])

    model = linear_model.LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    mse = MSE(y, y_pred) 
    answers['Q4b'].append(mse)


# In[24]:


answers['Q4b']


# In[25]:


### 5
#(a)
def feature5(sentence):
    feat = [1]
    feat.append(len(sentence))
    feat.append(sentence.count('!')) # Quadratic term
    feat.append(sum(i.isupper() for i in sentence))
    return feat

X = []
y = []

for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)


# In[26]:


answers['Q5a'] = X[0]


# In[27]:


###5(b)
mod = sklearn.linear_model.LogisticRegression( class_weight='balanced', C=1)
mod.fit(X,y)
predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)
answers['Q5b'] = [TP, TN, FP, FN, BER]


# In[28]:


assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)


# In[29]:


### 6
def feature6(review):
    review = review['review_sentences']
    feat = [1]
    for i in range(0, 5):
        feat.append(review[i][0])
    feat.append(len(review[5][1]))
    feat.append(review[5].count('!')) # Quadratic term
    feat.append(sum(i.isupper() for i in review[5][1]))
    
    return feat


# In[30]:


y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])


# In[31]:


answers['Q6a'] = feature6(dataset[0])
answers


# In[32]:


answers['Q6a'] = X[0]
answers


# In[33]:


mod = sklearn.linear_model.LogisticRegression(class_weight='balanced', C = 1)
mod.fit(X,y)
predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)

answers['Q6b'] = BER


# In[34]:


assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])
answers


# In[35]:


### 7


# In[36]:


# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[37]:


def pipeline(reg, bers, BER_test):
    mod = linear_model.LogisticRegression(class_weight='balanced', C=reg)
    
    # 50/25/25% train/valid/test split
    Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
    ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]
    
    mod.fit(Xtrain,ytrain)
    ypredValid = mod.predict(Xvalid)
    ypredTest = mod.predict(Xtest)
    
    # validation
    
    TP = sum([(a and b) for (a,b) in zip(yvalid, ypredValid)])
    TN = sum([(not a and not b) for (a,b) in zip(yvalid, ypredValid)])
    FP = sum([(not a and b) for (a,b) in zip(yvalid, ypredValid)])
    FN = sum([(a and not b) for (a,b) in zip(yvalid, ypredValid)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    BER = 1 - 0.5*(TPR + TNR)
    
    print("C = " + str(reg) + "; validation BER = " + str(BER))
    bers = bers.append(BER)
    
     # test

    TP = sum([(a and b) for (a,b) in zip(ytest, ypredTest)])
    TN = sum([(not a and not b) for (a,b) in zip(ytest, ypredTest)])
    FP = sum([(not a and b) for (a,b) in zip(ytest, ypredTest)])
    FN = sum([(a and not b) for (a,b) in zip(ytest, ypredTest)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    BER = 1 - 0.5*(TPR + TNR)
    
    BER_test = BER_test.append(BER)

    return mod


# In[38]:


bers = []
BER_test = []
for c in [0.01, 0.1, 1, 10, 100]:
    pipeline(c, bers, BER_test)
bers
BER_test


# In[39]:


bestC = 0.1
ber = 0.21299572460563176
answers['Q7'] = bers + [bestC] + [ber]
assertFloatList(answers['Q7'], 7)
answers


# In[40]:


### 8
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[41]:


# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]


# In[42]:


# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)


# In[43]:


reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)


# In[44]:


# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[45]:


predictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]


# In[46]:


answers["Q8"] = MSE(predictions, labels)
assertFloat(answers["Q8"])


# In[ ]:





# In[56]:


### 9
item = [d['book_id'] for d in dataTrain]
data0, rating0 =  [], []

for d in dataTest:
    num = item.count(d['book_id'])
    if num == 0:
        data0.append([d['user_id'], d['book_id']])
        rating0.append(d['rating'])
        
pred0 = [predictRating(u, i) for u, i in data0]

mse0 = MSE(pred0, rating0)
mse0


# In[57]:


data1, rating1 = [],[]

for d in dataTest:
    num = item.count(d['book_id'])
    
    if 1 <= num <= 5:
        data1.append([d['user_id'], d['book_id']])
        rating1.append(d['rating'])
        

pred1 = [predictRating(u, i) for u, i in data1]

mse1to5= MSE(pred1, rating1)
mse1to5


# In[58]:


data5, rating5 = [], []

for d in dataTest:
    num = item.count(d['book_id'])
        
    if num > 5:
        data5.append([d['user_id'], d['book_id']])
        rating5.append(d['rating'])

pred5 = [predictRating(u, i) for u, i in data5]

mse5 = MSE(pred5, rating5)
mse5


# In[ ]:





# In[50]:


answers["Q9"] = [mse0, mse1to5, mse5]
assertFloatList(answers["Q9"], 3)

answers


# In[51]:


### 10


# In[52]:


userAverages = defaultdict(list)

for d in dataTrain:
    userAverages[d['user_id']].append(d['rating'])

for i in userAverages:
    userAverages[i] = sum(userAverages[i]) / len(userAverages[i])


def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            # return RatingMean
            if user in userAverages:
                return userAverages[user]
            else:
                return ratingMean
            
            
item = [d['book_id'] for d in dataTrain]
data10, rating10 = [], []

for d in dataTest:
    num = item.count(d['book_id'])
    if num == 0:
        data10.append([d['user_id'], d['book_id']])
        rating10.append(d['rating'])
        
pred10 = [predictRating(u, i) for u, i in data10]

mse10 = MSE(pred10, rating10)
mse10


# In[59]:


answers["Q10"] = ("To improve the prediction function for unseen items, we can modify the predictRating function. Since previously the predictRating only use itemAverages for prediction function, we can add the userAverage to specify the condition and make mse smaller, inside of just categorize data into ratingMean. We can see that the mse become smaller for unseen data.", mse10)
assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])


# In[60]:


answers


# In[55]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()

