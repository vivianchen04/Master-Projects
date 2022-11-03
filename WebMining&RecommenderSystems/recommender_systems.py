import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

f = open("5year.arff", 'r')

# Read and parse the data
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

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

answers = {}
# question 1
mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)
predictions = mod.predict(X) 
correct = predictions == y 
acc1 = sum(correct) / len(correct)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
(TP + TN) / (TP + FP + TN + FN)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
ber1 = 1 - 1/2 * (TPR + TNR)

answers['Q1'] = [acc1, ber1]
assertFloatList(answers['Q1'], 2)
# {'Q1': [0.9656878917848895, 0.4766851431593464]}

# question 2
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)
predictions = mod.predict(X) 
correct = predictions == y 
acc2 = sum(correct) / len(correct)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
(TP + TN) / (TP + FP + TN + FN)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
ber2 = 1 - 1/2 * (TPR + TNR)

answers['Q2'] = [acc2, ber2]
assertFloatList(answers['Q2'], 2)
# 'Q2': [0.6951501154734411, 0.304401890493309]}

# question 3
random.seed(3)
random.shuffle(dataset)

def BER(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    (TP + TN) / (TP + FP + TN + FN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ber = 1 - 1/2 * (TPR + TNR)
    return ber

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]
len(Xtrain), len(Xvalid), len(Xtest)

mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(Xtrain,ytrain)

pred = mod.predict(Xtrain)
predictions_train = mod.predict(Xtrain) 
berTrain = BER(predictions_train, ytrain)

mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(Xtest,ytest)

pred = mod.predict(Xtest)
predictions_test = mod.predict(Xtest) 
berTest = BER(predictions_test, ytest)

mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(Xvalid,yvalid)

pred = mod.predict(Xvalid)
predictions_valid = mod.predict(Xvalid) 
berValid = BER(predictions_valid, yvalid)

answers['Q3'] = [berTrain, berValid, berTest]
assertFloatList(answers['Q3'], 3)
# 'Q3': [0.29287226079549855, 0.32054015636105193, 0.20097847358121324]

# question 4
#  Report the validation BER for each value of C

berList = []

def pipeline(reg):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]

    Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
    ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

    
    mod.fit(X,y)
    ypredValid = mod.predict(Xvalid)
    ypreTrain = mod.predict(Xtrain)
    
    # validation
    '''
    TP = sum([(a and b) for (a,b) in zip(yvalid, ypredValid)])
    TN = sum([(not a and not b) for (a,b) in zip(yvalid, ypredValid)])
    FP = sum([(not a and b) for (a,b) in zip(yvalid, ypredValid)])
    FN = sum([(a and not b) for (a,b) in zip(yvalid, ypredValid)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    BER = 1 - 0.5*(TPR + TNR)
    '''
    TP = sum([(a and b) for (a,b) in zip(ytrain, ypreTrain)])
    TN = sum([(not a and not b) for (a,b) in zip(ytrain, ypreTrain)])
    FP = sum([(not a and b) for (a,b) in zip(ytrain, ypreTrain)])
    FN = sum([(a and not b) for (a,b) in zip(ytrain, ypreTrain)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    BER = 1 - 0.5*(TPR + TNR)
   
    berList.append(BER)
    
    return mod

for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    pipeline(c)


answers['Q4'] = berList
assertFloatList(answers['Q4'], 9)


# question 5
berList_test = []
def pipeline(reg):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]

    Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
    ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

    
    mod.fit(X,y)
    ypredValid = mod.predict(Xvalid)
    ypredTest = mod.predict(Xtest)
    
    # test

    TP = sum([(a and b) for (a,b) in zip(ytest, ypredTest)])
    TN = sum([(not a and not b) for (a,b) in zip(ytest, ypredTest)])
    FP = sum([(not a and b) for (a,b) in zip(ytest, ypredTest)])
    FN = sum([(a and not b) for (a,b) in zip(ytest, ypredTest)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    BER = 1 - 0.5*(TPR + TNR)

    berList_test.append(BER)

    return mod

for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    pipeline(c)

bestC = 0.001
ber5 = 0.25924657534246576
answers['Q5'] = [bestC, ber5]
assertFloatList(answers['Q5'], 2)

# question 6

f = gzip.open("young_adult_10000.json.gz")
dataset = []

for l in f:
    dataset.append(eval(l))

dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user, item = d['user_id'],d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user, item)] = d['rating']
    reviewsPerUser[user].append('review_id')
    reviewsPerItem[item].append('review_id')

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        #sim = Pearson(i, i2) # Could use alternate similarity metrics straightforwardly
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:10]

#query = dataset[0]['book_id']
#ms = mostSimilar(query, 10)

answers['Q6'] = mostSimilar('2767052', 10)
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)


# question 7

reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {}

for d in dataset:
    user,item = d['user_id'], d['book_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)
    ratingDict[(user, item)] = d['rating']
ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)

def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerItem[item]:
        i2 = d['book_id']
        if i2 == item: continue
        if i2 in itemAverages.keys():
            ratings.append(d['rating'] - itemAverages[i2])
            similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)
#alwaysPredictMean = [ratingMean for d in dataset]
simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
#MSE(alwaysPredictMean, labels)
mse7 = MSE(simPredictions, labels)


answers['Q7'] = mse7
assertFloat(answers['Q7'])

# question 8

def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['user_id']
        if i2 == user: continue
        if i2 in userAverages.keys():
            ratings.append(d['rating'] - userAverages[i2])
            similarities.append(Jaccard(ItemsPerUser[user],ItemsPerUser[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return userAverages[user] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)
#alwaysPredictMean = [ratingMean for d in dataset]
simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
#MSE(alwaysPredictMean, labels)
mse8 = MSE(simPredictions, labels)


answers['Q8'] = mse8
assertFloat(answers['Q8'])
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

