# Note: pin install pandas

#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
from pprint import pformat
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics

df = pd.read_csv(
	"simplified_data.csv",
	usecols = ['Text', 'Summary', "Helpfulness", "Sentiment"],
	encoding = "ISO-8859-1"
	)
train, test = train_test_split(df, test_size=0.2)

train_list = []
test_list = []
# build list from numpy array
print(df.head(5))

train_summary_list = train["Summary"].apply(lambda x: np.str_(x)).values.tolist()
train_list.append(train_summary_list)
test_summary_list = test["Summary"].apply(lambda x: np.str_(x)).values
test_list.append(test_summary_list)

train_text_list = train["Text"].values

print(type(train_text_list))

train_text_list = train["Text"].values.tolist()
train_list.append(train_text_list)
test_text_list = test["Text"].values.tolist()
test_list.append(test_text_list)
train_list = list(map(lambda x, y: [x,y], train_summary_list, train_text_list))

train_list = np.array(train_list, np.string_)
print(type(train_list))
test_list = np.array(test_list, np.string_)

# create the transform
vectorizer = CountVectorizer(stop_words = 'english')

# tokenize and build vocab
vectorizer.fit(train_list)

trainFeatures = vectorizer.transform(train_list.ravel())
testFeatures = vectorizer.transform(test_list.ravel())

# Use frequency as feature
tf_transformer = TfidfTransformer(use_idf=False).fit(trainFeatures)
train_tf = tf_transformer.transform(trainFeatures)
test_tf = tf_transformer.transform(testFeatures)



prediction = dict()
prediction2 = dict()
predictionP = dict()
predictionP2 = dict()

# Multinomial Model:::::::::::::
clf = MultinomialNB().fit(trainFeatures,train["Helpfulness"])
prediction['Multinomial']  = clf.predict(testFeatures)

print("### Helpfulness Task with Occurence Feature ###")
#Accuracy:
print("Multinomial:")
print(np.mean(prediction['Multinomial'] == test["Helpfulness"]))
#Other Result Data
print(metrics.classification_report(test["Helpfulness"], prediction['Multinomial'], target_names = ["positive", "negative"]))


# Bernoulli Naive Bayes
nb = BernoulliNB().fit(trainFeatures,train["Helpfulness"])
prediction['Bernoulli'] = nb.predict(testFeatures)
#Accuracy:
print("Bernoulli:")
print(np.mean(prediction['Bernoulli'] == test["Helpfulness"]))
#Other Result Data
print(metrics.classification_report(test["Helpfulness"], prediction['Bernoulli'], target_names = ["positive", "negative"]))



############## Frequency as Feature ############

# Multinomial Model:::::::::::::
clf2 = MultinomialNB().fit(train_tf,train["Helpfulness"])
prediction2['Multinomial']  = clf2.predict(test_tf)
print("### Helpfulness Task with Frequency Feature ###")
#Accuracy:
print("Multinomial:")
print(np.mean(prediction2['Multinomial'] == test["Helpfulness"]))
#Other Result Data
print(metrics.classification_report(test["Helpfulness"], prediction2['Multinomial'], target_names = ["positive", "negative"]))


# Bernoulli Naive Bayes
nb2 = BernoulliNB().fit(train_tf,train["Helpfulness"])
prediction2['Bernoulli'] = nb2.predict(test_tf)

#Accuracy:
print("Bernoulli:")
print(np.mean(prediction2['Bernoulli'] == test["Helpfulness"]))
#Other Result Data
print(metrics.classification_report(test["Helpfulness"], prediction2['Bernoulli'], target_names = ["positive", "negative"]))


############ Second Task ##############

# Multinomial Model:::::::::::::
clf3 = MultinomialNB().fit(trainFeatures,train["Sentiment"])
predictionP['Multinomial']  = clf3.predict(testFeatures)
print("### Sentiment Task with Occurence Feature ###")
#Accuracy:
print("Multinomial:")
print(np.mean(predictionP['Multinomial'] == test["Sentiment"]))
#Other Result Data
print(metrics.classification_report(test["Sentiment"], predictionP['Multinomial'], target_names = ["positive", "negative"]))


# Bernoulli Naive Bayes
nb3 = BernoulliNB().fit(trainFeatures,train["Sentiment"])
predictionP['Bernoulli'] = nb3.predict(testFeatures)
#Accuracy:
print("Bernoulli:")
print(np.mean(predictionP['Bernoulli'] == test["Sentiment"]))
#Other Result Data
print(metrics.classification_report(test["Sentiment"], predictionP['Bernoulli'], target_names = ["positive", "negative"]))



############## Frequency as Feature ############

# Multinomial Model:::::::::::::
clf4 = MultinomialNB().fit(train_tf,train["Sentiment"])
predictionP2['Multinomial']  = clf4.predict(test_tf)
print("### Sentiment Task with Frequency Feature ###")

#Accuracy:
print("Multinomial:")
print(np.mean(predictionP2['Multinomial'] == test["Sentiment"]))
#Other Result Data
print(metrics.classification_report(test["Sentiment"], predictionP2['Multinomial'], target_names = ["positive", "negative"]))


# Bernoulli Naive Bayes
nb4 = BernoulliNB().fit(train_tf,train["Sentiment"])
predictionP2['Bernoulli'] = nb4.predict(test_tf)

#Accuracy:
print("Bernoulli:")
print(np.mean(predictionP2['Bernoulli'] == test["Sentiment"]))
#Other Result Data
print(metrics.classification_report(test["Sentiment"], predictionP2['Bernoulli'], target_names = ["positive", "negative"]))

