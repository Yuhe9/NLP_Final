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


df = pd.read_csv(
	"simplified_data.csv",
	usecols = ['Text', "Helpfulness", "Sentiment"],
	encoding = "ISO-8859-1"
	)
train, test = train_test_split(df, test_size=0.2)

# build list from numpy array
train_list = train["Text"].values
test_list = test["Text"].values
#uniqueVals = np.unique(my_list)
#theList = df['Text'].values
# create the transform
vectorizer = CountVectorizer(stop_words = 'english')

# tokenize and build vocab
vectorizer.fit(train_list)
print(train_list[0])
trainFeatures = vectorizer.transform(train_list)
testFeatures = vectorizer.transform(test_list)
# print(trainFeatures.shape)
# print(len(vectorizer.vocabulary_))
# print(vectorizer.vocabulary_.get(u'strawberry'))
# print(trainFeatures[0])

clf = MultinomialNB().fit(trainFeatures,train["Helpfulness"])
