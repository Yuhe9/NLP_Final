# Note: pin install pandas

#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
from pprint import pformat
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv(
	"simplified_data.csv",
	usecols = ['Sentiment','Helpfulness','Text']
	)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')
print(X_train_counts.head(5))
# add feature column representing feature in binary if use occurence as feature
# df["Feature"] = 