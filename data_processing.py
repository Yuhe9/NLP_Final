# Note: pin install pandas

#!/usr/bin/python
#pip install pyspellchecker
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import string
import nltk
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from spellchecker import SpellChecker



df = pd.read_csv(
	"Reviews.csv",
	usecols = ['HelpfulnessNumerator','HelpfulnessDenominator','Score','Summary','Text']
	)
# ignore noises
df = df[(df['HelpfulnessDenominator'] >= 5)]
# add the helpfulness column determined by the percentage of how many people think it is helpful
df["Helpfulness"] = (
	df["HelpfulnessNumerator"]/df["HelpfulnessDenominator"]).apply(lambda n: "Helpful" if n >= 0.8 else "Not_Helpful")
df["Sentiment"] = (df["Score"].apply(lambda score: "positive" if score > 3 else "negative"))
#task_data = df[(df['Score'] != 3)]
cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    if type(sentence) is str:
    	#clean up #https://www.kaggle.com/eugen1701/predicting-sentiment-and-helpfulness
	    sentence = sentence.lower()
	    sentence = cleanup_re.sub(' ', sentence).strip()
	    #remove stopwords and lemmatize
	    stopword_list = stopwords.words('english') + [u'p']
	    sentence_list = sentence.split()

	    sentence_no_stopword_list = sentence_list
	    lemmatizer = WordNetLemmatizer()
	    '''for word in sentence_list: # iterate over sentence_list #prob: might remove negation

	        if word in stopword_list:
	            print(word + " stopwords")
	            sentence_no_stopword_list.remove(word)
	    print("nosw: ") 
	    print(sentence_no_stopword_list)'''
	    sentence_lemma_list = []
	    for word in sentence_no_stopword_list:
	    	sentence_lemma_list.append(lemmatizer.lemmatize(word))
	    
	    spell = SpellChecker()
	    sentence_spell_list = []
	    #print("lemma: ")
	    #print(sentence_lemma_list)
	    for word in sentence_lemma_list:
	    	sentence_spell_list.append(spell.correction(word))
	    #print("spell: ")
	    #print(sentence_spell_list)
	    sentence = ' '.join(sentence_spell_list)
    return sentence

#df["Summary_Clean"] = df["Summary"].apply(cleanup)
df["Summary"] = df["Summary"].apply(cleanup)
df["Text"] = df["Text"].apply(cleanup)


task_data = df[(df['Score'] != 3)]
print(task_data.head(5))

task_data.to_csv('simplified_data.csv')


