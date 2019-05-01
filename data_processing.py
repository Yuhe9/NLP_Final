# Note: pin install pandas

#!/usr/bin/python
#pip install pyspellchecker
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import matplotlib as mpl

#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from spellchecker import SpellChecker
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
from PIL import Image

stopwords = set(STOPWORDS)

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

cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    if type(sentence) is str:
    	#clean up #https://www.kaggle.com/eugen1701/predicting-sentiment-and-helpfulness
	    sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
	    #sentence = cleanup_re.sub(' ', sentence).strip()
	    #stopword_list = stopwords.words('english') + [u'p']

	    sentence_list = sentence.split(' ')

	    lemmatizer = WordNetLemmatizer()
	    spell = SpellChecker()
	    sentence_clean_list = []
	    for word in sentence_list:
	        #if word in stopword_list:
	            #print(word + " stopwords")
	        sentence_clean_list.append(spell.correction(lemmatizer.lemmatize(word)))

	    sentence = ' '.join(sentence_clean_list)
	    #print(sentence)
	    #print(sentence)
    return sentence

#df["Summary_Clean"] = df["Summary"].apply(cleanup)
#df["Summary"] = df["Summary"].apply(cleanup)
#df["Text"] = df["Text"].apply(cleanup)

task_data = df[(df['Score'] != 3)]
print(task_data.head(5))

task_data.to_csv('simplified_data.csv')

#mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 



def show_wordcloud(data, title = None):
	#pic ref: https://geekestateblog.com/smart-home-amazon-alexa-echo/
    amazon_coloring = np.array(Image.open("amazon_smile.png"))

    wc = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        mask = amazon_coloring,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    )
    wc.generate(str(data))
    image_colors = ImageColorGenerator(amazon_coloring)
    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")

    plt.show()

    
# 
show_wordcloud(df[df.Score == 5]["Summary"]) #positive summary
show_wordcloud(df[df.Score == 1]["Summary"]) #negative summary

show_wordcloud(df[df.Helpfulness == "Helpful"]["Text"]) #helpful review
show_wordcloud(df[df.Helpfulness == "Not_Helpful"]["Text"]) #not helpful review




