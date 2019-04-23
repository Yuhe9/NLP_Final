# Note: pin install pandas

#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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
task_data = df[(df['Score'] != 3)]
print(task_data.head(5))
task_data.to_csv('simplified_data.csv')


