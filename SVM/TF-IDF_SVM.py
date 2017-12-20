#!/usr/bin/python
# -*- coding: ascii -*-
from sys import argv
from os.path import exists
import re
import os  
import math
import time
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from_file = "labeledmusic.txt"
to_file = "result.txt"

in_file = open(from_file,'r')
out_file = open(to_file,'w')
test_file = open("test.txt", 'w')

data = pd.read_csv("labeledmusic.txt", header = 0)
data.head()

lyrics = data['text']
labels = data['label']

vectorizer = TfidfVectorizer(min_df = 1)
vectors = vectorizer.fit_transform(lyrics).toarray()
tokennames = vectorizer.get_feature_names()

for i in range(900):
	if len(labels[i]) < 4:
		out_file.write("-1	")
	else:
		out_file.write("+1	")
	for j in range(len(vectors[i])):
		if vectors[i][j]>0:
			out_file.write(str(j+1) +":" + str(vectors[i][j]))
			out_file.write(" ")
	out_file.write("\n")

for i in range(901, len(vectors)):
	if len(labels[i]) < 4:
		test_file.write("-1	")
	else:
		test_file.write("+1	")
	for j in range(len(vectors[i])):
		if vectors[i][j]>0:
			test_file.write(str(j+1) +":" + str(vectors[i][j]))
			test_file.write(" ")
	test_file.write("\n")

in_file.close()
out_file.close()
test_file.close()