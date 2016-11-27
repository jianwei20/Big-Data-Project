#!/usr/bin/python
import sys
import csv
import re
import os
import itertools

def FindSubsets(S,m):
    #return set(itertools.combinations(S,m))
    return list(itertools.combinations(S,m))

# def ComposeFeature(Feature_supersets): #( ) or input resulf 



f = open('train10.csv', 'r')
rows = csv.reader(f)
Feature_supersets = {} #print(type) #dict
for features in rows:
	#print(features)  
	#print(type(features)) #<type 'list'>
	for n_feature in range(len(features)):
		#set(features)
		result=FindSubsets(features,n_feature) #n_feature+1
		#print(result)
		Feature_supersets[n_feature]=result #n_feature+1
	break
# print(Feature_supersets[n]) # n elements factorial supersets
# print(Feature_supersets[n][m]) # the m_th order subset of n elements factorial supersets
# print(Feature_supersets[n][m][p]) # th p_th element in subset
f.close()
#print(Feature_supersets)
#print(Feature_supersets[2][1])
#print(type(Feature_supersets[0][0][0]))


#====================================================================
#subfunction:
FeatureString=""
for i in Feature_supersets:
	for j in Feature_supersets[i]:
		if (len(j) == 0) : continue
		else : string='"label ~ '
		for k in j:
			string=string+k+' + '  ##using subfunction  #remove click
			FeatureString=string
		FeatureString=FeatureString[:-3]  #remove +
		FeatureString=FeatureString+'"'  #add ""
		print(FeatureString)
#====================================================================
#====================================================================
# import itertools

# def findsubsets(S,m):
#     return set(itertools.combinations(S,m))

# S = [1, 2, 3, 4, 6]
# for m in range(len(S)):
# 	result=findsubsets(S,m)
# 	print(result)
#====================================================================