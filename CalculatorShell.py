import sys
import csv
# import re
import os
import glob

files=glob.glob("data*.txt")

for file in files:
	f=open(file,"r")
	#print(f)
	##
	number=f.read() #print(type(number))=>string
	hours=number.split() # print(type(hour))=>list
	##
	hash = {}
	for time in hours:
		if time in hash:
			hash[time]+=1
		else:
			hash[time]=1
	##		
	time_list=list(hash.keys())
	time_list.sort()
	##
	OutputFileName=file+".csv"
	output = open(OutputFileName,"a")
	for time in time_list:
		#print("{}:{}".format(time,hash[time]))
		output.write("{}:{}\n".format(time,hash[time]))
	output.close()
	##
