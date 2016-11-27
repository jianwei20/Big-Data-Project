import sys
import csv
import re
import os

	f=open("file_name","r")
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
	output = open("FileName.csv","a")
	for time in time_list:
		#print("{}:{}".format(time,hash[time]))
		output.write("{}:{}\n".format(time,hash[time]))
	output.close()
	##
	i += 1