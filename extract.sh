#!/bin/bash
i=1
while [ $i -le 24 ]
do
 	cut -d ',' -f$i ~/Downloads/Click_Through_Rate_Prediction/train.csv > data$i.txt
	mv data$i.txt ~/Downloads/Click_Through_Rate_Prediction/DataExtracting 	
	i=`expr $i + 1`
done
