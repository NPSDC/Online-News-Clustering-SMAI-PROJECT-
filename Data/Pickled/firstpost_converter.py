import pickle
import sklearn.feature_extraction.text as TFE
import os
from reuters_converter import get_months

def main():	
	main_dir = os.path.join(os.environ['HOME'],'News/2014')
	directories = os.listdir(main_dir) 
	corpus = list()
	titles = list()
	topics = list()
	tot_days = list()
	start_day = 1
	cumulative_days, Months_List = get_months()
	for month in directories:
	 	month_days = cumulative_days[int(month) - 1] - 31
	 	days = os.listdir(os.path.join(main_dir, month))
	  	for day in days: 
	  		total_days = int(day) + month_days - start_day + 1
	 		files = os.listdir(os.path.join(main_dir, month, day))
	 		for file in files:
	 			if(file[-1] != 'e'):
	 				f = open(os.path.join(main_dir, month, day, file), 'rb')
	 				data = pickle.load(f)
	 				f.close()
 					tags = data[2]
 					for k in xrange(len(tags)):
 						corpus.append(data[1])
 						titles.append(data[0])
 						topics.append(tags[k])
 						tot_days.append(total_days)

	print len(corpus)
	pickle.dump((corpus, topics, titles, tot_days) , open("First_Post_2014.pickle", "wb"))

main()