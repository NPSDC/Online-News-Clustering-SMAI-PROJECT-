import dill as pickle
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
	doc_grp_id = list()
	start_day = 1
	id = 1
	cumulative_days, Months, Months_List = get_months()
	directories = sorted(directories)
	directories[1] = '10'
	directories = sorted(directories)
	directories[9] = '010'
	for month in directories:
	 	month_days = cumulative_days[int(month) - 1] - Months[Months_List[int(month) - 1]]
	 	days = os.listdir(os.path.join(main_dir, month))
	 	days = sorted(days)
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
 						doc_grp_id.append(id)
 					id += 1

	print tot_days[-1], doc_grp_id[-1]
	pickle.dump((corpus, topics, titles, tot_days, doc_grp_id) , open("First_Post_2014.pickle", "wb"))

if __name__ == '__main__':
	main()