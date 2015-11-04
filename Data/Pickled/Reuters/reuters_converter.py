import pickle
import sklearn.feature_extraction.text as TFE


def get_months():
	cumulative_months = []
	Months = {
		'JAN' : 31,
		'FEB' : 28,
		'MAR' : 31,
		'APR' : 30,
		'MAY' : 31,
		'JUN' : 30,
		'JUL' : 31,
		'AUG' : 31,
		'SEP' : 30,
		'OCT' : 31,
		'NOV' : 30,
		'DEC' : 31
	}
	Months_lists = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
	temp = 0
	for i in xrange(len(Months_lists)):
		cumulative_months.append(Months[Months_lists[i]] + temp)
		temp = cumulative_months[i]

	return (cumulative_months, Months_lists)

def count_days(date, cumulative_months, Months_lists):
	month = date[1]
	try:		
		index = Months_lists.index(month)
	except ValueError:
		 print month+" Not Valid"
		 raise ValueError

	days = int(date[0]) + cumulative_months[index - 1]
	return days

def date_convert(date, start_date, cumulative_months, Months_lists):
	if(start_date == date):
		count = 1
		
	days_current = count_days(date, cumulative_months, Months_lists)
	days_start = count_days(start_date, cumulative_months, Months_lists)
	count =  days_current - days_start + 1
	return count

def main():
	data = pickle.load(open("reuters_raw.pickle", "rb"))
	corpus = list()
	titles = list()
	topics = list()
	days = list()
	start_date = data[0][0]['date'][0].split('-')[:-1]
	cumulative_months, Months_lists = get_months()
	for article in data:
		if 'text' in article[0] and 'topics' in article[0]:
			topic = article[0]['topics']
			if len(topic) != 1:
				continue
			for piece in article[0]['text']:
				if isinstance(piece, dict) and 'body' in piece:
					# print "hell"
					titles.append(piece['title'][0])
					corpus.append(piece['body'][0])
					topics.append(topic[0]['d'][0])
					date = article[0]['date'][0].split('-')[:-1]
					days.append(date_convert(date, start_date, cumulative_months, Months_lists))
	print len(corpus)
	pickle.dump((corpus, topics, titles, days) , open("reuters.pickle", "wb"))

main()

