import pickle
import sklearn.feature_extraction.text as TFE

def main():
	data = pickle.load(open("reuters_raw.pickle", "rb"))
	corpus = list()
	titles = list()
	topics = list()
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
	print len(corpus)
	pickle.dump((corpus, topics, titles) , open("reuters.pickle", "wb"))

main()