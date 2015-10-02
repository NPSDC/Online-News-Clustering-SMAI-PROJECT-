import pickle
import sklearn.feature_extraction.text as TFE

vectorizer = TFE.CountVectorizer(min_df=1)
transformer = TFE.TfidfTransformer()

data = pickle.load(open("reuters.pickle", "r"))
corpus = list()
titles = list()
for article in data:
	if 'body' in article:
		corpus.append(article['body'])
		titles.append(article['title'])

counts = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(counts)

pickle.dump(tfidf, open("tfidf.pickle", "w"))