import sys
import featureExtraction as FE 
import dill as pickle


class BVsmVectorizer(object):

	def __init__(self, vocab):
		self.vocab  = dict()
		self.bvocab = dict()
		self.vectorizer = FE.CountVectorizer(min_df=1, fixed=False, vocabulary=vocab)
		self.transformer = FE.TfidfTransformer()


	def transform(self, corpus, vocab=None):
	    count = self.vectorizer.fit_transform(corpus)
	    self.transformer.fit(count)
	    TfIdf = self.transformer.transform(count).toarray()
	    print TfIdf.shape
	    self.vocab = self.vectorizer.vocabulary_
	    return TfIdf, self.vocab, self.bvocab


def main():
	argc = len(sys.argv)
	
	if argc < 2:
		print "Error: No input files"
		exit()
	data = pickle.load(open(sys.argv[1], "rb"))

	vocabulary = None
	if argc > 2:
		vocabulary = pickle.load(open(sys.argv[2], "rb"))

	corpus, topics, titles, time = data

	BV = BVsmVectorizer(vocabulary)
	N  = len(corpus)
	i  = 0
	while i*1000 < N:
		t, v, b = BV.transform(corpus[i*1000:(i+1)*1000], vocab=vocabulary)
		pickle.dump(t, open("inc.tfidf.pickle", "wb"))
		pickle.dump(v, open("inc.vocab.pickle", "wb"))
		vocabulary = v
		i += 1
		print i

	print v.__len__()

main()
