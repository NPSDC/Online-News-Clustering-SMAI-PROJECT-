import sys
import featureExtraction as FE 
import dill as pickle
import math
import scipy as sp
import numpy as np

s = 1.2
f = math.factorial
eps = 1. / (10 ** 9)

class BVsmVectorizer(object):

	def __init__(self, vocab):
		self.vocab  = dict()
		self.bvocab = dict()
		self.vectorizer = FE.CountVectorizer(min_df=1, fixed=False, vocabulary=vocab)
		self.transformer = FE.TfidfTransformer()

		self.wordWise = np.array(list())
		self.W = 0
		self.N = 0
		self.bwords = list()

	def getBurstyWords(self, count, vocab):
		N  = count.shape[0]         	# number of documents on this day
		W  = count.shape[1]         	# number of words on this day
		df = (count > 0.).sum(axis=0)	# number of documents containing each word

		newwords = W - self.W
		self.wordWise  = np.lib.pad(self.wordWise, pad_width=(0, newwords), \
			                        mode='constant', constant_values=0 )
		self.wordWise += df
		averages = self.wordWise / (self.N + N)      	# averages
		averages = [ averages, averages*s ]

		def costImprovement(i):
			def cost(q):
				return -math.log( averages[q][i]  + eps)*df[i] 	\
			           -math.log(1-averages[q][i] + eps)*(N - df[i])
			return cost(0) - cost(1)

		bwords = list()
		for w, i in vocab.items():
			if averages[1][i] >= 1. or i >= self.W or self.wordWise[i] == 0.:
				self.bvocab[i] = [0., 0.]
				continue
			improvmt = costImprovement(i)
			net_cost = improvmt + self.bvocab[i][0]
			if net_cost - math.log(self.wordWise[i]) > 0. and improvmt > 0.:
				bwords.append((str(w), i))
				if self.bvocab[i][0] > 0.:
					self.bvocab[i][0] = net_cost
				elif net_cost > 0.:
					self.bvocab[i][0] = net_cost
				else:
					self.bvocab[i] = [0., 0.]
			else:
				# if net_cost > 0.:
				# 	self.bvocab[i][0] = net_cost
				# else:
				self.bvocab[i][0]  = 0.

		print "bursty : ", len(bwords)
		self.N += N
		self.W += newwords
		self.bwords += bwords



	def transform(self, count):
	    self.transformer.fit(count)
	    TfIdf = self.transformer.transform(count).toarray()
	    return TfIdf

	def vectorize(self, corpus, vocab=None):
		count = self.vectorizer.fit_transform(corpus)
		self.vocab = self.vectorizer.vocabulary_
		t = self.transform(count)
		self.getBurstyWords(np.float64(count.toarray(), dtype=np.float64), self.vocab)
		return t, self.vocab, self.bvocab


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
	time = np.array(time)
	corpus = np.array(corpus)
	allDates = sorted(list(set(time)))
	# while  i*100 < 1000:
	for d in allDates:
		# c = [ i == d for i in time ]
		# C = list()
		# for i in xrange(len(time)):
		# 	if time[i] == d:
		# 		C.append(corpus[i])
		t, v, b = BV.vectorize(corpus[time == d], vocab=vocabulary)
		pickle.dump(t, open("inc.tfidf.pickle", "wb"))
		pickle.dump(v, open("inc.vocab.pickle", "wb"))
		vocabulary = v
		i += 1
		print i

	print len(set(BV.bwords))
	print type(BV.bwords[0])

	print v.__len__()

main()
