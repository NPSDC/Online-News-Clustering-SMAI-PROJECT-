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
		self.bursts = dict()
		self.vectorizer = FE.CountVectorizer(min_df=1, max_df=0.95, stop_words=FE.ENGLISH_STOP_WORDS, fixed=False, vocabulary=vocab)
		self.transformer = FE.TfidfTransformer()

		self.wordWise = np.array(list())
		self.W = 0
		self.N = 0

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

		for w, i in vocab.items():
			if not w in self.bursts:
				self.bursts[i] = 0.

			if averages[1][i] >= 1. or i >= self.W or self.wordWise[i] == 0.:
				self.bvocab[i] = [0., False]
				continue
			improvmt = costImprovement(i)
			net_cost = improvmt + self.bvocab[i][0]
			if net_cost - math.log(self.wordWise[i]) > 0. and improvmt > 0.:
				self.bvocab[i][0] = net_cost
				self.bvocab[i][1] = True
				if self.bvocab[i][0] > self.bursts[i]:
					self.bursts[i] = self.bvocab[i][0]
			else:
				if net_cost > 0.:
					self.bvocab[i][0] = net_cost
				else:
					self.bvocab[i]    = [0., False]

		self.N += N
		self.W += newwords

	def tfidf(self, count):
	    self.transformer.fit(count)
	    TfIdf = self.transformer.transform(count).toarray()
	    return TfIdf

	def bness(self):
		print len(self.vocab), self.W
		bWeights = np.array([ (self.bvocab[i][0] / self.wordWise[i] if self.bvocab[i][1] else 0.) for i in xrange(self.W) ])
		return bWeights

	def sab(self, t, b):
		SABV = b + t
		SABV/= np.linalg.norm(SABV)
		return SABV

	def smb(self, t, b):
		SMBV = b * t
		SMBV[:, b>0] /= np.linalg.norm(SMBV)
		return SMBV

	def vectorize(self, corpus, vocab=None):
		count = self.vectorizer.fit_transform(corpus)
		self.vocab = self.vectorizer.vocabulary_
		self.getBurstyWords(np.float64(count.toarray(), dtype=np.float64), self.vocab)
		t   = self.tfidf(count)
		b   = self.bness()
		sab = self.sab(t, b)
		smb = self.smb(t, b)
		return t, b, sab, smb


def main():
	argc = len(sys.argv)
	
	if argc < 2:
		print "Error: No input files"
		exit()
	data = pickle.load(open(sys.argv[1], "rb"))

	vocabulary = None
	grpid = None
	corpus, topics, titles, time = data

	BV = BVsmVectorizer(vocabulary)
	# N  = len(corpus)
	i  = 0

	time   = np.array(time)
	corpus = np.array(corpus)
	topics = np.array(topics)
	titles = np.array(titles)
	grpid  = np.array(grpid)
	allDates = sorted(list(set(time)))

	for d in allDates:

		try:
			t, b, sab, smb = BV.vectorize(corpus[time == d], vocab=vocabulary)
			vocabulary = BV.vocab
		except KeyboardInterrupt:
			print "Could Not Vectorise", d
			exit()

		try:
			pickle.dump( ((t, b, sab, smb), 
				          topics[time == d], titles[time == d], 
				            time[time == d], grpid[time == d]) ,
				              open("vectors/"+str(i), "wb") )

			pickle.dump(BV, open("vectors/vectorizer."+str(i%2), "wb"))
		except KeyboardInterrupt:
			print "Could Not Pickle", d
			exit()

		i += 1
		print i, "\n"

	print len(set(BV.bwords))

main()
