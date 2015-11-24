import math
import pickle
import numpy as np
from collections import defaultdict
import featureExtraction as FE

eps = 1. / (10 ** 25)
s = 1.2

def findBurstyWords(data, date):
	N = len(data)

	allDates = np.array(sorted(list(set(date))))
	D = len(set(date))

	cumulative = list()
	burstiness = dict()

	i = 0
	while i*100 < N:
		vectorizer = FE.CountVectorizer(min_df=1, fixed=False)
		dataVector = np.float64(vectorizer.fit_transform(data[i*100: (i+1)*100]).toarray())
		vectorizer.fixed = True

		dataVector = np.float64(vectorizer.transform(data).toarray())
		vocabulary = vectorizer.vocabulary_
		docFreqncy = FE._document_frequency(dataVector)

		W = dataVector.shape[1]
		dateWise = np.zeros((D, W), dtype=np.float64)
		dateDocs = dict()
		for d in xrange(D):
			dateDocs[d]  = (date == allDates[d])	# boolean of all docs on date allDates[d]
			dateWise[d] += (dataVector[dateDocs[d]] > 0).sum(axis=0)
						#  ^---------------------------------------^ number of docs on date allDates[d] which contained each word
						#  ^---------------------------^ boolean of all words (per doc) with > 0 occurrence
						#  ^----------------------^ dataVector of  all docs on date allDates[d] 
			dateDocs[d]  = float(dateDocs[d].sum())	# number of docs on date allDates[d]
			# dateWise[d] /= dateDocs[d]

		wordWise = dateWise.sum(axis=0)
		avgs = wordWise / N 	# probability of docs with each words over all dates
		avgs = np.array([avgs, avgs*s])

		def cost(d, j, T, w):

			if avgs[1][w] > 1.0:
				return [0]*d, 0, [0.], [0]

			if (d, j) in mem:
				return mem[(d, j)]

			if d == 0:
				return list(), 0, [0.], [0]

			def cost_(i, j, T):
				return ( j-i )*math.log(T) if ( j>=i ) else 0.

			def combination(n, r):
				f = math.factorial
				return f(n) / ( f(r) * f(n-r) )

			Q0, cost_x, L0, T0 = cost(d-1, 0, T, w)
			cost_0      = cost_x + cost_(0, j, T)

			Q1, cost_x, L1, T1 = cost(d-1, 1, T, w)
			cost_1      = cost_x + cost_(1, j, T)

			rel = dateWise[d][w]
			net = dateDocs[d]
			lfj = -math.log( combination(net, rel) )	\
			      -math.log( avgs[j][w]  + eps)*rel 	\
			      -math.log(1-avgs[j][w] + eps)*(net - rel)

			if cost_1 < cost_0 and d!=1 :
				Q1_ = Q1[:]; Q1_.append(1)
				L   = L1[:];
				T   = T1[:];
				improv = (cost_0 - cost_1)
				period = 1
				if L: 
					improv += L.pop()
					period += T.pop()
				L.append(improv)
				T.append(period)
				mem[(d, j)] = (Q1_, lfj + cost_1, L, T)
				return mem[(d, j)]
			else:
				Q0_ = Q0[:]; Q0_.append(0)
				L   = L0[:];
				T   = T0[:]
				if L and L[-1] != 0.: 
					L.append(0.)
					T.append(0)
				mem[(d, j)] = (Q0_, lfj + cost_0, L, T)
				return mem[(d, j)]

		newwords = set(vocabulary.keys()).difference(set(cumulative))
		print "NEW:\t", len(newwords)
		for k in newwords:
			# print
			# print "="*150
			# print
			w = vocabulary[k]
			mem = dict()
			Q, C, W, T = cost(D-1, 0, wordWise[w], w)
			if 1 in Q:
				burstiness[k] = max(W)
				burstiness[k]/= T[W.index(burstiness[k])]
				# print W
				# print Q
				# exit()
		cumulative += list(newwords)
		print "BURSTY:\t", len(burstiness.keys())
		i += 1
		print i

	print len(burstiness.keys()), "/", len(cumulative)

	return burstiness


def getBurstyWeight():
	data = pickle.load( open("reuters.pickle", "rb") )
	corpus = np.array(data[0])
	theday = np.array(data[3])
	vocabulary = dict()
	burstiness = dict()
	burstyWords = findBurstyWords(corpus, theday)
	print "Writing burstywords"
	pickle.dump(burstyWords, open("burstywords.pickle", "wb"))
	for word, weight in burstyWords.items():
		vocabulary[word] = vocabulary.__len__()
		burstiness[vocabulary[word]] = weight
	print "Writing burstiness"
	pickle.dump((vocabulary, burstiness), open("burstiness.pickle", "wb"))

	
getBurstyWeight()
