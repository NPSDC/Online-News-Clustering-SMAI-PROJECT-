import math
import pickle
from collections import defaultdict
import sklearn.feature_extraction.text as TFE

s = 1.2

def findBurstyWords(topic):
	data, _, date = topic

	vectorizer = TFE.CountVectorizer(min_df=1)
	dataVector = np.array(vectorizer(data), dtype=np.float64)
	vocabulary = vectorizer.vocabulary_

	N = data.shape[0]
	W = data.shape[1]

	avgs = dataVector.sum(axis=0) / N
	freq = dataVector > 0.
	

	def cost_(i, j, t):
		return (j-i)*math.log(t) if ( j>=i ) else 0.

	def cost(t, j, n, w):
		if t == 0:
			return 0, list()

		mincost = None
		min_q_i = None

		for i in [0, 1]:
			lfi, Q = cost(t-1, i, n)
			cost_i = lfi + cost_(i, j, n)
			if not mincost:
				mincost = cost_i
				min_q_i = i
			elif mincost > cost_i:
				mincost = cost_i
				min_q_i = i
			if t == 1:
				break
		Q.append(min_q_i)
		lfj = a[j]*X[t] - math.log(a[j])
		Q.append( {'gap': X[t]} )
		return Q, lfj + mincost

	for n in xrange(N):
		for w in xrange(W):
			Q, C = getCost(T, 0, T, w)
			print Q
			exit()


def main():
	topics = pickle.load( open("burstytopics.pickle", "rb") )
	vocabulary = dict()
	burstiness = dict()
	for topic in topics:
		burstyWords = findBurstyWords(topic)
		for word, weight in burstyWords:
			if not word in vocabulary:
				vocabulary[word] = vocabulary.__len__()
				burstiness[vocabulary[word]] = weight
			elif weight > burstiness[vocabulary[word]]:
				burstiness[vocabulary[word]] = weight
