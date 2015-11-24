import math

# a0 = 8./88.
# a1 = 9.6/88.
# X = [None, 20, 20, 2, 2, 2, 2, 20, 20]

# X = [None, 1, 1, 1, 1, 1, 1, 1, 1]

a0 = 10./64.
a1 = 12./64.
X = [None, 20, 6, 3, 2, 1, 1, 2, 3, 6, 20]

a = [a0, a1]

def type1():

	def cost_(i, j, t):
		tau = 0
		if j >= i:
			tau = (j - i) * math.log(t)
		return tau

	def cost(t, j, n):
		if t == 0:
			return 0, list()
		Q = None
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
		return lfj + mincost, Q

	def getCost(t):
		C0, Q0 = cost(t, 0, t); Q0.append(0)
		C1, Q1 = cost(t, 1, t); Q1.append(1)
		if C1 < C0:
			return Q1, C1
		else:
			return Q0, C0

	for i in xrange(len(X)):
		Q, C = getCost(i)
		print Q
		# print C


def type2():


	def cost(d, j, D, w):
		if avgs[1][w] > 1.0:
			print "X"
			return [0]*d, 0, 0

		if (d, j) in mem:
			return mem[(d, j)]

		if d == 0:
			return list(), 0, 0

		def cost_(i, j, t):
			return ( j-i )*math.log(t) if ( j>=i ) else 0.

		def combination(n, r):
			f = math.factorial
			return f(n) / ( f(r) * f(n-r) )

		Q, cost_x, I0 = cost(d-1, 0, D, w)
		cost_0      = cost_x + cost_(0, j, D)

		Q, cost_x, I1 = cost(d-1, 1, D, w)
		cost_1      = cost_x + cost_(1, j, D)

		rel = dateWise[d][w]
		net = dateDocs[d]
		lfj = -math.log( combination(net, rel) )	\
		      -math.log( avgs[j][w]  + eps)*rel 	\
		      -math.log(1-avgs[j][w] + eps)*(net - rel)

		if cost_1 < cost_0:
			improvemt = I1 + (cost_0 - cost_1)
			Q.append(1)
			mem[(d, j)] = (Q, lfj + cost_1, improvemt)
			return mem[(d, j)]
		else:
			Q.append(0)
			mem[(d, j)] = (Q, lfj + cost_0, I0)
			return mem[(d, j)]

