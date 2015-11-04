import math

# a0 = 8./88.
# a1 = 9.6/88.
# X = [None, 20, 20, 2, 2, 2, 2, 20, 20]

# X = [None, 1, 1, 1, 1, 1, 1, 1, 1]

a0 = 10./64.
a1 = 12./64.
X = [None, 20, 6, 3, 2, 1, 1, 2, 3, 6, 20]

a = [a0, a1]

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
