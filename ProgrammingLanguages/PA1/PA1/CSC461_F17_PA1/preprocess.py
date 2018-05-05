def normalize( rows ):
	cols = list(zip(*rows))
	for i,c in enumerate(cols[2:],2):
		m,M = min(c), max(c)
		cols[i] = [ (x-m)/(M-m) for x in c ]
	return list(zip(*cols))
	

