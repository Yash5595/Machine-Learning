import math
import numpy as np
import time
from itertools import chain, combinations


class Pattern(object):
	def __init__(self, items=[], support=0):
		self.items = items
		self.support = support
	def __repr__(self):
		return (str(self.items)+', '+str(self.support))
	def add_item(self,item):
		self.items.append(item)
	def update_support(self, sup):
		self.support = sup


def find_rules(patterns, min_conf):
	pat = [set(p.items) for p in patterns]
	sup = [p.support for p in patterns]

	rules = []

	for pattern in pat:
		left_list = []
		for s1 in chain.from_iterable(combinations(pattern, r) for r in reversed(range(1,len(pattern)))):
			left_list.append(set(s1))
		for left in left_list:
			if left not in pat:
				for s2 in chain.from_iterable(combinations(left, r) for r in reversed(range(1,len(left)+1))):
					if s2 in left_list:
						left_list.remove(s2)
			else:
				conf = sup[pat.index(pattern)]/sup[pat.index(left)]
				if conf>=min_conf:
					rules.append([left,pattern-left,conf])
				else:
					for s2 in chain.from_iterable(combinations(left, r) for r in reversed(range(1,len(left)+1))):
						if s2 in left_list:
							left_list.remove(s2)
	return rules


def create_file(patterns, rules):
	file_output = open('apriori_output','w')
	file_output.write('frequent patterns: '+str(len(patterns)))
	file_output.write('\n')
	file_output.write('\n')
	for p in patterns:
		file_output.write(str(p))
		file_output.write('\n')
	file_output.write('-----------------------------')
	file_output.write('\n')
	file_output.write('rules: '+str(len(rules)))
	file_output.write('\n')
	file_output.write('\n')
	for r in rules:
		rule = str(r[0])+' -> '+str(r[1])+', confidence: '+str(r[2])
		file_output.write(rule)
		file_output.write('\n')

def display(patterns, rules, k):
	print('frequent patterns: ', len(patterns))
	for p in patterns[0:k]:
		print(p)
	print('------------------')
	print('rules : ', len(rules))
	for r in rules[0:k]:
		print(r[0],' -> ',r[1],', confidence: ',r[2])
	print('------------------')





def find_count(candidates, data):
	count = [0]*len(candidates)
	for i in range(len(candidates)):
		candidate = candidates[i]
		for transaction in data:
			if candidate.issubset(set(transaction)):
				count[i] += 1
	return count

def generate_candidates(items, k):
	li = []
	for c in combinations(items, k):
		li.append(set(c))
	return li

def remove_low_conf(candidates, count, min_sup):
	candidates_new = []
	count_new = []
	for i in range(len(candidates)):
		if count[i] >= min_sup:
			candidates_new.append(candidates[i])
			count_new.append(count[i])
	return [candidates_new, count_new]

def prune(C_new, F_old, k):
	f_set = []
	for f in F_old:
		f_set.append(frozenset(f))
	for c in C_new:
		c_set = []
		for t in combinations(c,k-1):
			c_set.append(frozenset(t))
		if not set(c_set).issubet(set(f_set)):
			C_new.remove(c)
	return C_new

def main(f_name, min_sup, min_conf):
	data = np.genfromtxt(f_name, int)
	#data = [[7,8,9,6],[8,9,6],[9,6,5],[6],[6,5]]
	#data = [['c','f','a','m','p'],['c','f','a','b','m'],['f','b'],['c','b','p'],['c','f','a','m','p']]
	#data = [['bread','milk'],['bread','diaper','beer','eggs'],['milk','diaper','beer','coke'],['milk','diaper','beer','bread'],['milk','diaper','coke','bread']]

	patterns = []

	#print(data)

	C_set = generate_candidates(list(set().union(*data)), 1)
	C_count = find_count(C_set, data)
	[F_set,F_count] = remove_low_conf(C_set, C_count, min_sup)


	for i in range(len(F_set)):
		pat = Pattern(list(F_set[i]), F_count[i])
		patterns.append(pat)

	k = 2
	while F_set:
		C_set = generate_candidates(list(set().union(*F_set)), k)
		C_count = find_count(C_set, data)
		[F_set,F_count] = remove_low_conf(C_set, C_count, min_sup)

		for i in range(len(F_set)):
			pat = Pattern(list(F_set[i]), F_count[i])
			patterns.append(pat)

		k += 1


	rules = find_rules(patterns, min_conf)

	return[patterns,rules]



###########################################################





u_file = input('enter file name: ')
u_min_sup = int(input('enter minimum support: '))
u_min_conf = float(input('enter minimum confidence: '))

start_time = time.time()
[patterns,rules] = main(u_file,u_min_sup,u_min_conf)
print("--- %s seconds ---" % (time.time() - start_time))


create_file(patterns, rules)
display(patterns, rules, 5)
