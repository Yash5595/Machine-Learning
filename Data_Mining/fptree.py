import math
import numpy as np
import time
from itertools import chain, combinations


class Tree(object):
	def __init__(self, value, count=1, next_node=None, children=None, parent=None):
		self.value = value
		self.count = count
		self.children = []
		self.parent=None
		if children is not None:
			for child in children:
				self.add_child(child)
		if next_node is not None:
			assert isinstance(next_node, Tree)
		self.next_node = next_node
	def __repr__(self):
		return str(self.value)
	def add_child(self, node):
		assert isinstance(node, Tree)
		self.children.append(node)
	def add_next_node(self, node):
		assert isinstance(node, Tree)
		self.next_node = node
	def inc_count(self):
		self.count += 1
	def has_children(self):
		return not not self.children
	def add_parent(self,node):
		assert isinstance(node, Tree)
		self.parent = node

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


def create_FP_tree(ordered_data, item_list, pointer_list):
	root = Tree('', math.inf)
	for transaction in ordered_data:
		add_transaction(ordered_data, item_list, transaction, root,pointer_list)
	return root

def add_transaction(ordered_data, item_list, transaction, root, pointer_list):
	for item in transaction:
		is_child = False
		for node in root.children:
			if node.value == item:
				node.inc_count()
				root=node
				is_child = True
				break
		if not is_child:
			new_node = Tree(item)
			root.add_child(new_node)
			new_node.add_parent(root)
			pointer_node = pointer_list[item_list.index(item)]
			if pointer_node is None:
				pointer_list[item_list.index(item)] = new_node
			else:
				while pointer_node.next_node is not None:
					pointer_node = pointer_node.next_node
				pointer_node.add_next_node(new_node)
			root=new_node

def create_conditional_FP_tree(node, item_list):
	conditional_pointer_list = [None]*len(item_list)
	data = []
	while node is not None:
		transaction = []
		temp = node.parent
		while temp.value is not '':
			transaction = [temp.value] + transaction
			temp = temp.parent
		#print('cc_fp, tr:',transaction)
		for i in range(node.count):
			if len(transaction) > 0:
				data.append(transaction)
		node = node.next_node
	#print('cc_fp, data:',data)
	conditional_tree = create_FP_tree(data, item_list, conditional_pointer_list)
	#print('cc_fp, cpl: ',conditional_pointer_list)
	return [conditional_tree, conditional_pointer_list]

def fp_growth(tree, pointer_list, min_sup, item_list):
	patterns = []
	if tree.has_children():
		for node in reversed(pointer_list):
			if node is not None:
				current_count = 0
				temp=node
				while temp is not None:
					current_count += temp.count
					temp = temp.next_node
				if current_count >= min_sup:
					li = create_conditional_FP_tree(node, item_list)
					#print('creating tree for ',node.value)
					conditional_tree = li[0]
					conditional_pointer_list = li[1]
					#print('fp_gr, cpl: ',conditional_pointer_list)
					freq_patterns_base = fp_growth(conditional_tree, conditional_pointer_list, min_sup, item_list)
					#print('pattern base: ',freq_pattern_base)
					for pattern in freq_patterns_base:
						#patterns.append(pattern+[node.value])
						pattern.add_item(node.value)
						patterns += [pattern]
						#print('updated pattern ',pattern)

					#patterns.append([node.value])
					new_pattern = Pattern([node.value],current_count)
					#print('created new pattern ',new_pattern)
					patterns += [new_pattern]
					#print(patterns)
	return patterns

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



def main(min_sup, min_conf):
	data = np.genfromtxt('chess.dat', int)
	#data = [[7,8,9,6],[8,9,6],[9,6,5],[6],[6,5]]
	#data = [['c','f','a','m','p'],['c','f','a','b','m'],['f','b'],['c','b','p'],['c','f','a','m','p']]
	#data = [['bread','milk'],['bread','diaper','beer','eggs'],['milk','diaper','beer','coke'],['milk','diaper','beer','bread'],['milk','diaper','coke','bread']]
	item_list = list(set().union(*data))

	item_count = [0]*(len(item_list))


	for transaction in data:
		for item in transaction:
			item_count[item_list.index(item)] += 1

	item_list = [x for (y,x) in sorted(zip(item_count,item_list), reverse=True)]
	item_count.sort(reverse=True)


	for i in range(len(item_count)):
		if item_count[i] < min_sup:
			item_count = item_count[0:i]
			item_list = item_list[0:i]
			break


	ordered_data = []

	for transaction in data:
		ordered_data.append(sorted(set(transaction) & set(item_list), key = item_list.index))

	#print(item_list)
	#print(item_count)
	#print(ordered_data)

	pointer_list = [None]*len(item_count)
	#print(pointer_list)

	fp_tree = create_FP_tree(ordered_data, item_list, pointer_list)
	#patterns = list(set(fp_growth(fp_tree, pointer_list, min_sup, item_list)))
	patterns = fp_growth(fp_tree, pointer_list, min_sup, item_list)
	print('frequent patterns: ', len(patterns))
	for p in patterns[0:20]:
		print(p)
	print('------------------')


	rules = find_rules(patterns, min_conf)

	print('rules : ', len(rules))
	for r in rules[0:20]:
		print(r[0],' -> ',r[1],', confidence: ',r[2])
	print('------------------')
	return[patterns,rules]



###########################################################




start_time = time.time()

[patterns,rules] = main(2600,0.2)


print("--- %s seconds ---" % (time.time() - start_time))