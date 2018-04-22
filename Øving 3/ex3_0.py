import random


class treeNode:
	children = []*2
	attribute = -1
	label = -1

	def __init__(self,attribute):
		self.attribute = attribute


def importanceRandom(attributes): #Returns the most important attribute randomly.
		n = len(attributes)
		i = random.randint(0,n-1)
		return attributes[i]

def getData(fileName): #Returns the data in the file as an array of arrays
		dataFile = open(fileName,'r')
		data = []
		for line in dataFile:
			actualLine = []
			for letter in line:
				if letter != '\t' and letter != '\n':
					actualLine.append(int(letter))

			data.append(actualLine)
		dataFile.close()
		return data

def pluralityValue(examples): #Works only for two atributes
		element1 = 0
		element2 = 0
		for element in examples:
			if element[7] == 1:
				element1 += 1
			elif element[7] == 2:
				element2 += 1
		return max(element1,element2)

def checkSameClassification(examples):
		prev = -1
		for example in examples:
			if prev<0:
				prev = example[7]
			elif example[7] != prev:
				return 0
		return prev

def printTree(node):
	print(node.label)
	for child in node.children:
		if child.label >= 0 and child.label <= 7:
			printTree(child)

def decisionTreeLearning(examples,attributes,parentExamples):
	print('Start new run \n')
	tree = treeNode(-1)
	if examples == []:
		print('Empyt exampkasdf \n')
		tree.attribute = pluralityValue(parentExamples)
		return tree
	elif checkSameClassification(examples):
		print('Same class\n')
		tree.attribute = examples[0][7]
		return tree
	elif sum(attributes) < 1:
		print('Not enough attribues')
		tree.attribute = pluralityValue(examples)
		return tree

	else:
		A = importanceRandom(attributes)
		attributes.remove(A)
		tree.attribute = A
		for v in [1,2]:
			exs = []
			for example in examples:
				if example[A] == v:
					exs.append(example)
			subTree = decisionTreeLearning(exs,attributes,examples)
			subTree.label = A
			tree.children.append(subTree)
		return tree



print('')
print('')
examples = getData("training.txt")
testData = getData("test.txt")

n = len(examples[0])
startAttributes = []
for i in range(n):
    startAttributes.append(i)

theBestTree = decisionTreeLearning(examples,startAttributes,[])
print('')
print('')
print('')
print('')
#print('Now we print!')
printTree(theBestTree)




#
# def learn(examples,attributes,parentExamples):
# 		if !len(examples):
# 			return pluralityValue(parentExamples)
# 		elif checkSameClassification(examples):
# 			return examples[0][8]
# 		elif sum(attributes) == 0:
# 			return pluralityValue(examples)
#
# 		else:
# 			A = importanceRandom(attributes)
#
# 			for value in range(1:3):
# 				exs = []
# 				for example in examples:
# 					if example[A] == value:
# 						exs.append(example)
# 				subNode = treeNode = learn(exs,attributes)
#
#
#
# 			for example in examples:
# 				if example[]
# 			newTreeNode = treeNode()
#
# 			subtreeNode1 = treeNode()
# 			subtreeNode1.parent =
#
#
#
#
# 			subtreeNode2 = treeNode()
