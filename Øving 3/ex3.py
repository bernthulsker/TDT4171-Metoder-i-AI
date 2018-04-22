import random
import math as maths

class treeNode:
    testAttribute = -1 #Which attribue, 0-6, are we testing on here
    label = -1 #Which classification you end up in. I.e example[-1] which is either 1 or 2

    def __init__(self,testAttribute,label):
        self.label =label
        self.testAttribute = testAttribute
        self.children = [None] *2

def pluralityValue(examples): #Returns the most commom label. Favours 2 FIX IF TIME
    noOfElement1 = 0
    noOfElement2 = 0

    for example in examples:
        if example[-1] == 1:
            noOfElement1 += 1
        elif example[-1] == 2:
            noOfElement2 += 2
    if noOfElement1 > noOfElement2:
        return 1
    elif noOfElement2>= noOfElement1:
        return 2
    return 0

def checkSameClassification(examples):
		prev = -1
		for example in examples:
			if prev<0:
				prev = example[-1]
			elif example[-1] != prev:
				return 0
		return prev #Returns the classification if there is only one

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

def H(p):
    if p >= 1 or p <= 0:
        return 0
    else :
        return -(p*maths.log(p,2) + (1-p)*maths.log((1-p),2))

def gain(examples,attribute):
    S1 = []
    S2  = []
    for example in examples:
        if example[attribute] == 1:
            S1.append(example)
        elif example[attribute] == 2:
            S2.append(example)

    x11 = 0
    x12 = 0
    for example in S1:
        if example[-1] == 1:
            x11 += 1
        elif example[-1] == 2:
            x12 += 2

    x21 = 0
    x22 = 0
    for example in S2:
        if example[-1] == 1:
            x21 += 1
        elif example[-1] == 2:
            x22 += 2

    remainder = len(S1)/len(examples) * H(x11/(x12+x11)) + len(S2)/len(examples) * H(x21/(x21+x22))
    #remainder = len(S1)/len(examples) * H(x11/(x12+x11)) + len(S2)/len(examples) * H(x21/(x21+x22))


    return H(len(S1)/len(S2)) - remainder

def importance(attributes,entropy,examples): #Returns the most important. If entropy, use entroy. Else return random
    if not entropy:
        n = len(attributes)
        i = random.randint(0,n-1)
        return attributes[i]
    else:
        currentBestGain = 0
        currentBestAttribute = -1
        for A in attributes:
            nextGain = gain(examples,A)
            if nextGain > currentBestGain:
                currentBestGain = nextGain
                currentBestAttribute = A
        return A

def treeLearning(examples, attributes, parentExamples,entropy):
    #print('New node')
    sameClassValue = checkSameClassification(examples)
    if not len(examples):
        #print('No examples')
        return treeNode(-1,pluralityValue(parentExamples))
    elif sameClassValue > 0:
        #print('Same classification')
        return treeNode(-1,sameClassValue)
    elif not len(attributes):
        #print('No attributes')
        return treeNode(-1,pluralityValue(examples))
    else:
        A = importance(attributes,entropy,examples)
        node = treeNode(A,-1)

        attributes.remove(A)

        for i in [1,2]:
            exs = []
            for e in examples:
                if e[A] == i:
                    exs.append(e)
            subTree = treeLearning(exs,attributes,examples,entropy)

            node.children[i-1] = subTree
        return node

def printTree(tree, level,prevTestAttribute,resutPrev):
    print('Level: ' + repr(level) + ' Node with testAttribute: '+repr(tree.testAttribute) + ' and label: ' + repr(tree.label) + 'PreTestAttr: ' + repr(prevTestAttribute) + ' Prevres ' + repr(resutPrev))
    for i in [0,1]:
        child = treeNode
        child = tree.children[i]
        if child != None:
            printTree(child,level +1,tree.testAttribute,i+1)

def classify(example,treeRoot):
    for i in [0,1]:
        if treeRoot.children[i] != None and example[treeRoot.testAttribute] == i+1:
            return classify(example,treeRoot.children[i])
    if treeRoot.label != -1:
        return treeRoot.label
    else:
        return -1

def testTree(tree, testData):
    totalCases = 0
    correctlyClassified = 0

    for test in testData:
        result = classify(test,tree)
        totalCases += 1

        if result == test[-1]:
            correctlyClassified +=1
    return ([totalCases,correctlyClassified])

print('Here we go!')
trainingData = getData('training.txt')
testData = getData('test.txt')

originalAttributes = []

for i in range(len(trainingData[0])-1):
    originalAttributes.append(i)

rootNode = treeLearning(trainingData,originalAttributes,[],1)

print('Learning complete')
print('')
print('')
print('')

[cases, successes] = testTree(rootNode,testData)


print('The results are in. Of ' + repr(cases) + '; ' + repr(successes) + ' were correct.')
