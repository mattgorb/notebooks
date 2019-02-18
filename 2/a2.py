import numpy as np
import random
import sys
import copy



def printState_8p(startState):
	cp=copy.copy(startState)
	zero=cp.index(0)
	cp[zero]="-"
	converted=np.array(cp).reshape(3,3)
	print(converted)

def print_blank(n):
	if(n>0):
		print ' ',
		print_blank(n-1)
def printPath_8p(startState, goalState, path):
	print("Path from")
	printState_8p(startState)
	print("to")
	printState_8p(goalState)
	print(" is "+str(len(path))+ " nodes long:")
	for p in range(0,len(path)):
		print_blank(p)
		printState_8p(path[p])
	
    

def findBlank_8p(startState):
	return startState.index(0)

def actionsF_8p(startState):
	#print(startState)
	blank=findBlank_8p(startState)
	if(blank==0):
		return ['down','right']
	elif(blank==1):
		return ['left','down','right']
	elif(blank==2):
		return ['left','down']
	elif(blank==3):
		return ['up','down', 'right']
	elif(blank==4):
		return ['up','right','left','down']
	elif(blank==5):
		return ['left','down','up']
	elif(blank==6):
		return ['up','right']
	elif(blank==7):
		return ['left','up','right']
	elif(blank==8):
		return ['left','up']

def takeActionF_8p(startState, action):
	cp=copy.copy(startState)
	blank=findBlank_8p(cp)
	if(action=='down'):
		cp[blank], cp[blank+3] = cp[blank+3], cp[blank]
	elif(action=='up'):
		cp[blank], cp[blank-3] = cp[blank-3], cp[blank]
	elif(action=='left'):
		cp[blank], cp[blank-1] = cp[blank-1], cp[blank]
	elif(action=='right'):
		cp[blank], cp[blank+1] = cp[blank+1], cp[blank]
	return cp


def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit):
    if(state == goalState):
        return []
    if(depthLimit==0):
        return 'cutoff'
    cutoffOccurred = False
    for action in actionsF(state):
        childState = takeActionF(state, action)
        result = depthLimitedSearch(childState, goalState, actionsF, takeActionF, depthLimit-1)
        if(result=='cutoff'):
            cutoffOccurred = True
        elif(result!='failure'):
            result.insert(0, childState)
            return result
    if(cutoffOccurred):
        return 'cutoff'
    else:
        return 'failure'








def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
    for depth in range(maxDepth):
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)
        if(result=='failure'):
            return 'failure'
        if(result!='cutoff'):
            result.insert(0, startState)   
            return result
    return 'cutoff'


def actionsF_Hanoi(startState):
	#print(startState)
	blank=findBlank_8p(startState)
	if(blank==0):
		return ['down','right']
	elif(blank==1):
		return ['left','down','right']
	elif(blank==2):
		return ['left','down']
	elif(blank==3):
		return ['up','down', 'right']
	elif(blank==4):
		return ['up','right','left','down']
	elif(blank==5):
		return ['left','down','up']
	elif(blank==6):
		return ['up','right']
	elif(blank==7):
		return ['left','up','right']
	elif(blank==8):
		return ['left','up']


startState_hanoi=[[7,6,5,4,3,2,1,0],[],[]]
startState_hanoi=[[],[7,6,5,4,3,2,1,0],[]]

goalState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
#startState = [1, 2, 3, 4, 5, 0, 6, 7, 8]
startState = [1, 0, 3, 4, 2, 5, 6, 7, 8]
solutionPath = iterativeDeepeningSearch(startState,goalState, actionsF_8p, takeActionF_8p, 15)
printPath_8p(startState, goalState, solutionPath)
