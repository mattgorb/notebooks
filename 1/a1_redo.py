import copy
import sys

successors = {'a':  ['b', 'c', 'd'],
              'b':  ['e', 'f', 'g'],
              'c':  ['a', 'h', 'i'],
              'd':  ['j', 'z'],
              'e':  ['k', 'l'],
              'g':  ['m'],
              'k':  ['z']}
def successorsf(state):
	return copy.copy(successors.get(state, []))

def gridSuccessors(state):
    row, col = state
    # succs will be list of tuples () rather than list of lists [] because state must
    # be an immutable type to serve as a key in dictionary of expanded nodes
    succs = []
    for r in [-1, 0, 1]:   #check each row
        for c in [-1, 0, 1]:  # check in each col
            newr = row + r
            newc = col + c
            if 0 <= newr <= 9 and 0 <= newc <= 9:  
                succs.append( (newr, newc) )
    return succs


'''Keeping this as a script-y function to illustrate logic used. 
Depending on where the space is, I create new states by flipping the space with its neighbors.  
For most states, four new copies can be made by flipping the space with its neighbors from 
index-2 to index+2.  Exceptions are made when the space gets closer to the end.  
''' 
def camelSuccessorsf(state):

	empty_loc=state.index(' ')

	copy1=list(state).copy()
	copy2=list(state).copy()
	copy3=list(state).copy()
	copy4=list(state).copy()
	if(empty_loc==0):
		copy1[empty_loc+2], copy1[empty_loc] = copy1[empty_loc], copy1[empty_loc+2]
		return [tuple(copy1)]

	elif(empty_loc ==1):
		copy1[empty_loc-1], copy1[empty_loc] = copy1[empty_loc], copy1[empty_loc-1]
		copy2[empty_loc+1], copy2[empty_loc] = copy2[empty_loc], copy2[empty_loc+1]
		copy3[empty_loc+2], copy3[empty_loc] = copy3[empty_loc], copy3[empty_loc+2]
		return tuple(copy1),tuple(copy2),tuple(copy3)

	elif(empty_loc==7):
		copy1[empty_loc-1], copy1[empty_loc] = copy1[empty_loc], copy1[empty_loc-1]
		copy2[empty_loc+1], copy2[empty_loc] = copy2[empty_loc], copy2[empty_loc+1]
		copy3[empty_loc-2], copy3[empty_loc] = copy3[empty_loc], copy3[empty_loc-2]
		return tuple(copy3),tuple(copy1),tuple(copy2)
	elif(empty_loc ==8):
		copy1[empty_loc-2], copy1[empty_loc] = copy1[empty_loc], copy1[empty_loc-2]
		return [tuple(copy1)]

	else:
		copy1[empty_loc-1], copy1[empty_loc] = copy1[empty_loc], copy1[empty_loc-1]
		copy2[empty_loc+1], copy2[empty_loc] = copy2[empty_loc], copy2[empty_loc+1]
		copy3[empty_loc+2], copy3[empty_loc] = copy3[empty_loc], copy3[empty_loc+2]
		copy4[empty_loc-2], copy4[empty_loc] = copy4[empty_loc], copy4[empty_loc-2]
		return tuple(copy4),tuple(copy1),tuple(copy2), tuple(copy3)


startState=('R', 'R', 'R', 'R', ' ', 'L','L','L','L')
goalState=('L','L','L','L', ' ', 'R', 'R', 'R', 'R')

def searchNodes(startState, goalState, successorsf, breadthFirst):
	expanded={}
	unExpanded=[(startState, None)]
	if(startState==goalState):
		return [startState]
	while unExpanded:
		currentState=unExpanded.pop()
		children=[child for child in successorsf(currentState[0]) if child!=currentState[0]  ]
			
		
		cleanup=[]
		for child in children:
			if child not in expanded and (child,currentState) not in unExpanded and child!=currentState:
				cleanup.append(child)
		children=cleanup
		for child in children:
			expanded[child]=currentState[0]

		if(currentState[0] not in expanded):
			expanded[currentState[0]] = currentState[1]
		
		if(goalState in children):
			solution=[currentState[0],goalState]
			currentState=expanded[currentState[0]]
			while currentState:
				solution.insert(0,currentState)
				currentState=expanded[currentState]
			return solution

		
		children.sort()
		children.reverse()
		children_tupled=[(child,parent) for child in children for parent in currentState[0] ]
		
		if(breadthFirst):
			unExpanded=children_tupled+unExpanded		
		else:
			unExpanded.extend(children_tupled)
	return "Goal not found"


def breadthFirstSearch(startState, goalState, successorsf):
    return searchNodes(startState, goalState, successorsf, True)


def depthFirstSearch(startState, goalState, successorsf):
    return searchNodes(startState, goalState, successorsf, False)


print(depthFirstSearch('a', 'denver', successorsf))
#print(depthFirstSearch('a', 'z', successorsf))
'''
print(depthFirstSearch((0, 0), (9, 9), gridSuccessors))
print(breadthFirstSearch((0, 0), (9, 9), gridSuccessors))
'''
print(len(depthFirstSearch(startState, goalState, camelSuccessorsf)))

