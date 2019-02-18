import copy
import sys

def successorsf(state):
	return copy.copy(successors.get(state, []))

successors = {'a':  ['b', 'c', 'd'],
              'b':  ['e', 'f', 'g'],
              'c':  ['a', 'h', 'i'],
              'd':  ['j', 'z'],
              'e':  ['k', 'l'],
              'g':  ['m'],
              'k':  ['z']}


def breadthFirstSearch_original(startState, goalState, successorsf):
	queue = [startState]
	parent = {} 
	parent[startState]= 0    
	visited=[startState]
	while queue:
		node=queue.pop(0)
		children=successorsf(node)
		for child in children:
			if child not in visited:
				queue.append(child)
				visited.append(child)
				parent[child]= node
				if(child==goalState):
					holder=child
					break;
	steps_list=[]	
	while holder!=0:
		steps_list.append(holder)
		holder=parent[holder]
	return list(reversed(steps_list))
	

def breadthFirstSearch_First(startState, goalState, successorsf):
	queue = [startState]
	parent = {} 
	parent[startState]= 0    
	visited=[startState]
	while queue:
		node=queue.pop(0)
		children=successorsf(node)
		for child in children:
			successorsf(child).append(node)
		for child in children:
			if child not in visited:
				queue.append(child)
				visited.append(child)
				parent[child]= node
				if(child==goalState):
					holder=child
					break;
	steps_list=[]	
	while holder!=0:
		steps_list.append(holder)
		holder=parent[holder]
	return list(reversed(steps_list))



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



 

def depthFirstSearch(startState, goalState, successorsf):
    parent = {}
    parent[startState]= 0 
    parent = dfs(successorsf, startState, parent, goalState)
    steps_list=[]
    print(parent)
    holder=goalState
    while holder!=0:
        steps_list.append(holder)
        holder=parent[holder]
    return list(reversed(steps_list))



def dfs(startState, goalState, successorsf):
	parent = {}
	stack=[]
	parent[startState]= 0 
	stack.append(startState);
	visited=[startState]
	while(len(stack)>0):
		children= successorsf(stack[-1])
		hasUnvisitedChild=next((child for child in children if child not in visited), "None")			
		if(hasUnvisitedChild!="None"):
			index=children.index(hasUnvisitedChild)
			if(children[index]==goalState):
				break;
			visited.append(children[index])
			stack.append(children[index])
		else:
			stack.pop()
	return stack
	





def dfs_2(startState, goalState, successorsf):
	parent = {}
	stack=[]
	parent[startState]= 0 

	stack.append(startState);
	visited=[startState]
	#parent[startState]=0
	while(len(stack)>0):
		
		children= successorsf(stack[-1])
		hasUnvisitedChild=next((child for child in children if child not in visited), "None")
				
		if(hasUnvisitedChild!="None"):
			index=children.index(hasUnvisitedChild)
			#parent[children[index]]=stack[-1]
			if(children[index]==goalState):
				print(len(stack))
				break;
			visited.append(children[index])
			stack.append(children[index])
		else:
			stack.pop()
	return stack



#print(depthFirstSearch('a', 'm', successorsf))
#print('path from (0, 0) to (9, 9) is', depthFirstSearch((0, 0), (9, 9), gridSuccessors))


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

def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))


startState=('R', 'R', 'R', 'R', ' ', 'L','L','L','L')
goalState=('L','L','L','L', ' ', 'R', 'R', 'R', 'R')


#print(breadthFirstSearch_(startState, goalState, camelSuccessorsf))

#print(dfs2('a', 'z', successorsf))
#dfs2((0, 0), (9, 9), gridSuccessors)

#print(path)
#print(dfs_2((0, 0), (9, 9), gridSuccessors))
#print(len(dfs_2(startState, goalState, camelSuccessorsf)))

	
#children=camelSuccessorsf(children)
#children=camelSuccessorsf(children)















def nodeSearch(startState, goalState, successorsf, breadthFirst):

    '''
    Given a startState, and goalState, and a function to
    calculate successors, and a boolean representing a breadthFirst search, 
    nodeSearch returns a breadthFirst search if breadthFirst==True, 
    or a depthFirst search if breadthFirst==False'''
    
    # Initialize expanded to be empty dictionary
    expanded = {}
    # Initialize unExpanded to be list containing (startState, None)
    unExpanded = [(startState, None)]

    if startState == goalState:
        return [startState]

    while unExpanded:

        state = unExpanded.pop()
        parent = state[0]

        # remove children which match parent
        children = [child for child in successorsf(parent) if child != parent  ]

        filteredChildren = []

        # Filter children, removing those in expanded, and whose key is in unExpanded, 
        # and where the child is the same as the parent
        for child in children:
            if child not in expanded and (child, parent) not in unExpanded and child != parent:
                filteredChildren.append(child)

        children = filteredChildren

        # don't overwrite the expanded dictionary if the parent is already in it
        if(parent not in expanded):
            expanded[parent] = state[1]

        # if the children contain the final state, iterate through the expanded
        # dictionary, adding parents to the solutionPath, then return it.
        if goalState in children:
            solutionPath = [state[0],goalState]

            parent = expanded[parent]
            while parent:
                solutionPath.insert(0, parent)
                parent = expanded[parent]

            return solutionPath

        # Sort and reverse the children
        children.sort()
        children.reverse()

        # convert the children to tuples of the child and parent
        childTuples = [(c, p) for c in children for p in [state[0]]]

        # if a Breadth-First search, add the childTuples to the front of the unExpanded list
        # if Depth-First, add the childTuples to the end of the unExpanded list
        if breadthFirst:
            unExpanded = childTuples + unExpanded
        else:
            unExpanded.extend(childTuples)
            
def breadthFirstSearch2(startState, goalState, successorsf):
    '''
    Given a startState, and goalState, and a function to
    calculate successors, breadthFirstSearch returns a search of the nodes
    using a Breadth First Search algorithm '''
    
    return nodeSearch(startState, goalState, successorsf, True)


def depthFirstSearch2(startState, goalState, successorsf):
    '''
    Given a startState, and goalState, and a function to
    calculate successors, depthFirstSearch returns a search of the nodes
    using a Depth First Search algorithm '''

    return nodeSearch(startState, goalState, successorsf, False)
	

print(len(depthFirstSearch2(startState, goalState, camelSuccessorsf)))
