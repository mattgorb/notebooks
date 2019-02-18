#!/usr/bin/env python
# coding: utf-8

# # A3: A\*, IDS, and Effective Branching Factor
# 
# ### Name: Matt Gorbett

# ## Overview
# 
# For this assignment, I implemented the aStarSearch function along with a modified version of iterative deepening search from the last assignment.  On top of this, I also implemented the effective branching factor binary search algorithm along with a method to test the effectiveness of both aStarSearch and ITS, titled runExperiment.  Here is a summary of my methods:
# 
# #### aStarSearch(startState, actionsF, takeActionF, goalTestF, hF)
# This method was taken was lecture notes on aStarSearch.  Understanding this function was hard enough, I'm glad we didn't have to implement it from scratch.  
# 
# #### iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
# This was modified from my last assignment to include cost.  I modified each function response to return a tuple of (x,cost=1).  
# 
# #### ebf(nodes, depth, precision=0.01)
# This was a fun function to implement.  It was a basic binary search algorithm searching for a midpoint with precision=0.01.  My implementation of the function from the notes was the following:
# calc=(1-midpoint**(depth+1))/(1-midpoint)
# If this value minus the total number of nodes is less than the precision, we have found the point and we can return the value.  I confirmed these were the same values as the values when calculated in the original notebook.  
# 
# #### h1_8p(state, goal)
# This function returns 0.
# 
# #### h2_8p(state, goal)
# This function returns the Manhattan distance.  To do this, I returned the (rows,columns) from the findBlank_8p function and execute the euclidean distance, which is:
# abs(statePosition[0]-goalPosition[0])+abs(statePosition[1]-goalPosition[1])
# 
# For the 8 puzzle, it is required to find the distance on both the x and y planes
# 
# #### h3_8p(state, goal)
# This function returns the Euclidean distance.  To do this, I returned the (rows,columns) from the findBlank_8p function and execute the euclidean distance, which is:
# math.sqrt(abs(statePosition[0]-goalPosition[0])**2+abs(statePosition[1]-goalPosition[1])**2).
# 
# #### runExperiment(goalState1, goalState2, goalState3, h) *extra credit included
# For this function, I created three tables for each of the three goal states.  I looped three the three goal states and executed IDS and A* with each of the 3 heuristic functions.  I was able to log the time each run took and add it to the pandas table.  A tricky part of this method was logging the depth and nodes of each algorithm.  I used global variables for each search function and incremented them where necessary.  Once each function completed, I set the variables back to 0.     
# 
# 

# In[87]:


import numpy as np
import copy
import math

class Node:
    def __init__(self, state, f=0, g=0 ,h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
    def __repr__(self):
        return "Node(" + repr(self.state) + ", f=" + repr(self.f) +                ", g=" + repr(self.g) + ", h=" + repr(self.h) + ")"

def aStarSearch(startState, actionsF, takeActionF, goalTestF, hF):
    """
    this method was taken from http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/07%20Informed%20Search.ipynb"""
    global Nodes
    global prevBest
    prevBest=None
    Nodes=0
    h = hF(startState)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'))

def aStarSearchHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax):
    global Depth
    global Nodes
    global prevBest
    
    if goalTestF(parentNode.state):
        return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode.state)
    if not actions:
        return ("failure", float('inf'))
    children = []

    for action in actions:
        (childState,stepCost) = takeActionF(parentNode.state, action)
        h = hF(childState)
        g = parentNode.g + stepCost
        f = max(h+g, parentNode.f)
        childNode = Node(state=childState, f=f, g=g, h=h)
        children.append(childNode)

            

    while True:
        Nodes+=1
        # find best child
        children.sort(key = lambda n: n.f) # sort by f value
        #print(children)
        bestChild = children[0]
        
        '''prevBest was my attempt to not include failures in this function, because they result in infinite loops.  However
        i wasnt quite able to get this to work.  keeping code in here for future purposes.  '''
        
        '''if(prevBest is None):
            prevBest=bestChild
        else:
            if(prevBest.f==bestChild.f and prevBest.g==bestChild.g and prevBest.h==bestChild.h):
                
                return "Could not find"
            else:
                prevBest=copy.copy(bestChild)
                print(prevBest)
        '''
        
        if bestChild.f > fmax:
            return ("failure",bestChild.f)
        # next lowest f value
        alternativef = children[1].f if len(children) > 1 else float('inf')
        # expand best child, reassign its f value to be returned value
        Depth=min(fmax,alternativef)
        result,bestChild.f = aStarSearchHelper(bestChild, actionsF, takeActionF, goalTestF,
                                            hF, min(fmax,alternativef))
        if result is not "failure":               #        g
            result.insert(0,parentNode.state)     #       / 

            return (result, bestChild.f)          #      d
                                                  #     / \ 
#Used these methods for testing purposes.  
successors = {'a': ['b','c'],                 #  a   e  
                  'b': ['d','e'],                 #   \         
                  'c': ['f'],                     #    c   i
                  'd': ['g', 'h'],                #     \ / 
                  'f': ['i','j']}                 #      f  

def actionsF(s):                              #       \ 
    try:                                      #        j
        ## step cost of each action is 1
        return [(succ,1) for succ in successors[s]]
    except KeyError:
            return []
def takeActionF(s,a):
    return a
def goalTestF(s):
    return s == goal
def h1(s):
    return 0







def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit, cost=0):
    global Nodes
    global Depth
    if(state == goalState):
        return []
    if(depthLimit==0):
        return 'cutoff'
    cutoffOccurred = False
    for action in actionsF(state):
        childState = takeActionF(state, action)
        Nodes+=1
        cost+=childState[1]
        result = depthLimitedSearch(childState[0], goalState, actionsF, takeActionF, depthLimit-1)
        if(result=='cutoff'):
            cutoffOccurred = True
        elif(result!='failure'):
            result.insert(0, childState[0])
            return result
    if(cutoffOccurred):
        return 'cutoff'
    else:
        return 'failure'



def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
    """
    taken from previous assignment.added nodes and depth global variables for experiment"""
    global Nodes
    global Depth
    for depth in range(maxDepth):
        Depth=depth
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)
        if(result=='failure'):
            return 'failure'
        if(result!='cutoff'):
            result.insert(0, startState)
            return result
        
    return 'cutoff'



def actionsF_8p(startState):
	blank=findBlank(startState)
	if(blank==0):
		return [('down',1),('right',1)] #actions available when 0 is at first square
	elif(blank==1):
		return [('left',1),('down',1),('right',1)]
	elif(blank==2):
		return [('left',1),('down',1)]
	elif(blank==3):
		return [('up',1),('down',1), ('right',1)]
	elif(blank==4):
		return [('up',1),('right',1),('left',1),('down',1)]
	elif(blank==5):
		return [('left',1),('down',1),('up',1)]
	elif(blank==6):
		return [('up',1),('right',1)]
	elif(blank==7):
		return [('left',1),('right',1),('up',1)]
	elif(blank==8):
		return [('left',1),('up',1)]


def takeActionF_8p(startState, action):
	cp=copy.copy(startState) #create copy of startState to alter
	blank=findBlank(cp)
	if(action[0]=='down'):
		cp[blank], cp[blank+3] = cp[blank+3], cp[blank]
	elif(action[0]=='up'):
		cp[blank], cp[blank-3] = cp[blank-3], cp[blank]
	elif(action[0]=='left'):
		cp[blank], cp[blank-1] = cp[blank-1], cp[blank]
	elif(action[0]=='right'):
		cp[blank], cp[blank+1] = cp[blank+1], cp[blank]
	return (cp,1)

def findBlank(startState):
	return startState.index(0) #to get the index of the 0 in a normal (non NumPy) list

def goalTestF_8p(state, goal):
    return state==goal

def ebf(nodes, depth, precision=0.01):
    """binary search, with last equal to number of numbers.  Execute the equation given in lecture notes to 
    determine if we've found b (calc).  """
    if (depth==0):
        return 1

    first = 1
    last = nodes
    found = False
    midpoint = 0


    while first <= last and not found:
        midpoint = (first + last) / 2
        if(midpoint!=1):
            calc=(1-midpoint**(depth+1))/(1-midpoint)
        else:
            calc=1
        if abs(calc - nodes) < precision:
            found = True
        else:
            if nodes < calc:
                last = midpoint
                
            else:
                first = midpoint
                

    return midpoint




start = 'a'
goal = 'h'
result = aStarSearch(start,actionsF,takeActionF,goalTestF,h1)
print('Path from a to h is', result[0], 'for a cost of', result[1])
print(result)


# ## Heuristic Functions
# 
#   * `h1_8p(state, goal)`: $h(state, goal) = 0$, for all states $state$ and all goal states $goal$,
#   * `h2_8p(state, goal)`: $h(state, goal) = m$, where $m$ is the Manhattan distance that the blank is from its goal position,
#   * `h3_8p(state, goal)`: $h(state, goal) = e$, where e is the euclidean distance from blank to its goal position. 

# In[88]:


import math
def findBlank_8p(startState):
	row=math.floor(startState.index(0)/3) #equal to the index of 0 divided by 3 and rounded down
	column=startState.index(0)%3 #the column is convienently equal to the remainder when divided by 3
	return (row,column)

def h1_8p(state, goal):
    return 0

def h2_8p(state,goal):
    statePosition = findBlank_8p(state)
    goalPosition = findBlank_8p(goal)
    return abs(statePosition[0]-goalPosition[0])+abs(statePosition[1]-goalPosition[1])
    
def h3_8p(state,goal):
    statePosition = findBlank_8p(state)
    goalPosition = findBlank_8p(goal)
    return math.sqrt(abs(statePosition[0]-goalPosition[0])**2+abs(statePosition[1]-goalPosition[1])**2)


# ## Comparison

# Apply all four algorithms (`iterativeDeepeningSearch` plus `aStarSearch` with the three heuristic
# functions) to three eight-tile puzzle problems with start state
# 
# $$
# \begin{array}{ccc}
# 1 & 2 & 3\\
# 4 & 0 & 5\\
# 6 & 7 & 8
# \end{array}
# $$
# 
# and these three goal states.
# 
# $$
# \begin{array}{ccccccccccc}
# 1 & 2 & 3  & ~~~~ & 1 & 2 & 3  &  ~~~~ & 1 & 0 &  3\\
# 4 & 0 & 5  & & 4 & 5 & 8  & & 4 & 5 & 8\\
# 6 & 7 & 8 &  & 6 & 0 & 7  & & 2 & 6 & 7
# \end{array}
# $$

# In[89]:


import pandas as pd
import time


def runExperiment(goalState1, goalState2, goalState3, h):
    """set global variables to find nodes and depth of each search.  for each goal, create a new pandas dataframe and
    execute ebf and time methods to add to dataframe along with nodes and depth.  
    """
    global Depth
    global Nodes
    Depth=0
    Nodes=0
    
    h1_8p = h[0]
    h2_8p = h[1]
    h3_8p = h[2]
    
    for goal in [goalState1,goalState2,goalState3]:
        results=pd.DataFrame(columns=['Algorithm','Depth', 'Nodes', 'EBF','Duration (sec)'])
        print(goal)
        Nodes = 0
        Depth = 0
        start_time = time.time()
        solutionPath = iterativeDeepeningSearch(startState, goal, actionsF_8p, takeActionF_8p, 10)
        end_time = time.time()

        results.loc[-1] = ["IDS", Depth,Nodes,ebf(Nodes,Depth),end_time-start_time]
        results.index = results.index + 1
        
        start_time = time.time()
        solutionPath =aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goal),lambda s: h1_8p(s, goal))
        end_time = time.time()        
        results.loc[-1] = ["A*H1", Depth,Nodes,ebf(Nodes,Depth),end_time-start_time]
        results.index = results.index + 1   

        start_time = time.time()
        solutionPath = aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goal),lambda s: h2_8p(s, goal))
        end_time = time.time()        
        results.loc[-1] = ["A*H2", Depth,Nodes,ebf(Nodes,Depth),end_time-start_time]
        results.index = results.index + 1   
        
        start_time = time.time()
        solutionPath =aStarSearch(startState, actionsF_8p, takeActionF_8p, lambda s: goalTestF_8p(s, goal),lambda s: h3_8p(s, goal))
        end_time = time.time()
        results.loc[-1] = ["A*H3", Depth,Nodes,ebf(Nodes,Depth),end_time-start_time]
        results.index = results.index + 1   
        
        print(results.to_string(index=False))
        print()
        print()

    
startState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goalState1 = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goalState2 = [1, 2, 3, 4, 5, 8, 6, 0, 7]
goalState3 = [1, 0, 3, 4, 5, 8, 2, 6, 7]
runExperiment(goalState1, goalState2, goalState3, [h1_8p, h2_8p, h3_8p])


# First, some example output for the ebf function.  During execution, this example shows debugging output which is the low and high values passed into a recursive helper function.
# 
# ## Tests from original notebook 

# In[90]:


ebf(10, 3)


# The smallest argument values should be a depth of 0, and 1 node.

# In[91]:


ebf(1, 0)


# In[92]:


ebf(2, 1)


# In[93]:


ebf(2, 1, precision=0.000001)


# In[94]:


ebf(200000, 5)


# In[95]:


ebf(200000, 50)


# Here is a simple example using our usual simple graph search.

# In[96]:


def actionsF_simple(state):
    succs = {'a': ['b', 'c'], 'b':['a'], 'c':['h'], 'h':['i'], 'i':['j', 'k', 'l'], 'k':['z']}
    return [(s, 1) for s in succs.get(state, [])]

def takeActionF_simple(state, action):
    return action

def goalTestF_simple(state, goal):
    return state == goal

def h_simple(state, goal):
    return 1


# In[97]:


actions = actionsF_simple('a')
actions


# In[98]:


takeActionF_simple('a', actions[0])


# In[99]:


goalTestF_simple('a', 'a')


# In[100]:


h_simple('a', 'z')


# In[101]:


iterativeDeepeningSearch('a', 'z', actionsF_simple, takeActionF_simple, 10)


# In[102]:


aStarSearch('a',actionsF_simple, takeActionF_simple,
            lambda s: goalTestF_simple(s, 'z'),
            lambda s: h_simple(s, 'z'))


# ## Grading

# # Results

# In[103]:


get_ipython().run_line_magic('run', '-i A3grader.py')


# ## Extra Credit

# Implemented in my runExperiment method.  
# 
