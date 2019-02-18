#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Iterative-Deepening Search

# Matt Gorbett

# ## Overview

# Iterative Deepening Search is a search algorithm which uses a depth first search on increasingly large depth-limits until a search result is found.  In this way it has qualities of both depth first and breadth first searches.  The algorithm iterates through depths, and in each level of depth-limit it does a depth first search to attempt to find the correct value.  If it is not found, it moves to a higher depth-limit in the tree until either a value is found or the algorithm has reached its maximum search space.  
# 
# This was a fun and tricky project with interesting methods to write.  My laptop broke on me on Monday which made it extra spicy to complete on time.  I ended up converting from Ubuntu to Mac for this assignment, so far I am pleased with my purchase and the OS conversion has been pretty seamless.  Even though this was a little tricky at times, I feel like I learned the algorithm pretty well and began to get a grasp on some deeper parts of these ideas. 
# 
# 
# #### Contents:
# -  DepthLimitedSearch, IterativeDeepeningSearch Functions
# -  8 Puzzle
# -  Testing Code
# -  A2Grader Test Results
# -  15 Puzzle
# -  Hanoi Towers 

# # DepthLimitedSearch, IterativeDeepeningSearch Functions

# ### Functions: 
# #### iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth)
# Main function used to iterate over hte depth limited search up until a maxDepth limit specified in input.  
# #### depthLimitedSearch(startState, goalState, actionsF, takeActionF, depthLimit)
# Main function to do a depth search up until a limit (depthLimit)

# In[1775]:


import numpy as np
import copy
import math

def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit):
    if(state == goalState):
        return []
    if(depthLimit==0):
        return 'cutoff'
    cutoffOccurred = False
    for action in actionsF(state):
        childState = takeActionF(state, action)
        #if(childState):
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


# # 8 Puzzle
# 
# ### Functions
# #### findBlank_8p(state) 
# Finds the 0 in the 8 puzzle and returns the matrix positions
# #### actionsF_8p(state) 
# Returns the actions-left, right, up, down.  
# #### takeActionF_8p(state, action)
# This method executes an action that gets input into it.  For an 8 puzzle, the left and right actions switch
# the before and after values .
# #### findBlank(startState):find blank in a list rather than the matrix row column value
# I used this to find the index position of all lists used in this assignment.  
# #### printPath_8p(startState, goalState, path), print_8p()
# I am not proud of my two print functions, however I was running out of time on this assignment.  
# They do the trick, however ugly they are.  

# In[1776]:



        
def findBlank_8p(startState):
	row=math.floor(startState.index(0)/3)
	column=startState.index(0)%3
    #return (zero)
	return (row,column)

def findBlank(startState):
	return startState.index(0)

def actionsF_8p(startState):
	#print(startState)
	blank=findBlank(startState)
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
		return ['left','right','up']
	elif(blank==8):
		return ['left','up']


def takeActionF_8p(startState, action):
	cp=copy.copy(startState) #create copy of startState to alter
	blank=findBlank(cp)
	if(action=='down'):
		cp[blank], cp[blank+3] = cp[blank+3], cp[blank]
	elif(action=='up'):
		cp[blank], cp[blank-3] = cp[blank-3], cp[blank]
	elif(action=='left'):
		cp[blank], cp[blank-1] = cp[blank-1], cp[blank]
	elif(action=='right'):
		cp[blank], cp[blank+1] = cp[blank+1], cp[blank]
	return cp



def printState_8p(startState):
	cp=copy.copy(startState)
	zero=cp.index(0)#find zero value
	cp[zero]="-"#replace zero value with -
	print(str(cp[0])+' '+str(cp[1])+' '+str(cp[2]))#print the array without brackets and parentheses.  
	print(str(cp[3])+' '+str(cp[4])+' '+str(cp[5]))    
	print(str(cp[6])+' '+str(cp[7])+' '+str(cp[8]))
        
def printPath_8p(startState, goalState, path):
	print("Path from") #print full startState 
	print(str(startState[0])+' '+str(startState[1])+' '+str(startState[2]))
	print(str(startState[3])+' '+str(startState[4])+' '+str(startState[5]))    
	print(str(startState[6])+' '+str(startState[7])+' '+str(startState[8]))
	print("to") #print goalState
	print(str(goalState[0])+' '+str(goalState[1])+' '+str(goalState[2]))
	print(str(goalState[3])+' '+str(goalState[4])+' '+str(goalState[5]))    
	print(str(goalState[6])+' '+str(goalState[7])+' '+str(goalState[8]))
	print("is "+str(len(path))+ " nodes long:") #count steps to get from start to goal
	printer='' #for spacing steps
	for p in range(0,len(path)):
		path[p][path[p].index(0)]='-'#change 0 to - for each array
		print(printer+str(path[p][0])+' '+str(path[p][1])+' '+str(path[p][2]))
		print(printer+str(path[p][3])+' '+str(path[p][4])+' '+str(path[p][5]))    
		print(printer+str(path[p][6])+' '+str(path[p][7])+' '+str(path[p][8]))
		print("")#print blank line
		printer += ' ' #add space to printer variable each iteration to add stepping procedure onto print method


# # Testing code
# #### This code is from the original notebook for this homework.  These came in handy for testing purposes and I will leave these here to show results.  

# In[1777]:


startState = [1, 0, 3, 4, 2, 5, 6, 7, 8]


# In[1778]:


printState_8p(startState)  # not a required function for this assignment, but it helps when implementing printPath_8p


# In[1779]:


findBlank_8p(startState)


# In[1780]:


actionsF_8p(startState)


# In[1781]:


takeActionF_8p(startState, 'down')


# In[1782]:


printState_8p(takeActionF_8p(startState, 'down'))


# In[1783]:


goalState = takeActionF_8p(startState, 'down')


# In[1784]:


newState = takeActionF_8p(startState, 'down')


# In[1785]:


newState == goalState


# In[1786]:


startState


# In[1787]:


path = depthLimitedSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# Notice that `depthLimitedSearch` result is missing the start state.  This is inserted by `iterativeDeepeningSearch`.
# 
# But, when we try `iterativeDeepeningSearch` to do the same search, it finds a shorter path!

# In[1788]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# Also notice that the successor states are lists, not tuples.

# In[1789]:


startState = [4, 7, 2, 1, 6, 5, 0, 3, 8]
path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# In[1790]:


startState = [4, 7, 2, 1, 6, 5, 0, 3, 8]
path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 5)
path


# Humm...maybe we can't reach the goal state from this state.  We need a way to randomly generate a valid start state.

# In[1791]:


import random


# In[1792]:


random.choice(['left', 'right'])


# In[1793]:


def randomStartState(goalState, actionsF, takeActionF, nSteps):
    state = goalState
    for i in range(nSteps):
        state = takeActionF(state, random.choice(actionsF(state)))
    return state


# In[1794]:


goalState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
randomStartState(goalState, actionsF_8p, takeActionF_8p, 10)


# In[1795]:


startState = randomStartState(goalState, actionsF_8p, takeActionF_8p, 50)
startState


# In[1796]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 20)
path


# Let's print out the state sequence in a readable form.

# In[1797]:


for p in path:
    printState_8p(p)
    print()


# Here is one way to format the search problem and solution in a readable form.

# In[1798]:


printPath_8p(startState, goalState, path)


# # A2Grader Test Results

# In[1799]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # 15 puzzle
# 
# [15 puzzle Wiki](https://en.wikipedia.org/wiki/15_puzzle)
# 
#    This function follows the same logic as the 8 puzzle, only with more spaces and a different endpoint destination for the 0.  I tried to solve simple puzzles so that this algorithm didn't take forever to run.  I found one good one and two easy ones.  
# 
# ## Functions
#    #### actionsF_15p(startState)
#    I was able to group points together in this function that have the same moving options.  For example, in all the middle boxes (5,6,9,10), you can move left, right, up, or down.  This consolidates and cleans up the code a little bit, though there is still much room for cleaning up this code even more.  
#    
#    #### takeActionF_15p(startState, action):
#    This is very similiar to the 8 puzzle actions.  The only difference is that if you want to move up, you need to move back 4 spaces in the array, and down 4 spaces forward.  
# 
# 
# 
# ## Solvable
# https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
# I implemented this function on a 4x4 matrix to check whether my puzzles were solvable.  They were taking so long and I wasn't sure, so I wrote this to be safe.  Functions: getRow(), countInversions(), solveable()
# 

# In[1800]:


def actionsF_15p(startState):
	blank=findBlank(startState)
	if(blank==0):
		return ['right','down'] #these are the valid moves when the 0 is at the first space.  
	elif(blank in (1,2)):
		return ['left','right','down']
	elif(blank==3):
		return ['left','down']
	elif(blank in (4,8)):
		return ['right','up','down']
	elif(blank in (5,6,9,10)):
		return ['left','right','up','down']
	elif(blank in (7,11)):
		return ['left','up','down']
	elif(blank==12):
		return ['right','up']
	elif(blank in (13,14)):
		return ['left','right','up']
	elif(blank==15):
		return ['left','up']

def takeActionF_15p(startState, action):
	cp=copy.copy(startState)
	blank=findBlank(cp)
	if(action=='down'):
		cp[blank], cp[blank+4] = cp[blank+4], cp[blank] #move up 4 in the array to simulate moving down on a 4x4 matrix
	elif(action=='up'):
		cp[blank], cp[blank-4] = cp[blank-4], cp[blank]
	elif(action=='left'):
		cp[blank], cp[blank-1] = cp[blank-1], cp[blank]
	elif(action=='right'):
		cp[blank], cp[blank+1] = cp[blank+1], cp[blank]
	return cp

def countInversions(state):
    inversionCount=0
    for i in range(len(state)-1):
        if(state[i] !=0 and state[i+1] !=0 ):
            if(state[i]>state[i+1]): #an inversion is defined as a number in an array being greater than the next number.  
                inversionCount+=1
    return inversionCount

def getRow(state):
    blank=findBlank(state)
    return math.floor(blank/4)

def solveable(state):
    count=countInversions(state)
    row=getRow(state)
    if(count%2==1 and (row==1 or row==3)):#the link above shows specific states where a 15 puzzle is solvable
        return True
    elif(count%2==0 and (row==0 or row==2)):
        return True
    else:
        return False


startState15 = [1, 2, 6, 3,4, 9, 5, 7, 8, 13, 11, 15,12, 14, 0, 10]
goalState15 = [0, 1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15]

path = iterativeDeepeningSearch(startState15, goalState15, actionsF_15p, takeActionF_15p, 25)

print("Path from " +str(startState15) + " to "+ str(goalState15) +":")
for p in path:
    print(p)
    
print()
startState15 = [0, 1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15]
path = iterativeDeepeningSearch(startState15, goalState15, actionsF_15p, takeActionF_15p, 25)

print("Path from " +str(startState15) + " to "+ str(goalState15) +":")
for p in path:
    print(p)


startState15 = [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15]
isSolveable=solveable(startState15)

print()
print("Is Solvable?")
print(isSolveable)

path = iterativeDeepeningSearch(startState15, goalState15, actionsF_15p, takeActionF_15p, 25)

print("Path from " +str(startState15) + " to "+ str(goalState15) +":")
for p in path:
    print(p)


# # Towers of Hanoi (attempt)
# 
# I wasn't able to get this working correctly, but I enjoyed it nonetheless.  I felt like I was pretty close to solving this algorithm.  
# 
# 
# ## Functions
#   #### depthLimitedSearch_hanoi
#    A slight modification of the main depthLimitedSearch function that compares a multidimensional numpy array for all correct values.  I needed this to compare the state and the goalState
#    
#   #### iterativeDeepeningSearch_hanoi
#   This was the same as the original iterativeDeepeningSearch, I needed it to call depthLimitedSearch_hanoi
#   
#   ##### actions_hanoi
#   I was able to implement a generator correctly here, which I was happy about.  Logic: Get the top value from each tower (if it has one), if its less than the value at the top of either of the other two towers, you can move it there.  First value in the response is the origin, second is the destination.  one_two goes from tower one to two.
#   
#   #### take_action_hanoi(startState, action)
#   Pop from one list and put into another and return.  
#   

# In[1801]:


def depthLimitedSearch_hanoi(state, goalState, actionsF, takeActionF, depthLimit):
    if((state == goalState).all()):
        return []
    if(depthLimit==0):
        return 'cutoff'
    cutoffOccurred = False

    for action in actionsF(state):
        childState = takeActionF(state, action)
        result=depthLimitedSearch_(childState, goalState, actionsF, takeActionF, depthLimit-1)
        if(result=='cutoff'):
            cutoffOccurred = True
        elif(result!='failure'):
            result.insert(0, childState)
            return result
    if(cutoffOccurred):
        return 'cutoff'
    else:
        return 'failure'



def iterativeDeepeningSearch_hanoi(startState, goalState, actionsF, takeActionF, maxDepth):
    for depth in range(maxDepth):
        result = depthLimitedSearch_hanoi(startState, goalState, actionsF, takeActionF, depth)
        if(result=='failure'):
            return 'failure'
        if(result!='cutoff'):
            result.insert(0, startState)   
            return result
    return 'cutoff'




def actions_hanoi(state):
    actions=[]
    i=state[0]
    if(i):
        a=i[-1]
    else:
        a=4
    j=state[1]
    if(j):
        b=j[-1]
    else:
        b=4
    k=state[2]
    if(k):
        c=k[-1]
    else:
        c=4
    if(a<b):
        yield 'one_two'
    if(a<c):
        yield 'one_three'
    if(b<a):
        yield 'two_one'
    if(b<c):
        yield 'two_three'
    if(c<a):
        yield 'three_one'
    if(c<b):
        yield 'three_two'

def take_action_hanoi(startState, action):
    try:
        cp=copy.copy(startState)
        if(action=='one_two'):
            pop=cp[0].pop(-1)
            cp[1].append(pop)
        if(action=='one_three'):
            pop=cp[0].pop(-1)
            cp[2].append(pop)
        if(action=='two_one'):
            pop=cp[1].pop(-1)
            cp[0].append(pop)
        if(action=='two_three'):
            pop=cp[1].pop(-1)
            cp[2].append(pop)
        if(action=='three_one'):
            
            pop=cp[2].pop(-1)
            cp[0].append(pop)
        if(action=='three_two'):
            pop=cp[2].pop(-1)
            cp[1].append(pop)
        return cp
    except:
        
        return cp


HanoiStart=np.array([[3,2,1],[],[]])
HanoiEnd=np.array([[],[3,2,1],[]])

path = iterativeDeepeningSearch_hanoi(HanoiStart, HanoiEnd, actions_hanoi, take_action_hanoi,8)
print(path)

