#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Solution to the Towers of Hanoi Puzzle
# 
# ## Name: Matt Gorbett

# This assignment for me was a lesson in keeping an eye on the details.  I had several bugs throughout the project, including:
# 
# -  I needed to include only one valid winner ([[], [], [1, 2, 3]]) so that the length of Q had the right number of options.  This one took me the longest to figure out.  
# -  I commented out the greedy statement in my epsilon greedy statement so as to debug and train on random, and forgot to uncomment it out for awhile.  
# -  During the extra credit, I didn't update the validMoves function in the greedy epsilon function, so the training wouldn't solve the function.  
# -  Finally, adding a -1 to the Q difference times the learning rate was tricky.   
# 
# This was a very good assignment.  I feel like I learned a lot this way by getting to implement this algorithm.  A few notes on each function below:
# 
# #### validMoves 
# I used the same function from a previous assignments extra credit.  This is definitely not the best way to write this method, but it works.  
# 
# #### winner(state)
# I needed to switch the function from return state==[[], [], [1, 2, 3]] OR state==[[], [1, 2, 3], []] to  state==[[], [], [1, 2, 3]].  This helped me get the length of Q.  
# 
# #### makeMove(state, move)
# This is the same function for a 4 and 3 peg tower.  It is a simple 4 line function that is clear and concise.  
# 
# #### theTupler(state, move)
# This was a fun one to figure out.  I neglected to read the directions on writing this function and was having a hard time writing a hashable list to convert into a dictionary key.  I eventually figured it out on my own.  
# 
# #### epsilonGreedy(epsilon, Q, towers, validMovesF)
# 
# #### trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF,makeMoveF,train4=False,printMoves=False)
# train4 and print are automatically false unless otherwise stated.  For my extra credit, I added the ability to use this function to train on 4 pegs instead of 3.  Otherwise, setting the Q value to learningRate*(-1+Q[theTupler(towers, move)]-Q[theTupler(towersOld, moveOld)] was the trickiest part of training. 
# 
# ####  testQ(Q, maxSteps, validMovesF, makeMoveF,train4=False)
# Test also has the ability to take in a 4 peg tower.  The biggest line in this code is         move=moves[np.argmax(np.array([Q.get(theTupler(towers, m), 0) for m in moves]))]
# Q holds all the information on where is the smartest place to go.  The greatest value is where it should go.  
# 
# #### printState(state)
# This function is more or less hardcoded.  I print out the value of the list if it is there, unless I print blanks.  sys.stdout.write helped a lot instead of using print()
# 
# ## Extra Credit
# 
# #### validMoves_4disk(state)
# Same as the 3 peg function but accounting for an extra value.  
# 
# #### winner4(state)
# Same as 3 peg function but adding a 4 into the winning list. 
# 
# #### printState_4disk(state)
# Same as 3 peg function but with an extra set of lines.  
# 
# #### makeMove_4disk=makeMove
# 
# 
# # Functions below 

# In[135]:


import random 
from copy import deepcopy
import numpy as np
import copy
import sys

def validMoves(state):
    '''Used function from past assignments extra credit.  Valid returns list of where a state can move'''
    stateCopy=copy.copy(state)
    i=stateCopy[0]
    if(i):
        a=i[0]
    else:
        a=4
    j=stateCopy[1]
    if(j):
        b=j[0]
    else:
        b=4
    k=stateCopy[2]
    if(k):
        c=k[0]
    else:
        c=4
    valid=[]
    if(a<b):
        valid.append([1,2])
    if(a<c):
        valid.append([1,3])
    if(b<a):
        valid.append([2,1])
    if(b<c):
        valid.append([2,3])
    if(c<a):
        valid.append([3,1])
    if(c<b):
        valid.append([3,2])
    return valid
        

def winner(state):
    return state==[[], [], [1, 2, 3]]

    
def makeMove(state, move):
    newState=deepcopy(state)
    popped=newState[move[0]-1].pop(0)#pop value from list indicated in first move value
    newState[move[1]-1].insert(0,popped) #insert popped value at top of second move value
    return newState

def theTupler(state, move):
     return (tuple([tuple(state[0]),tuple(state[1]),tuple(state[2])]), tuple(move))#Needed this function to create a hashable list to use as the Q dictionary key

def epsilonGreedy(epsilon, Q, towers, validMovesF):
    moves = validMovesF(towers) #only difference in this function was getting a valid move list for towers of hanoi
    if np.random.uniform() < epsilon:
    	return random.choice(moves)
    else:
        # Greedy Move
        Qs = np.array([Q.get(theTupler(towers, m), 0) for m in moves])
        return moves[np.argmax(Qs)]

def trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF,makeMoveF,train4=False,printMoves=False):
    '''This function will train 3 or 4 pegs.  If train4 gets passed in as True, it will update the towers variable, winners 
    function, and pass in a different validMovesF function.  '''
    outcomes = np.zeros(nRepetitions)
    epsilon=1.0
    Q = {}
    showMoves=printMoves
    for nGames in range(nRepetitions):
        step = 0
        if(train4):
            towers = [[1, 2, 3,4], [], []]
        else:
            towers = [[1, 2, 3], [], []]
        done = False

        epsilon *= epsilonDecayFactor
        while not done: 
            step += 1

            move = epsilonGreedy(epsilon, Q, towers, validMovesF)
            towersNew = deepcopy(towers)

            towersNew=makeMove(towersNew, move)

            if theTupler(towers, move) not in Q:
                Q[theTupler(towers, move)] = 0  # initial Q value for new board,move
            if(showMoves):
                  printState(towersNew)
	    
            if(train4):
                winnerFunc=winner4
            else:
                winnerFunc=winner
            if winnerFunc(towersNew):
                if(showMoves):
                      printState(towersNew)
                Q[theTupler(towers, move)] = 1
                done = True
                outcomes[nGames] = step

            if step > 1:
                Q[theTupler(towersOld, moveOld)] += learningRate*(-1+Q[theTupler(towers, move)]-Q[theTupler(towersOld, moveOld)])

            towersOld, moveOld = towers, move # remember board and move to Q(board,move) can be updated after next steps
            towers = towersNew
    return Q, outcomes

def testQ(Q, maxSteps, validMovesF, makeMoveF,train4=False): 
    if(train4):
        towers = [[1, 2, 3,4], [], []]
    else:
        towers = [[1, 2, 3], [], []]
    done = False
    states=[]
    for step in range(maxSteps):
        
        states.append(towers)
        moves = validMovesF(towers)
        move=moves[np.argmax(np.array([Q.get(theTupler(towers, m), 0) for m in moves]))]
        towers=makeMoveF(towers, move)
        if(train4):
            winnerFunc=winner4
        else:
            winnerFunc=winner
        if winnerFunc(towers):
            states.append(towers)
            return states


def printState(state):
	sys.stdout.write((str(state[0][2])+" ") if len(state[0])==3 else "  ")
	sys.stdout.write((str(state[1][2])+" ") if len(state[1])==3 else "  ")
	sys.stdout.write((str(state[2][2])+" ") if len(state[2])==3 else "  ")
	sys.stdout.write("\n")	
	sys.stdout.write((str(state[0][1])+" ") if len(state[0])>=2 else "  ")
	sys.stdout.write((str(state[1][1])+" ") if len(state[1])>=2 else "  ")
	sys.stdout.write((str(state[2][1])+" ") if len(state[2])>=2 else "  ")
	sys.stdout.write("\n")	
	sys.stdout.write((str(state[0][0])+" ") if len(state[0])>=1 else "  ")
	sys.stdout.write((str(state[1][0])+" ") if len(state[1])>=1 else "  ")
	sys.stdout.write((str(state[2][0])+" ") if len(state[2])>=1 else "  ")
	sys.stdout.write("\n------\n")


# ## Grading

# Download and extract `A4grader.py` from [A4grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A4grader.tar).

# In[136]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# ## Extra Credit and example 3 peg run

# Modify your code to solve the Towers of Hanoi puzzle with 4 disks instead of 3.  Name your functions
# 
#     - printState_4disk
#     - validMoves_4disk
#     - makeMove_4disk
# 
# Find values for number of repetitions, learning rate, and epsilon decay factor for which trainQ learns a Q function that testQ can use to find the shortest solution path.  Include the output from the successful calls to trainQ and testQ.

# In[137]:


def validMoves_4disk(state):

    actions=[]
    stateCopy=copy.copy(state)
    i=stateCopy[0]
    if(i):
        a=i[0]
    else:
        a=5
        
    j=stateCopy[1]
    if(j):
        b=j[0]
    else:
        b=5
        
    k=stateCopy[2]
    if(k):
        c=k[0]
    else:
        c=5

    valid=[]


    if(a<b):
        valid.append([1,2])
    if(a<c):
        valid.append([1,3])
    if(b<a):
        valid.append([2,1])
    if(b<c):
        valid.append([2,3])
    if(c<a):
        valid.append([3,1])
    if(c<b):
        valid.append([3,2])
    return valid






def winner4(state):
    return  state==[[], [], [1, 2, 3, 4]]


def printState_4disk(state):
	sys.stdout.write((str(state[0][3])+" ") if len(state[0])==4 else "  ")
	sys.stdout.write((str(state[1][3])+" ") if len(state[1])==4 else "  ")
	sys.stdout.write((str(state[2][3])+" ") if len(state[2])==4 else "  ")
	sys.stdout.write("\n")	    
	sys.stdout.write((str(state[0][2])+" ") if len(state[0])>=3 else "  ")
	sys.stdout.write((str(state[1][2])+" ") if len(state[1])>=3 else "  ")
	sys.stdout.write((str(state[2][2])+" ") if len(state[2])>=3 else "  ")
	sys.stdout.write("\n")	
	sys.stdout.write((str(state[0][1])+" ") if len(state[0])>=2 else "  ")
	sys.stdout.write((str(state[1][1])+" ") if len(state[1])>=2 else "  ")
	sys.stdout.write((str(state[2][1])+" ") if len(state[2])>=2 else "  ")
	sys.stdout.write("\n")	   
	sys.stdout.write((str(state[0][0])+" ") if len(state[0])>=1 else "  ")
	sys.stdout.write((str(state[1][0])+" ") if len(state[1])>=1 else "  ")
	sys.stdout.write((str(state[2][0])+" ") if len(state[2])>=1 else "  ")
	sys.stdout.write("\n------\n")

    

print("Three disk test:")    
#3 disk test
Q, stepsToGoal = trainQ(1000, 0.5, 0.7, validMoves, makeMove)
path=testQ(Q, 20, validMoves, makeMove)
for s in path:
    printState(s)
print("\n\nFour disk test:")
# 4 disk test
makeMove_4disk=makeMove
Q, stepsToGoal=trainQ(500, 0.5, .7, validMoves_4disk, makeMove_4disk, train4=True)
path=testQ(Q, 20, validMoves_4disk, makeMove_4disk, train4=True)
for p in path:
    printState_4disk(p)
print("Length of 4 disk test run: "+str(len(path)))

