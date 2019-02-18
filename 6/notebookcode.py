#!/usr/bin/env python
# coding: utf-8

# # Assignment 6: Min-Conflicts

# ## Matt Gorbett

# The min-conflicts algorithm is a search method for solving constraint satisfaction problems.  Below I will attempt to explain the provided algorithm in english, because I think this will help me understand it more.  This is a fun algorithm, I immediately envisioned a project where I could generate sports schedules or some comparable application.  
# 
# #### Attempting to explain the min_conflicts function and its calling functions 
# 
# For each computer science course, the function calls the min_conflicts_value function, which in turn calls the argmin_random_tie function.  The domain of the variable gets passed into argmin_random_tie, which is 3 classrooms * 8 times=24 total options for each course.  
# argmin_random_tie iterates through the domain of each course variable to determine which room/time combination has the least amount of conflicts.  To do this, it iterates through each room/time combo in the list and calls the nconflicts function.  The nconflicts function returns a 1 if the course/room/time combo has a conflict with another course/room/time combo in its domain list.  From here, it counts the number of 1's in the list and returns this value to argmin_random_tie.  argmin_random_tie keeps a variable of the lowest number of conflicts retrieved from nconflicts for each tuple in the list and sets the best tuple value based on its nconflicts value.  If there is a tie for lowest number of conflicts, it selects a random tuple value. From here, argmin_min_random_tie returns a tuple value to min_conflicts. 
# After this, min_conflicts iterates through for max_steps and finds a list of variables in the solution dictionary that are conflicted.  It takes a random variable from the list of conflicted variables, finds the variables value that has the lowest number of conflicts, and sets a new value equal to the value with the lowest number of conflicts.  It does this until either max_steps are reached or it has a dictionary with no conflicts.  From here, it returns the solution and the number of steps it took.  

# ## Function Notes

# #### assignments, steps = schedule(classes, times, rooms, max_steps)
# 
# A big part of this assignment was setting up the data correctly to pass into the already defined min_conflicts function.  To do this, I first iterated through each of the 3 lists: classes times, and rooms.  In this nested for loop, I add each class to a dictionary as the key if it doesn't exist.  For the value, I add a tuple in the form (classroom, time).  When the course key already exists, I append another tuple to the value with the new classroom and time. So the end variable is a dictionary with courses as the keys with lists of tuples that have classrooms and times.  Next, I get the neighbors of each class.  The neighbors for each course are simply the other courses.   Finally, I return the min_conflicts function with its results to get a schedule with steps.  
# 
# #### result = constraints_ok(class_name_1, value_1, class_name_2, value_2)
# This function compares two classes and their values.  First I check to make sure I'm not comparing two of the same classes and if I am I return false.  Next, I check whether the compared classes have the same time and same room, if they do return false.  Finally I compare the keys third index, which is the level of the course.  I get the third character in the key, and if the two are the same, and they have the same time, I will return false.  The one exception here is that I check if the classes are 163 and 164, and if they are I return true because this is ok.  
#      
#      
# #### display(assignments, rooms, times)
# This was a fun function to write.  The key for me was creating a pandas table and setting its columns to the classrooms and the index to the times.  From here, all I needed to do was iterate through the solution dictionary and find the tuple value in the pandas table using the pandas .loc[] method and setting the cell to the dictionary key (class).  I was happy with how simple this one turned out.  
# 
# 

# ## min-conflicts given functions

# In[149]:


import sys
import random

def min_conflicts(vars, domains, constraints, neighbors, max_steps=1000): 
    """Solve a CSP by stochastic hillclimbing on the number of conflicts."""
    # Generate a complete assignment for all vars (probably with conflicts)
    current = {}
    for var in vars:
        val = min_conflicts_value(var, current, domains, constraints, neighbors)

        current[var] = val
    # Now repeatedly choose a random conflicted variable and change it

    for i in range(max_steps):
        conflicted = conflicted_vars(current,vars,constraints,neighbors)
        if not conflicted:
            return (current,i)
        var = random.choice(conflicted)
        val = min_conflicts_value(var, current, domains, constraints, neighbors)
        current[var] = val
    return (None,None)

def min_conflicts_value(var, current, domains, constraints, neighbors):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(domains[var],
                             lambda val: nconflicts(var, val, current, constraints, neighbors)) 

def conflicted_vars(current,vars,constraints,neighbors):
    "Return a list of variables in current assignment that are in conflict"
    return [var for var in vars
            if nconflicts(var, current[var], current, constraints, neighbors) > 0]

def nconflicts(var, val, assignment, constraints, neighbors):
    "Return the number of conflicts var=val has with other variables."
    # Subclasses may implement this more efficiently
    def conflict(var2):
        val2 = assignment.get(var2, None)
        
        return val2 != None and not constraints(var, val, var2, val2)
    return len(list(filter(conflict, neighbors[var])))

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best


# # My functions

# In[150]:




times = ['9 am','10 am','11 am','12 pm','1 pm','2 pm','3 pm','4 pm']
classrooms = ['CSB 130','CSB 325','CSB 425']
classes=['CS160', 'CS163', 'CS164',
'CS220', 'CS270', 'CS253',
'CS320', 'CS314', 'CS356', 'CS370',
'CS410', 'CS414', 'CS420', 'CS430', 'CS440', 'CS445', 'CS453', 'CS464',
'CS510', 'CS514', 'CS535', 'CS540', 'CS545']

#classes_dict = {var: [(time for time in times,room for room in classrooms)] for var in classes}


def constraints_ok(key, val, key2, val2):
    if(key==key2):
        return False
    if(val[0]==val2[0] and val[1]==val2[1]):
        return False
    if(key[2]==key2[2] and val[1]==val2[1] ):
        if((key=='CS163' and  key2=='CS164') or (key=='CS164' and  key2=='CS163')):
            return True
        else:
            return False    

    return True

def display(solution,rooms, times):
    dis=pd.DataFrame(columns = rooms, index=[i for i in times],data='')
    for key, value in solution.items():
        dis.loc[value[1], value[0]]=key
    return dis

def schedule(classes, times, classrooms, max_steps):
    classes_domain={}
    for time in times:
        for room in classrooms:
            for c in classes:
                if c not in classes_domain:
                    classes_domain[c]=[(room,time)]
                else:
                    classes_domain[c].append((room,time))

    neighbors={}
    for key, value in classes_domain.items():
        neighbors[key]=[c for c in classes if c!=key]

    vars=classes
    return min_conflicts(vars, classes_domain, constraints_ok, neighbors,max_steps)
    




assignments, steps = schedule(classes, times, classrooms, 100)
print(display(assignments, classrooms, times))


def extra_credit(classes, times, classrooms, maxsteps):
    high=0
    for i in range(maxsteps):
        assignments, steps =schedule(classes, times, classrooms, 100)
        preference_count=0

        for key, value in assignments.items():
            if(value[1] not in ('9 am', '12 pm', '4 pm')):
               preference_count+=1
            if(key in ('CS163','CS164')and value[1] in ('1 pm','2 pm')):
               preference_count+=1

        if(preference_count>high):
               high=preference_count

               preferred_schedule=assignments

    return preferred_schedule,high
            
print("\n\n\nEXTRA CREDIT")
solution, high=extra_credit(classes, times, classrooms, 1000)    
print("Preferences met: "+str(high))
display=display(solution, classrooms, times)
print(display)


# In[151]:


max_steps = 100
assignments, steps = schedule(classes, times, rooms, max_steps)
print('Took', steps, 'steps')
print(assignments)


# In[152]:


display(assignments, rooms, times)


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A6.ipynb```.  So, for me it would be ```Anderson-A6.ipynb```.  Submit the file using the ```Assignment 6``` link on [Canvas](https://colostate.instructure.com/courses/68135).

# ## Grading
# 
# Download [A6grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A6grader.tar) and extract `A6grader.py` from it.  Grader will be available soon.

# In[ ]:


get_ipython().run_line_magic('run', '-i A6grader.py')


# # Extra Credit
# 
# Solve the scheduling problem again but with the addition of
# these preferences:
# 
#   * prefer schedules that do not schedule classes at 9 am, 12 pm or 4 pm.
#   * prefer schedules with CS163 and CS164 meeting at 1 pm or 2 pm.
#   
# 
# ## My solution:
# * My solution is above the grader, in my main chunk of code.  
# To do this, I called schedule 1000 times and counted how many of the preferences have been met.  Logically, the ideal scenario is obvious.  Since there are 23 classes and only 24 slots, all but one slot will be taken up.  This means that ideally 8/9 slots that cover 9,12,and 4 pm will have to be taken, with one slot open.  
# 
# Additioally, the ideal schedule will have both CS163 and CS164 scheduled at either 1 or 2 pm.  This has been reflected in my solution, with 17 preferences met.  
# 17 preferences explanation:
# It is preferred that classes meet at 10, 11, 1, 2, or 3.  This means that ideally all these slots will be full.  5*3 classrooms=15 preferences.  
# Cs 163 at 1 or 2=1 preference
# Cs 164 at 1 or 2=1 preference
# 15+1+1=17 preferences fulfilled.  
