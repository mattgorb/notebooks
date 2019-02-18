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

times = [' 9 am',
         '10 am',
         '11 am',
         '12 pm',
         ' 1 pm',
         ' 2 pm',
         ' 3 pm',
         ' 4 pm']

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
    
#print(neighbors)

#print(classes_domain)

vars=classes

solution, steps = min_conflicts(vars, classes_domain, constraints_ok, neighbors,6000)

import pandas as pd

def display(solution):
    dis=pd.DataFrame(columns = classrooms, index=[i for i in times],data='')

    #transform=dict((v,k) for k,v in solution.items())

    for key, value in solution.items():
        dis.loc[value[1], value[0]]=key

    for index, row in dis.iterrows():
        val=dis
    print(dis)
    print(classes)
    #r=[]
    #c=[]
    #for k,v in solution.items():

    #print(r)

    #pd.DataFrame({t for t in solution.iter()})


#print(solution)
#print(steps)

display(solution)

                
