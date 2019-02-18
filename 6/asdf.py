import numpy as np
import math
import operator
import random
from matplotlib import pyplot as plt
#reinforement learning with the following solution strategies:
#random
#greedy
#epsilon greedy
#upper confidence bound
#thompson sampling bernoulli
#thompson sampling binomial
random.seed()

#10 armed bandit
num_bandits=5
mab=np.random.rand(num_bandits)
print(str(num_bandits)+" armed bandit probabilities")
print(mab)

#number of pulls in game
maxsteps=500


def greedy():
    best=max(Q.items(), key=operator.itemgetter(1))
    return best[0]

def random_selection():
    return random.randint(1, num_bandits)

def epsilonGreedy():
    if np.random.uniform() < epsilon:
        # Random Move
        return random_selection()
    else:
        # Greedy Move
        return greedy()

#random 
print("\nRandom Action")
Q={i:0 for i in range(1,num_bandits+1)}
action_attempts={i:0 for i in range(1,num_bandits+1)}
score=0
rando=[]
for step in range(maxsteps):
    selection=random_selection()
    action_attempts[selection]+=1    
    reward=0
    if(random.uniform(0, 1)<mab[selection-1]):
        reward=1
        score+=1
    Q[selection]+=((1/action_attempts[selection])*(reward-Q[selection]))
    rando.append(score)
print(Q)
print(score)



#greedy
print("\nGreedy Action")
Q={i:0 for i in range(1,num_bandits+1)}
action_attempts={i:0 for i in range(1,num_bandits+1)}
score=0
gre=[]
for step in range(maxsteps):
    for i in range(1,num_bandits+1):
        reward=0
        if(random.uniform(0, 1)<mab[i-1]):
            reward=1
        action_attempts[i]+=1  
        Q[i]+=((1/action_attempts[i])*(reward-Q[i]))
    selection=greedy()
    if(random.uniform(0, 1)<mab[selection-1]):
        score+=1
    gre.append(score)
print(Q)
print(score)



#epsilon greedy no decay
print("\nEpsilon Greedy No Decay")
epsilon=.5
Q={i:0 for i in range(1,num_bandits+1)}
action_attempts={i:0 for i in range(1,num_bandits+1)}
score=0
learningRate=0.2
ep=[]
for step in range(maxsteps):
    selection=epsilonGreedy()
    action_attempts[selection]+=1
    #print(selection)
    reward=0
    if(random.uniform(0, 1)<mab[selection-1]):
        reward=1
        score+=1
    Q[selection]+=((1/action_attempts[selection])*(reward-Q[selection]))
    ep.append(score)
print(Q)
print(score)




#epsilon greedy with decay
epsilon=1
decayRate=.9
print("\nEpsilon Greedy with Decay")
Q={i:0 for i in range(1,num_bandits+1)}
action_attempts={i:0 for i in range(1,num_bandits+1)}
score=0
ep_dec=[]
for step in range(maxsteps):
    epsilon*=decayRate
    selection=epsilonGreedy()
    action_attempts[selection]+=1
    reward=0
    if(random.uniform(0, 1)<mab[selection-1]):
        reward=1
        score+=1
    Q[selection]+=((1/action_attempts[selection])*(reward-Q[selection]))
    ep_dec.append(score)
print(Q)
print(score)


#Thompson sampling
alpha=[1]*num_bandits
beta=[1]*num_bandits
print("\nThompson Sampling")
score=0
ts=[]
for step in range(maxsteps):
    sample=[np.random.beta(alpha[x],beta[x]) for x in range(num_bandits)]
    selection=sample.index(max(sample))
    reward=0
    if(random.uniform(0, 1)<mab[selection-1]):
        reward=1
        score+=1
    alpha[selection]+=reward
    beta[selection]+=(1-reward)
    ts.append(score)
print(score)




#UCB
print("\nUCB")
ucb=[]
probs=[1]*num_bandits
action_attempts={i:0 for i in range(1,num_bandits+1)}
score=0
np.seterr(divide = 'ignore') 
for step in range(maxsteps):
    sample=[probs[x]+np.sqrt(np.log(step)/(1+action_attempts[x+1])) for x in range(num_bandits)]
    selection=sample.index(max(sample))
    action_attempts[selection+1]+=1
    reward=0
    if(random.uniform(0, 1)<mab[selection]):
        reward=1
        score+=1
    probs[selection]+=((1/action_attempts[selection+1])*(reward-probs[selection]))
    ucb.append(score)
print(probs)
print(score)




X = np.arange(maxsteps)
figure = plt.figure()
tick_plot = figure.add_subplot(1, 1, 1)
tick_plot.plot(X, ucb,  color='blue', linestyle='-',label='UCB')
tick_plot.plot(X, rando,  color='green', linestyle='-',label='Random')
tick_plot.plot(X, ts,  color='red', linestyle='-',label='Thompson Sampling')
tick_plot.plot(X, ep_dec,  color='yellow', linestyle='-',label='Epsilon Greedy')
tick_plot.plot(X, ep,  color='orange', linestyle='-',label='Epsilon')
tick_plot.plot(X, gre,  color='purple', linestyle='-',label='Greedy')
plt.title("Scores of different RL Policies")
plt.legend(loc='upper left')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.show()
