import numpy as np
import random
import matplotlib.pyplot as plt

#initialising points and lists
global rewards, qValues
rewards = []
qValues = []
points = 0

"""
def createGrid()

Function that creates the domain, using a list, and initialises the qValues, as a list of lists.
It returns the starting position of the Agent.
"""

def createGrid():
    global initialList
    #Fill the list
    initialList = [-1,1,-1,0,0,0,0,0,0,1,2,1,-1,0,1,0,-1,0,-1,0]
    qValues = []
    #create a list of lists of random numbers
    for i in range(20):
        qValues.append([random.random(),random.random(),random.random(), random.random()])
    startPosition = initialList.index(0)
    #get the initial position
    initialList[startPosition] = 100
    #reshape the list into a 2d array.
    grid = np.reshape(initialList, (4,-6))

    return startPosition


"""
def env_move_det(s, a)

Function that takes a state and an action,
and computes in a deterministic way the next action.
Returns the next action.
"""

def env_move_det(s, a):

    global initialList
    #get the previous state
    previousState = s

    #update the index at every move
    if a == 0 and (s >=0 and s <= 19):
        s -= 5
        #print("The index of my new position is: "+str(s))
    elif a == 1 and (s >=0 and s <= 19):
        s += 5
        #print("The index of my new position is: "+str(s))
    elif a == 2 and (s >=0 and s <= 19) :
        s += 1
        #print("The index of my new position is: "+str(s))
    elif a == 3 and (s >=0 and s <= 19):
        s -= 1
        #print("The index of my new position is: "+str(s))
    else:
        pass

    #check for the boundaries of the world
    if s > 19 or s < 0:
        s = previousState
        return s


    return s

"""
def env_move_sto(s, a)

Function that takes a state and an action,
and computes in a stochastic way the next action.
Returns the next action
"""

def env_move_sto(s, a):

    #create a random number
    prob = random.random()
    #get the previous state
    previousState = s

    #compute the next action ortogonal to the action
    if a == 0:
        if prob < 0.8:
            s -= 5
        elif prob >= 0.8 and prob < 0.9:
            s -=1
        elif prob > 0.9:
            s += 1
    elif a == 1:
        if prob < 0.8:
            s += 5
        elif prob >= 0.8 and prob < 0.9:
            s +=1
        elif prob > 0.9:
            s -= 1
    elif a == 2:
        if prob < 0.8:
            s += 1
        elif prob >= 0.8 and prob < 0.9:
            s -= 5
        elif prob > 0.9:
            s += 5
    elif a == 3:
        if prob < 0.8:
            s -= 1
        elif prob >= 0.8 and prob < 0.9:
            s += 5
        elif prob >= 0.9:
            s -= 5

    #checking for boundaries
    if s > 19 or s < 0:
        s = previousState
        return s


    return s

"""
def env_reward(s, a, next_s)

Function that computes the reward for the next state.
Return the points for each reward.
"""

def env_reward(s, a, next_s):

    global initialList
    #initialise the points to zero
    points = 0

    #checking if the number at the given index is a reward, and if the next state is not equal to the current one
    if initialList[next_s] == 0 and (s != next_s):
        points += 0
        return points
    elif initialList[next_s] == 1 and (s != next_s):
        points += 5
        return points
    elif initialList[next_s] == -1 and (s != next_s):
        points -= 5
        return points
    elif initialList[next_s] == 2 and (s != next_s):
        points += 100
        return points

    points += 0
    return points


"""
def agt_choose(s, epsilon)

Function that uses epsilon greedy to choose the next action.
If the random number in greater then 1 - epsilon, get the index of the highest number
and transform it into an action.
If not, select a random index and transform it into an action.

Returns the action 
"""

def agt_choose(s, epsilon):

    global qValues
    #create a random number
    prob = random.random()

    #if 1 - epsilon is greather or equal then the random number
    if prob <= 1 - epsilon:
        #get the index of the biggest values in the given state
        action = np.argmax(qValues[s])
        if action == 0:
            return 0
        elif action == 1:
            return 1
        elif action == 2:
            return 2
        elif action == 3:
            return 3
    #otherwise, pick a random one.
    else:
        action = random.randint(0,3)
        if action == 0:
            return 0
        elif action == 1:
            return 1
        elif action == 2:
            return 2
        elif action == 3:
            return 3

"""
def agt_learn_sarsa(alpha, s, a, r, next_s, next_a)

Function that integrates the sarsa algorithm, and updates the value of qValues.
"""

def agt_learn_sarsa(alpha, s, a, r, next_s, next_a):

    global qValues, gamma

    #converting the actions, from string to int
    if next_a == "up":
        next_a = 0
    elif next_a == "down":
        next_a = 1
    elif next_a == "right":
        next_a = 2
    elif next_a == "left":
        next_a = 3
    #applying the Sarsa algorithm
    return (1-alpha) * qValues[s][a] + alpha *(r + gamma * qValues[next_s][next_a])

"""
def agt_learn_q(alpha, s, a, r, next_s)

Function that integrates the q algorithm, and updates the value of qValues
"""

def agt_learn_q(alpha, s, a, r, next_s):

    global qValues, gamma

    #applying the q algorithm
    return (1- alpha) * qValues[s][a] + alpha * (r+ gamma * np.argmax(qValues[next_s]))

"""
def agt_learn_final(alpha, s, a, r)

Function that returns 0 when the agent reaches the goal state
"""

def agt_learn_final(alpha, s, a, r):

    global qValues
    #update qValues
    return (1-alpha) * qValues[s][a] + alpha * r

"""
def agt_reset_value()

Function that resets the values of qValues
"""

def agt_reset_value():

    global qValues
    qValues = []
    #reset all the values
    for i in range(20):
        qValues.append([random.random(),random.random(),random.random(),random.random()])

"""
def testing()

Function that tests the different functions within a given number of EPOCHS, EPISODES and T
"""
def testing():

    global rewards, initialList, qValues, gamma
    #initialise all the values
    EPOCHS = 500
    EPISODES = 500
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.99
    rewards = []

    #filling the reward list with zeroes.
    for i in range(1, EPISODES +1):
        rewards.append(0)
    #looping trough the EPOCHS
    for epoch in range(EPOCHS):
        agt_reset_value()
        #looping trough the EPISODES
        for episode in range(EPISODES):
            #check if learning
            learning = episode < EPISODES - 50
            if learning:
                eps = epsilon
            else:
                eps = 0
            cumulative_gamma = 1
            s = createGrid()
            a = agt_choose(s,eps)
            #looping trough T
            for timestep in range(2*20):
                next_s = env_move_det(s, a)
                r = env_reward(s, a, next_s)
                #update the value of reward at the given index
                rewards[episode] += ((cumulative_gamma *r ) / EPOCHS)
                cumulative_gamma *= gamma
                next_a = agt_choose(next_s, eps)
                #if learning
                if (learning):
                    #check if the agent is in the absorbing state, or if the loop finishes
                    if (initialList[next_s] == initialList.index(2)) or (timestep == (2*20)-1):
                        #set value to 0
                        qValues[s][a] = agt_learn_final(alpha,s,a,r)
                    else:
                        #apply the q algorithm
                       	qValues[s][a] = agt_learn_q(alpha, s, a, r, next_s)
                a = next_a
                s = next_s
    #plot into a graph
    plt.plot(rewards)
    #show the graph
    plt.show()

testing()