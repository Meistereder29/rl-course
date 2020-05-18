import gym
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


n_episodes = 500000


def getValueFunction(rewardList):
    stateValueFunction = np.zeros((2,10,10))
    for x in range(0,2):
        for y in range(0,10):
            for z in range(0,10):
                if rewardList[x][y][z]:
                    avReturn = sum(rewardList[x][y][z])/len(rewardList[x][y][z])
                    stateValueFunction[x][y][z] = avReturn
                else:
                    stateValueFunction[x][y][z] = 0
    return stateValueFunction
                    

def main(episodes):
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    
    # start monte carlo simulation
     
    returnOfStates = [] # create a list which stores the returns for every state. Dimension is 2x10x10 
    for x in range(0,2):
        returnOfStates.append([])
        for y in range(0,10):
            returnOfStates[x].append([])
            for z in range(0,10): 
                returnOfStates[x][y].append([])
    
    episodeCounter = 0
    while episodeCounter < episodes:
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        
        # getting indices to access state-value function
        index0 = int(obs[2]) # 0 or 1 for usable ace
        index1 = obs[0]-12   # value of player card
        index2 = obs[1]-1    # value of dealer card
        
        while not done:
            
            if obs[0] >= 20:
                #print("stick")
                obs, reward, done, _ = env.step(0)
            else:
                #print("hit")
                obs, reward, done, _ = env.step(1)
            
            if index1 >= 0: # skip states where player_sum is below 12
                returnOfStates[index0][index1][index2].append(reward)
            
            index0 = int(obs[2]) # update state
            index1 = obs[0]-12
            index2 = obs[1]-1
        
            
        if (episodeCounter % 50000 == 0):
            print(episodeCounter) # visualize simulation progress
        episodeCounter+=1
        
    valueFunction = getValueFunction(returnOfStates) # calculate value function from return list
    print(valueFunction)
    
    # plotting results 
    
    fig = plt.figure(figsize=plt.figaspect(1.2),dpi=200)
    x = np.arange(1,11,1)
    y = np.arange(12,22,1)
    x,y = np.meshgrid(x,y)
    
    ax = fig.add_subplot(2,1,1,projection='3d')
    aceFalse = valueFunction[0]
    ax.plot_wireframe(x,y,aceFalse)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax = fig.add_subplot(2,1,2,projection='3d')
    aceTrue = valueFunction[1]
    ax.plot_wireframe(x,y,aceTrue)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig("prediction_figure.pdf")
    
    
if __name__ == "__main__":
    main(n_episodes)
