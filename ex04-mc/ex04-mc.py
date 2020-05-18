import gym
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


n_episodes = 10000


def getValueFunction(rewardList):
    stateValueFunction = np.zeros((2,10,10))
    for x in range(0,2):
        for y in range(0,10):
            for z in range(0,10):
                #if rewardList[x][y][z]:
                avReturn = sum(rewardList[x][y][z])/len(rewardList[x][y][z])
                stateValueFunction[x][y][z] = avReturn
                #else:
    return stateValueFunction
                    

def main(episodes):
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    
    # start monte carlo simulation
    
    
    returnOfStates = []
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
        
        ace = int(obs[2]) # save initial state
        pSum = obs[0]
        dSum = obs[1]
        
        while not done:
            # print("Episode: {} Index: [{}][{}][{}] Values: {} {} {}".format(episodeCounter,ace,pSum-12,dSum-1,obs[2],obs[0],obs[1])) 
            #print("observation:", obs)
            
            if obs[0] >= 20:
                #print("stick")
                obs, reward, done, _ = env.step(0)
            else:
                #print("hit")
                obs, reward, done, _ = env.step(1)
            
            if pSum >= 12:
                returnOfStates[ace][pSum-12][dSum-1].append(reward)
                # print("Episode: {} Index: [{}][{}][{}] Values: {} {} {}".format(episodeCounter,int(obs[2]),obs[0]-12,obs[1]-1,obs[2],obs[0],obs[1])) 
            
            ace = int(obs[2]) # save initial state
            pSum = obs[0]
            dSum = obs[1]
        
            #print("obs:", obs)
            #print("reward:", reward)
            #print("")
        episodeCounter+=1
    print(returnOfStates)
    valueFunction = getValueFunction(returnOfStates)
    print(valueFunction)
        
    

if __name__ == "__main__":
    main(n_episodes)
