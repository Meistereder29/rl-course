import gym
import numpy as np
import matplotlib.pyplot as plt
 

# parameters
intervalls = 20   
n_episodes = 5000
#epsilon = 0.1

def getState(observation):
    # converting input to integer to avoid floating point error
    xt = int(observation[0]*10)
    xt_p = int(observation[1]*100)
    state_pos = int(np.round((xt+12)*19/18))
    state_vel = int(np.round((xt_p+7)*19/14))
    state = state_pos * 20 + state_vel
    return state

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    init = True
    while True:
        env.render()
        
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def epsilonGreedy(Q,state,epsilon): # epsilon-greedy
    coin = np.random.rand()

    if coin < epsilon:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(Q[state][:])
    return action

def run(env):
    
    Q = np.random.rand(400,3)
    e = np.ones([400,3])
    state_dist = np.zeros([400,1])
    epsilon = 0.05 # use decaying value
    gamma = 0.9
    alpha = 0.6
    lamb = 0.5
    render = False
    for i in range(n_episodes):
        
        # print episode and reder 
        if i % 100 == 0:
            print("Episode {}".format(i))
            
        if i % 1000 == 0 and i >0:
            render = True
        else:
            render = False
        
        # init
        observation = env.reset()
        state = getState(observation)
        action = 1
        done = False
        step = 0
        #if epsilon > 0.1:
        #    epsilon = epsilon - 0.001
        while not done:
            #print(action)
            if render:
                env.render()
            observation, reward, done, info = env.step(action)
            #print("Step = {}".format(step))
            state_prime = getState(observation)
            a_prime = epsilonGreedy(Q,state_prime,epsilon)
            a_star = np.argmax(Q[state][:])
            delta = reward + gamma * Q[state_prime][a_star] - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * delta
            #e[state][action] = e[state][action] + 1
            
            # for s in range(20):
            #     for a in range(3):
            #         Q[s][a] = Q[s][a] + alpha * delta * e[s][a]
            #         if a_prime == a_star:
            #             e[state][action] = gamma*lamb*e[state][action]
            #         else:
            #             e[state][action] = 1
            state = state_prime
            state_dist[state]+=1
            action = a_prime
            step += 1
        render = False
    
    print(Q)
    print(state_dist[state_dist !=0])
    
def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    #random_episode(env)
    run(env)
    env.close()
    
    # observation = np.array([-1.2,-0.07])
    # while observation[0] <= 0.61:
    #     while observation[1] <= 0.07:
    #         state = getState(observation)
    #         print("Observation = {} State = {}".format(observation,state))
    #         observation[1]+=(0.14/19)
    #     observation[1] = -0.07
    #     observation[0] += 1.8/19
    # print(observation[0] - (1.8/19))

if __name__ == "__main__":
    main()
