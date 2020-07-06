import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    z = state.dot(theta)
    exp = np.exp(z)
    return exp/np.sum(exp)



def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
            
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    alpha = 0.0025
    gamma = 0.99
    last_100_episodes = []
    n_episodes = 1200
    for e in range(n_episodes):
        if e % 100 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        
        # TODO: keep track of previous 100 episode lengths and compute mean
        if e < 100:
            last_100_episodes.append(sum(rewards))
        else:
            last_100_episodes.append(sum(rewards))
            last_100_episodes.pop(0)
        mean = np.mean(last_100_episodes)
        #if e % 200 == 0:
           # alpha = alpha/10
        if e % 100 == 0:
            print("episode: " + str(e) + " length: " + str(len(states)) + " Mean of last 100 episodes: " + str(mean)) 
        steps = len(states)
        # TODO: implement the reinforce algorithm to improve the policy weights

        # calculate Gt
        G_t = np.zeros([steps])
        for t in range(steps):
            for k in range(t+1,steps+1):
                G_t[t] += np.power(gamma,k-t-1) * rewards[k-1]
            pi = policy(states[t], theta)
            # update only theta of taken action A_t
            action = actions[t]
            dpi = softmax_grad(pi)[action,:]
            dlog = dpi[action] / pi[action]
            theta[:,action] = theta[:,action] + alpha * np.power(gamma, t) * G_t[t] * (states[t] * (1 - pi[action]))
            if np.isnan(theta).any():
                print("ALARM! ALAAAARM!!")
        debug = True
def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
