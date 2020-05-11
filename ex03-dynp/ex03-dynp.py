import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    delta = 0
    
    while True:
        
        for s in range(n_states):
            v = V_states[s]
            V_a = np.zeros(n_actions) 
            for a in range(n_actions):
                transition_matrix = np.array(env.P[s][a])
                p_vector = transition_matrix[:,0]
                nstate_vector = transition_matrix[:,1]
                r_vector = transition_matrix[:,2]
                
                V_a[a] = 0
                for i in range(0,3):
                    index = nstate_vector[i]
                    V_a[a]+=p_vector[i] * (r_vector[i]) # + gamma*V_states[index]) 
                    # TODO: FIx that shit
                                                               
            a_max = np.argmax(V_a)
            V_states[s] = V_a[a_max]
            
            delta = np.max(delta,np.abs(v - V_states[s]))
    
        if (delta < theta):
            policy = np.zeros(n_states)
            for s in range(n_states):
                policy[s] = np.argmax(V_states)
    
    return policy
def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
