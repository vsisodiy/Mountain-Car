from environment import MountainCar
import numpy as np
import sys

mode = sys.argv[1]
mc = MountainCar(mode)


if mode == "raw":
    weight = np.zeros((mc.state_space, mc.action_space))
    b = 0
if mode == "tile":
    weight = np.zeros((mc.state_space, mc.action_space))
    b = 0


def vectorize_state(state):
    state_vector = np.zeros(mc.state_space)
    for k in state:
        state_vector[k] = state[k]
    return state_vector.reshape(-1,1)

def action_selection(state, weight, b):
    state_vector = vectorize_state(state)
    Q = np.dot(state_vector.T, weight) + b
    return np.argmax(Q), np.max(Q)


def q_learning(env,episodes,episode_length,epsilon,lr,disc_factor, weight, b):

    Reward_list = []
    for i in range(0, episodes):
        state = env.reset()
        reward = 0

        for j in range(0, episode_length):
            explore = np.random.uniform(0, 1) < epsilon
            if explore:
                action = np.random.randint(0, env.action_space)
            else:
                action = action_selection(state, weight, b)[0]

            next_state, r, has_ended = env.step(action)
            curr_st_vector = vectorize_state(state)
            q_val = np.dot(curr_st_vector.T, weight[:, action].reshape(-1,1))[0][0] + b
            best_next_q = action_selection(next_state, weight, b)[1]
            td_tar = r + (disc_factor * best_next_q)
            grad_w = np.zeros((mc.state_space, mc.action_space))
            grad_w[:,action] = curr_st_vector.reshape(1,-1)
            weight -= lr * (q_val - td_tar) * grad_w
            b -= lr * (q_val - td_tar)
            reward += r
            state = next_state
            
            if has_ended:
                break
        
        Reward_list.append(reward)
    return Reward_list, weight, b

Reward_list, weight1, b1 = q_learning(mc,int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),float(sys.argv[8]),float(sys.argv[7]), weight, b)


file5 = open(sys.argv[3], "w")
for i in range(len(Reward_list)):
    file5.writelines(str(Reward_list[i]) + "\n")
file5.close()

file6 = open(sys.argv[2], "w")
file6.writelines(str(b1) + "\n")
for i in range(weight.shape[0]):
    for j in range(weight1.shape[1]):
        file6.writelines(str(weight1[i][j]) + "\n")
file6.close()






