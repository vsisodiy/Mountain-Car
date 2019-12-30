# Mountain-Car

In Mountain Car you control a car that starts at the bottom of a valley. Goal is to reach the top right. The state of the environment is represented by two variables, position and velocity. position can be between -1.2 and 0.6 inclusive and velocity can be between -0.07 and 0.07 inclusive. These are just measurements along the x-axis. The actions that you may take at any state are f0; 1; 2g which respectively correspond to (0) pushing the car left, (1) doing nothing, and (2) pushing the car right.

Implementation Details:

A learning agent interacts with the environment solely based on calls to step and reset methods of the environment. In Environment.py, we have two functions that we use in Q-learning apart from the setting the enviroment. They are:

* reset(): Reset the environment to starting conditions.

* step(action): Take a step in the environment with the given action. action must be either 0, 1 or 2. This will return a tuple of (state; reward; done) which is the next state, the reward observed, and a boolean indicating if you reached the goal or not, ending the episode. The state will be either a raw' or tile representation, as definedned below, depending on how you initialized Mountain Car.

* _init_(mode): Initializes the environment to the a mode specified by the value of mode. This can be a string of either "raw" or "tile". "raw" mode tells the environment to give you the state representation of raw features encoded in a sparse format: {0 -> position, 1 -> velocity}.

In "tile" mode you are given indices of the tiles which are active in a sparse format: {T1 -> 1, T2 -> 1, .... Tn -> 1} where Ti is the tile index for the ith tiling. All other tile indices are assumed to map to 0. In tile method, we can instead draw two grids over the state space, each offset slightly from each other. Now we can map the a point to two indices, one for each grid. 

The epsilon-greedy action selection method selects the optimal action with probability 1-E and selects uniformly at random from one of the 3 actions (0, 1, 2) with probability E. The reason that we use an epsilon-greedy action selection is we would like the agent to do explorations as well. For the purpose of testing, we will test two cases: E = 0 and 0 < E < 1.

Command - python q learning.py [args...]

Where above [args...] is a placeholder for command-line arguments: <mode> <weight out> <returns out> <episodes> <max iterations> <epsilon> <gamma> <learning rate>. These arguments are described in detail below:
1. <mode>: mode to run the environment in. Should be either ``raw'' or ``tile''.
2. <weight out>: path to output the weights of the linear model.
3. <returns out>: path to output the returns of the agent
4. <episodes>: the number of episodes your program should train the agent for. One episode is a sequence of states, actions and rewards, which ends with terminal state or ends when the maximum episode length has been reached.
5. <max iterations>: the maximum of the length of an episode. When this is reached, we terminate the current episode.
6. <epsilon>: the value for the epsilon-greedy strategy
7. <gamma>: the discount factor
8. <learning rate>: the learning rate of the Q-learning algorithm

