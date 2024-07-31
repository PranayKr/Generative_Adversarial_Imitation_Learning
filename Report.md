# A brief introduction to the Problem Statement
Breakout-V0 is an Open-AI Gym Environment in which the paddle agent needs to :
A) Hit a ball towards a brick wall and acquires points ( +1 for every hit per brick)
B) Not miss hitting the ball once it rebounds ,otherwise it loses life . The agent has a total of 5 lives . Losing all the lives leads to termination of the game and the games then resets to initial state .
# Relevant Concepts
This section provides a theoretical background describing current work in this area as well as concepts and techniques used in this work.

a) Reinforcement Learning: Reinforcement Learning has become very popular with recent breakthroughs such as AlphaGo and the mastery of Atari 2600 games. Reinforcement Learning (RL) is a framework for learning a policy that maximizes an agent’s long-term reward by interacting with the environment. A policy maps situations (states) to actions. The agent receives an immediate short-term reward after each state transition. The long-term reward of a state is specified by a value-function. The value of state roughly corresponds to the total reward an agent can accumulate starting from that state. The action-value function corresponding to the long-term reward after taking action a in state s is commonly referred to as Q-value and forms the basis for the most widely used RL technique called Q-Learning

b) Temporal Difference Learning : Temporal Difference learning is a central idea to modern day RL and works by updating estimates for the action-value function based on other estimates. This ensures the agent does not have to wait until the actual cumulative reward after completing an episode to update its estimates, but is able to learn from each action.

c) Q-Learning : Q-learning is an off-policy Temporal Difference (TD) Control algorithm. Off-policy methods evaluate or improve a policy that differs from the policy used to make decisions. These decisions can thus be made by a human expert or random policy, generating (state, action, reward, new state) entries to learn an optimal policy from.
Q-learning learns a function Q that approximates the optimal action-value function. It does this by randomly initializing Q and then generating actions using a policy derived from Q, such as e-greedy. An e-greedy policy chooses the action with the highest Q value or a random action with a (low) probability of , promoting exploration as e (epsilon) increases. With this newly generated (state (St), action (At), reward (Rt+1), new state (St+1)) pair , Q is updated using rule 1.
This update rule essentially states that the current estimate must be updated using the received immediate reward plus a discounted estimation of the maximum action-value for the new state. It is important to note here that the update is done immediately after performing the action using an estimate instead of waiting for the true cumulative reward, demonstrating TD in action. The learning rate α decides how much to alter the current estimate and the discount rate γ decides how important future rewards (estimated action-value) are compared to the immediate reward.

d) Experience Replay: Experience Replay is a mechanism to store previous experience (St, At, Rt+1, St+1) in a fixed size buffer. Minibatches are then randomly sampled, added to the current time step’s experience and used to incrementally train the neural network. This method counters catastrophic forgetting, makes more efficient use of data by training on it multiple times, and exhibits better convergence behavior

e) Fixed Q Targets : In the Q-learning algorithm using a function approximator, the TD target is also dependent on the network parameter w that is being learnt/updated, and this can lead to instabilities. To address it, a separate network with identical architecture but different weights is used. And the weights of this separate target network are updated every few steps to be equal to the local network that is continuously being updated.

f) Deep Q-Learning Algorithm : In modern Q-learning, the function Q is estimated using a neural network that takes a state as input and outputs the predicted Q-values for each possible action. It is commonly denoted with Q(S, A, θ), where θ denotes the network’s weights. The actual policy used for control can subsequently be derived from Q by estimating the Q-values for each action give the current state and applying an epsilon-greedy policy. Deep Q-learning simply means using multilayer feedforward neural networks or even Convolutional Neural Networks (CNNs) to handle raw pixel input

g) Value-Based Methods : Value-Based Methods such as Q-Learning and Deep Q-Learning aim at learning optimal policy from interaction with the environment by trying to find an estimate of the optimal action-value function . While Q-Learning is implemented for environments having small state spaces by representing optimal action-value function in the form of Q-table with one row for each state and one column for each action which is used to build the optimal policy one state at a time by pulling action with maximum value from the row corresponding to each state ; it is impossible to maintain a Q-Table for environments with huge state spaces in which case the optimal action value function is represented using a non-linear function approximator such as a neural network model which forms the basis of Deep Q-Learning algorithm.


h) Policy-Based Methods (for Discrete Action Spaces) :
Unlike in the case of Value-Based Methods the optimal policy can be found out directly from interaction with the environment without the need of first finding out an estimate of optimal action-value function by using Policy-Based Methods. For this a neural network is constructed for approximating the optimal policy which takes in all the states in state space as input(number of input neurons being equal to number of states) and returns the probability of each action being selected in action space (number of output neurons being equal to number of actions in action space) The agent uses this policy to interact with the environment by passing only the most recent state as input to the neural-net model .Then the agent samples from the action probabilities to output an action in response .The algorithm needs to optimize the network weights so that the optimal action is most likely to be selected for each iteration during training, This logic helps the agent with its goal to maximize expected return.


NOTE : The above mentioned process explains the way to approximate a Stochastic Policy using Neural-Net by sampling of action
probabilities. Policy-Based Methods can be used to approximate a Deterministic Policy as well by selecting only the Greedy Action for the latest input state fed to the model during forward pass for each iteration during training


i) Policy-Based Methods (for Continuous Action Spaces) : Policy-Based Methods can be used for environments having continuous action space by using a neural network used for estimating the optimal policy having an output layer which parametrizes a continuous probability distribution by outputting the mean and variance of a normal distribution. Then in order to select an action , the agent needs to pass only the most recent state as input to the network and then use the output mean and variance to sample from the distribution.


Policy-Based Methods are better than the Value-Based Methods owing to the following factors:
a) Policy-Based Methods are simpler than Value-Based Methods because they do away with the intermediate step of estimating the optimal action-value function and directly estimate the optimal policy.
b) Policy-Based Methods tend to learn the true desired stochastic policy whereas Value-Based Methods tend to learn a deterministic or near-deterministic policy.
c) Policy-Based Methods are best suited for estimating optimal policy for environments with Continuous Action-Spaces as they
directly map a state to action unlike Value-Based Methods which need to first estimate the best action for each state which can be carried out provided that the action space is discrete with finite number of actions ; but in case of Continuous Action Space the Value-Based Methods need to find the global maximum of a non-trivial continuous action function which turns out to be an Optimization Problem in itself.


j) Policy-Gradient Methods : Policy-Gradient Methods are a subclass of Policy-Based Methods which estimate the weights of an optimal policy by first estimating the gradient of the Expected Return (cumulative reward) over all trajectories as a function of network weights of a Neural-Net Model representing policy PI to be optimized using Stochastic Gradient Ascent by looking at each state-action pair in a Trajectory separately and by taking into account the magnitude of cumulative reward i.e. expected return for that Trajectory. Either network weights are updated to increase the probability of selecting an action for a particular state in case of receiving a positive reward or the network weights are updated to decrease the probability of selecting an action for a particular state in case of receiving a negative reward by calculating the gradient i.e. derivative of the log of probability of selecting an action given a state using the Policy PI with weights Theta and multiplying it with the reward (expected return) received over all state-action pairs present in a trajectory summed up over all the sampled set of trajectories. The weights Theta of the policy are now updated with the gradient estimate calculated over several iterations with the final goal of converging to the weights of an optimal policy.


k) Actor-Critic Methods : Actor-Critic Methods are at the intersection of Value-Based Methods such as Deep-Q Network and Policy-Based Methods such as Reinforce. They use Value-Based Methods to estimate optimal action-value function which is then used as a baseline to reduce the variance of Policy-Based Methods. Actor-Critic Agents are more stable than Value-Based Agents and need fewer samples/data to learn than Policy-Based Agents. Two Neural Networks are used here one for the Actor and one for the Critic. The Critic Neural-Net takes in a state to output State-Value function of Policy PI. The Actor Neural Net takes in a state and outputs action with highest probability to be taken for that state which is used then to calculate TD-Estimate for current state by using the reward for current state and the next state. The Critic Neural-Net gets trained using this TD-Estimate value. The Critic Neural-Net then is used to calculate the Advantage Function (sum of reward for current state and difference of discounted TD-Estimate of next state and TD-Estimate of current state) which is then used as a baseline to train the Actor Neural-net.

## High-level Architectural Design of the Solution
![solution_archtecture_diagram](https://github.com/PranayKr/Generative_Adversarial_Imitation_Learning/blob/main/GAIL_BREAKOUT_ARCHITECTURE.png)

## Description of the Learning Algorithms / Model Architectures used
For building the Generator Model Actor-Critic Neural Net Architecture was used .
## Actor Model Architecture (Policy-Net) Details : 
Convolutional Neural Net Architecture was used with 3 Convolutional Layers having 32 , 64 , 64 number of output filters | 8 , 4 , 3 Kernel Size | and Stride Value of 4 , 2 , 1 respectively in the same order . The 3-layered Convolutional stack had RELU Activation function . The flattened output feature vector of the last convolutional layer was provided as input to a fully connected layer on which RELU activation function was applied . The 2nd Fully connected layer had number of output neurons equivalent to the length of the Discrete Action Space i.e. 4 . Random Categorical Function was applied on top of the output of the 2nd fully connected layer to get the probability distribution of each action corresponding to the normalized input state / observation to the Actor Model . Hence the Actor Model outputs a policy matching the Input state to each action with varying probabilities
## Critic Model Architecture (Value-Net) Details :
Critic Model was implemented using similar ConvNet Architecture as the Actor Model except for the difference being that the final fully connected layer has only 1 output neuron corresponding to the value function predictions done by the Critic Model
## Proximal Policy Optimization Algorithm :
Proximal Policy Optimization Algorithm was used to train the Actor-Critic Model described above which is an on-policy gradient method for reinforcement learning
## Discriminator Model Architecture Details :
The Discriminator Network also uses similar Convolutional Layer Stack as used in the Actor and Critic Models but the flattened feature vector of the last convolutional layer is concatenated with the one-hot encoded action vector mentioned before . This concatenated vector is now provided as input to the first fully connected layer followed by another fully connected layer having 1 output neuron .
## A brief summary of the functioning of GAIL Model used for this task :
1) The Actor Model interacts with Break-Vo Open AI Gym environment to receive a batch of randomly sampled observations as input ; the number of observations being equivalent to the batch size . The Observation array is pre-processed to reduce its dimensionality and then normalized before being fed to the Actor Model . The Actor Model outputs a policy mapping the actions to the input state with varying probabilities
2) Hence we get corresponding values of state , action , reward and next-state . The state and action pairs generated are randomly sampled based on the batch-size number and are provided as input to the Discriminator Network along with the State Action pairs generated from the Expert Trajectory Dataset .
3)  One extra pre-processing step when dealing with Observation states from Expert Demonstration dataset is to read the images present in the paths mentioned in the image path values provided as Expert Observation State values and then pre-process them .
4)  The Discriminator Network takes as input the state-action pairs randomly sampled from the Generator Model (Actor-Critic PPO Model) and the state-action pairs randomly sampled from the Expert Trajectory / Demonstration Dataset . The Action input is one-hot encoded and noise is added to it to stabilize the training . The Discriminator Network also uses similar Convolutional Layer Stack as used in the Actor and Critic Models but the flattened feature vector of the last convolutional layer is concatenated with the one-hot encoded action vector mentioned before . This concatenated vector is now provided as input to the first fully connected layer followed by another fully connected layer having 1 output neuron .
5)  Hence in this way the Discriminator Network is used twice separately with input state-action pairs from the Generator and the Expert Demonstration Dataset to calculate the log probability of s,a pairs from the Expert Dataset and inverse of log probability of the s,a pairs from the Generator Model
6) The overall loss function for the Discriminator which it is supposed to minimize is the negative of average of the sum of both the log probability value for the Expert trajectory and Inverse of Log Probability for the Generator trajectory
7) Adam Optimizer is used for minimizing the loss / cost function of the Generator with a learning rate of 1e-5
8) The log of the probabilities of generator model s,a pairs clipped within some value range is now provided as a reward signal to the Critic Network of the Generator Model
9)  The Critic Network is then used to calculate the Generalized Advantage Estimate values which are used to train the Actor Model

## Loss Functions used for the PPO (Actor-Critic) Generator Model :
## 1) Clipped Surrogate Function Loss for the Actor Model :
Ratio of the exponents of Action Probabilities by the Current Policy and Old Policy is calculated followed by clipping them in a given range and then calculating the mean of minimum of the product of the GAE values and Normal Ratios and the product of GAE values and clipped ratios
## 2) Entropy Bonus Loss :
Mean / Average of the product of Action Probabilities predicted by the current policy and log of the clipped current policy action probabilities is calculated . The Entropy Bonus Loss acts as an Entropy Regularization term which helps in ensuring sufficient exploration which allows the agent to discover faster and better policies
## 3) Value Function Loss for the Critic Model :
Mean of Squared Difference of Predicted Value Functions for the current state and Temporal Difference Estimate (TD-Estimate) of the Critic Model for the next states i.e. the sum of Current Reward and product of the Discount Factor Gamma and the next states is calculated
All the 3 loss functions mentioned as above are added together to get the overall loss function for the PPO (Actor-Critic) Generator Model after multiplying the Entropy Bonus Loss and Value Function Loss by their corresponding parameter values
## 4) Gradient Ascent Algorithm for loss calculation :
Negative of the Total loss is taken to maximize the loss / cost Function of the Discriminator
![ppo_calculation_equation](https://github.com/PranayKr/Generative_Adversarial_Imitation_Learning/blob/main/ppo_calculation.png)
## 5) Adam Optimizer
Adam Optimizer was used to train the PPO (Actor-Critic) Generator Model with a learning rate of 1e-4

Hence on the highest level the GAIL model is functioning similar to a Generative Adversarial Network (GAN) model as it has two sub-models / networks : the Generator (PPO Actor-Critic Model) and the Discriminator which are playing a zero-sum game / MiniMax Game with each other to achieve Nash Equilibrium at which stage the Generator Model should be able to generate trajectories (State-Action pairs) which should be able to match the probability distribution of Expert Trajectories (State-Action pairs) and hence fool / confuse the Discriminator Model in accurately classifying the trajectories from the Expert Dataset and Generated ones

## Hyper-Parameters Used
1) Batch_Size : 32

2) d_step : 1 ( Discriminator Step during Training )

3) a_step : 3 ( Generator (PPO (Actor-Critic)) Step during Training)

4) Target_Score : 16

5) Gamma (Discount Factor) : 0.99

6) Learning Rate ( Discriminator) : 1e-5

7) Learning Rate (Generator(PPO (Actor-Critic))) : 1e-4

8) Number of Episodes : 1000 (1e3)

9) clip_value (for surrogate function clipping) : 0.2

10) c_1 (parameter / multiplying factor for Value Function Loss ) : 1

11) c_2 (parameter / multiplying factor for Entropy Bonus Loss) : 0.01

12) Lambda (for Generative Adversarial Estimate (GAE) calculation) : 1

## RESULTS ACHIEVED
Unfortunately I was able to achieve a mean score of around +1.7 only over 100 consecutive episodes after training the model
for 1000 iterations Hence there is much scope for improvement to get better results .

## Output Results Video
![trained_agent_playing_atari_breakout_gym_env](https://github.com/PranayKr/Generative_Adversarial_Imitation_Learning/blob/main/Breakout_Gym.gif)

## Comparison plot showing the rewards acquired by the trained agent and the expert over 1000 trajectories / episodes









