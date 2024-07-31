# Implementation of Generative Adversarial Imitation Learning (GAIL) with Proximal Policy Optimization (PPO) Algorithm for replicating the recorded behavior of an expert in the context of the Atari Breakout Game (Breakout-V0 Open AI Gym environment)
# Problem Statement Description
Breakout-V0 is an Open-AI Gym Environment in which the paddle agent needs to :
A) Hit a ball towards a brick wall and acquires points ( +1 for every hit per brick)
B) Not miss hitting the ball once it rebounds ,otherwise it loses life . The agent has a total of 5 lives . Losing all the lives leads to termination of the game and the games then resets to initial state .
# Results Showcase :
![trained_agent_playing_atari_breakout_gym_env](https://github.com/PranayKr/Generative_Adversarial_Imitation_Learning/blob/main/Breakout_Gym.gif)
# Solution Architecture Diagram :
![solution_archtecture_diagram](https://github.com/PranayKr/Generative_Adversarial_Imitation_Learning/blob/main/GAIL_BREAKOUT_ARCHITECTURE.png)
# STATE SPACE
Continuous : The observation is an RGB image of the screen, which is an array of shape (210, 160, 3)
# ACTION SPACE
Discrete : The action space is 4 dimensional. Four discrete actions of the Paddle correspond to: a) 0 — NOOP (no movement) b) 1 — FIRE c) 2 — RIGHT d) 3 — LEFT
# SOLUTION CRITERIA
The environment is considered as solved when the agent gets an average score of +16 over 100 consecutive episodes.
# Installation Instructions to setup the Project (on Windows 10 OS ):
Setting Up Python Environment :
Download and install Anaconda 3 (latest version 5.3) from this link (https://www.anaconda.com/download/) for the specific Operating System and Architecture (64-bit or 32-bit) being used for Python 3.6 + version onwards
Create (and activate) a new environment with Python 3.6 from the provided environment file atari_breakout_env.yml . Open Anaconda prompt and then execute the below given commands:
a) conda env create -f atari_breakout_env.yml
b) Add the path to System’s Environment Variables : << Relative Path where Anaconda3 is installed on system>> \envs\atari_breakout_env\Scripts \.qt-post-link.bat to solve the error as stated as below :
ERROR conda.core.link:_execute(699): An error occurred while installing package ‘defaults::qt-5.9.7-vc14h73c81de_0’. Rolling back transaction: done LinkError: post-link script failed for package defaults::qt-5.9.7-vc14h73c81de_0 location of failed script: F:\Anaconda3_Reinstall\envs\atari_breakout_env\Scripts\.qt-post-link.bat
c) Activate the Conda Environment created : conda activate atari_breakout_env
d) Download Roms.rar from the link https://github.com/openai/atari-py it contains 2 zip files inside HC ROMS.zip and ROMS.zip . Extract Roms.rar followed by extracting ROMS.zip. Then run the following command from Anaconda Prompt Shell : python -m atari_py.import_roms C:\Users\<<Username>>\Downloads\Roms\ROMS to solve the error as stated as below :
Exception: ROM is missing for breakout, see https://github.com/openai/atari-py#roms for instructions
# Details of running the Code to Train the Agent / Test the Already Trained Agent :
After creation and activation of the environment for the implementation as mentioned in the previous section :
1) Download the zipped source code file Breakout.zip and extract and save to a local system path
2) In the Anaconda Prompt Shell navigate inside the Breakout/ folder where the source code has been downloaded .
3) Run the command “jupyter notebook” from the Anaconda prompt shell window to open the jupyter notebook web-app tool in the browser from where any of the provided training and testing source codes present in notebooks(.ipynb files) can be opened.
4) Execute the Training / Testing python script files directly using python command from the Anaconda Prompt Window
## NOTE :
a) Please change the name of the (*.ckpt) file where the model weights are getting saved during training to avoid overwriting of already existing pre-trained model weights existing currently with the same filename.
b) Please set the default values of the respective command line (argparser) flags / arguments in the testing and training python script files’ argparser() function before executing them from the Anaconda Prompt window or explicitly pass their values to command line (argparser) flags / arguments while running the respective python scripts
# Generative Adversarial Imitation Learning (GAIL) with Proximal Policy Optimization (PPO) Algorithm Training Details (Files Used) :
## For Training :
a) Open the below mentioned Jupyter Notebook and execute all the cells :
GAIL_PPO_Breakout_Train.ipynb
b) Execute the python script : run_gail_breakout.py from the Anaconda Prompt Window using the command
1) python run_gail_breakout.py ( if the command line arguments default values have been set / modified in the script itself)
2) python run_gail_breakout.py — logdir <<value>> — savedir <<value>> — gamma <<value>> — numepisodes <<value>>
( if the command line arguments default values have not been set / modified in the script )
## Some Issues faced during Training :
## NOTE : Frequently after running either of the training scripts mentioned above
## (GAIL_PPO_Breakout_Train.ipynb or run_gail_breakout.py ) :
1) The ball doesn’t render / initialize in the open-ai gym environment Breakout-v0
2) The paddle seems to continuously take the same action and hence remains static / fixed in the same place in the environment … either due to the ball not rendering
   properly as discussed in first point or internal calculations of the action probabilities by the Neural Net models as random number generations play a role for initializing of weight parameters in the calculations
   Workaround : Please restart execution of the training scripts and the training should proceed normally as expected

## Neural Net Model Architecture files used for training :
1) Generator (PPO Actor-Critic) Model : policy_net_breakout.py
2) Discriminator Model : discriminator_breakout.py

# Generative Adversarial Imitation Learning (GAIL) with Proximal Policy Optimization (PPO) Algorithm Testing Details (Files Used) :
PPO Algorithm Training file used : ppo_breakout.py
## For Testing :
a) Open the Jupyter Notebook file “GAIL_PPO_Breakout_Test.ipynb” and run the code to test the results obtained using Pre-trained model weights.
b) Execute the python script : test_policy_breakout.py from the Anaconda Prompt Window using the command
1) python test_policy_breakout.py ( if the command line arguments default values have been set / modified in the script itself)
2) python test_policy_breakout.py — modeldir <<value>> — model <<value>> — logdir <<value>> — numepisodes <<value>>
( if the command line arguments default values have not been set / modified in the script )

## Neural Net Model Architecture files used for testing :
Generator (PPO Actor-Critic) Model : policy_net_breakout.py

Pretrained Model Weights provided : GAIL_model_breakout-v0_1000iter.ckpt
## Note : 
Trained model files saved in the folder path : trained_models/gail/
## The Trained model files are :
1) checkpoint
2) GAIL_model_breakout-v0_1000iter.ckpt.data-00000-of-00001
3) GAIL_model_breakout-v0_1000iter.ckpt.index
4) GAIL_model_breakout-v0_1000iter.ckpt.meta

Python Script file used for generating expert_returns numpy array and collating together the obs.npy , actions.npy , rewards.npy , episode__starts.npy files into a single expert dataset file (expert_breakout_v0.npz) : expert_dataset_generator_breakout.py
Note : Expert Dataset File saved in the folder path : expert_trajectory/

Environment file for the implementation : atari_breakout_env.yml

Output Results (GIF / Video ) files present in the folder path : Output_Results_Video_Gif_files/
1) Breakout_v0_Output.mp4
2) Breakout-V0-GailPPO_Result.gif
3) Breakout-V0-GailPPO_Result_2.gif



