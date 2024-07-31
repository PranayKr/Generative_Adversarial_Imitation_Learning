import numpy as np
import math
import os

""" LOAD EXPERT OSREVATIONs DATA """

states_list = np.load('ml-engineer-testing-task-data/ml-engineer-testing-task/data/obs.npy')

states_list = np.array(states_list)

print(states_list)

print("Number of Observations : " + str(len(states_list)))

""" LOAD EXPERT ACTIONS DATA """

actions = np.load('ml-engineer-testing-task-data/ml-engineer-testing-task/data/actions.npy')
print(actions)

actions_list=[]

for i in range(len(actions)):
    for j in range(len(actions[i])):
        actions_list.append(actions[i][j])

print(actions_list)

print("Number of actions : "+str(len(actions_list)))

actions_set = set(actions_list)

actions_set_list = list(actions_set)

print("The Discrete actions set for the Breakout Game : "  + str(actions_set_list))

print("Number of  Discrete  Actions :  " + str(len(actions_set_list)))

assert len(states_list) == len(actions_list)

""" LOAD EXPERT EPISODE STARTS DATA """

episode_starts = np.load('ml-engineer-testing-task-data/ml-engineer-testing-task/data/episode_starts.npy')
#print(episode_starts)

print(len(episode_starts))

trueflaglist = []

for i in range(len(episode_starts)):
    if(episode_starts[i] == True):
        trueflaglist.append(i)
        
        
print("Start of Tracjectories Indices : " + str(trueflaglist))

""" NUMBER OF TRAJECTORIES  """

print("Number of Trajectories : " + str(len(trueflaglist)))

""" LOAD EXPERT REWARDS DATA """

rewards = np.load('ml-engineer-testing-task-data/ml-engineer-testing-task/data/rewards.npy')
print(rewards)

print("Number of Rewards : " + str(len(rewards)))

positiverewardindexlist= []

postiverewardslist =[] 

rewardslist =[] 
    
for i in range(len(rewards)):
    for j in range(len(rewards[i])):
        rewardslist.append(rewards[i][j])
        if(rewards[i][0] > 0):
            positiverewardindexlist.append(i)
            postiverewardslist.append(rewards[i][j])

            
print(rewardslist)

print("Number of Rewards : " + str(len(rewardslist)))   
           
print(postiverewardslist)

print("Number of Positive Rewards : " + str(len(postiverewardslist)))   

print(positiverewardindexlist)

print(len(positiverewardindexlist))


""" CALCULATION OF EXPERT EPISODE REATURN VALUES FOR ALL THE TRAJECTORIES """

# rewardsumperepisode = 0

episode_returns = np.zeros((len(trueflaglist),))

reward_return_list = []

rewardsum =0

episode_idx=0

for i in range(len(rewardslist)):
    for j in range(len(trueflaglist)):
        if(j != len(trueflaglist) - 1):
            if(i==trueflaglist[j]):
                #print(i)
                print(trueflaglist[j])
                print(trueflaglist[j+1])
                episode_len = trueflaglist[j+1] - trueflaglist[j]
                print("episode_len :" +str(episode_len))
                print(rewardslist[trueflaglist[j]:trueflaglist[j+1]])
                perepisode_rewards_list =rewardslist[trueflaglist[j]:trueflaglist[j+1]]
                rewardsum = sum(perepisode_rewards_list)
                print("rewardsum : " +str(rewardsum))
                reward_return_list.append(rewardsum)
                episode_returns[episode_idx] = rewardsum
                episode_idx +=1
        elif(j == len(trueflaglist) - 1):
            if(i==trueflaglist[j]):
                #print("last index" +str(episodeindexlist[j]))
                perepisode_rewards_list = rewardslist[trueflaglist[j]:len(rewardslist)]
                print("last episode : "+ str(perepisode_rewards_list))
                rewardsum = sum(perepisode_rewards_list)
                print("last_rewardsum : " + str(rewardsum))
                reward_return_list.append(rewardsum)
                episode_returns[episode_idx] = rewardsum
                episode_idx +=1
                
print("reward_return_list :" + str(reward_return_list))

print("final episode_returns : " +str(episode_returns))

print("final episode_returns length : " + str(len(episode_returns)))

assert len(episode_returns)  == len(trueflaglist)


""" GENERATION OF .NPZ FILE TO DUMP EXPERT DATA PROVIDED """

numpy_dict = {
        'actions': actions_list,
        'obs': states_list,
        'rewards': rewardslist,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

print(numpy_dict)        

#save_path = "expert_breakout_v0"

save_path = "expert_trajectory"

expert_file = "expert_breakout_v0.npz"

expert_path = save_path + "/" + expert_file

if not os.path.exists(save_path):
    os.makedirs(save_path)
if(save_path is not None):
    np.savez(expert_path , **numpy_dict)


""" LOADING FROM NPZ FILE EXPERT TRAJ DATA """

#expert_path = "expert_breakout_v0\expert_breakout_v0.npz"

expert_data  = np.load(expert_path)

print(len(expert_data))

for item in expert_data:
    print(item)
    print(type(item))
    print(expert_data[item])
    print(expert_data[item].shape)
    print(type(expert_data[item]))
    print(len(expert_data[item]))
    
for i in range(len(expert_data)):
    print(expert_data.values)
















    





