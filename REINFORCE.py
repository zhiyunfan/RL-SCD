import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from gym_wrapper_scd import SCD_ENV
from tqdm import tqdm

gamma = 0.99
episode_number = 1280*100


env = SCD_ENV()
torch.manual_seed(1)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # nn.Linear = y=wx+b, FC(Fully Connected Layer)
        # input(state) : 1x4
        self.affine1 = nn.Linear(122, 64) # (1x4) x (4x128) = (1x128) // parameter(w) = 4 * 128
        self.affine2 = nn.Linear(64, 2) # (1x128) x (128x2) = (1x2) // parameter(w) = 128 * 2
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x) # output = 1x2 , nums of action : 2
        return F.softmax(action_scores, dim=-1)

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


# s -> Model -> a
# Categorical(Discrete) Distribution
#   >>
def select_action(state):                    # state = [ , , ,] =
    state = torch.Tensor(state)            # # state = [[ , , , ]]  ## torch.Tensor(state).unsqueeze(0)
    probs = model(state)                     # Action(0,1)  (State->Model->Action)
    m = Categorical(probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))       # (log_pi(a) * Gt)
    return action.unsqueeze(-1).numpy() # bs, 1


def finish_episode():
    model_rewards = np.array(model.rewards)  # <class 'tuple'>: (400, 16, 1)
    batch_size = model_rewards.shape[1]
    R = np.zeros((batch_size, 1))
    policy_loss = []
    rewards = []
    for r in model_rewards[::-1, :]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards) #float32  400.16.1
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for log_prob, reward in zip(model.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward[:, 0])  # policy_loss (size(rewards),)
    optimizer.zero_grad()                       #
    policy_loss = torch.cat(policy_loss).sum()  # List-> (Tensor -> Cat) -> Sum ## policy_loss

    policy_loss.backward()                      # SGD
    optimizer.step()                            # Wight Update
    del model.rewards[:]
    del model.saved_log_probs[:]


def main():
    file_num = 50
    total_reward_list = []
    true_episode = 0
    for i in tqdm(range(file_num)):  #
        while True:
            true_episode += 1
            state = env.reset()     #
            total_reward = 0
            actions_lst = []
            for t in tqdm(range(400)):
                # env.render()
                action = select_action(state)   # action  0 1
                actions_lst.append(action)
                state, reward, done, info = env.step(actions_lst) #
                model.rewards.append(reward)              #
                total_reward += reward
                if done:
                    total_reward_list.append(total_reward)
                    break
            finish_episode()
            if info['file_done']:
                break

        # if true_episode % 10 == 0: # moving, average
        print('Day:', true_episode, "rewards:", total_reward_list[-5:], ", mean:", np.mean(total_reward_list[-5:], dtype=int))

        # if true_episode % 10 == 0:  # moving, average
        torch.save(model.state_dict(), './SaveModel/Reinforce_Save_'+str(i)+'.pth')



if __name__ == '__main__':
    main()
