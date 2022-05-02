import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from attention_model import AttentionModel
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
gamma = 0.99
episode_number = 1280*100

input_dim = 60
embedding_dim = 128
hidden_dim = 128
n_encode_layers = 3
normalization = 'batch'
tanh_clipping = 10.0
episode_length = 400
batch_size = 16
file_num = 1

writer = SummaryWriter(logdir='log_dir')

model = AttentionModel(input_dim, embedding_dim, hidden_dim, episode_length, batch_size=batch_size, n_encode_layers=n_encode_layers, \
                       normalization=normalization, tanh_clipping=tanh_clipping)

optimizer = optim.Adam(model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

# env = SCD_ENV()
# torch.manual_seed(1)

def train_episode(return_log_p, rewards_):
    batch_size = return_log_p.size()[0] #  (16, 400, 1)
    R = np.zeros((batch_size, 1))
    policy_loss = []
    rewards = []
    rewards_ =rewards_.permute(1,0,2).numpy()
    for r in rewards_[::-1, :]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    rewards = rewards.permute(1, 0, 2)

    for log_prob, reward in zip(return_log_p, rewards):
        policy_loss.append(-log_prob * reward[:, 0])  # policy_loss (size(rewards),)
    optimizer.zero_grad()                       #
    policy_loss = torch.cat(policy_loss).sum()  # List-> (Tensor -> Cat) -> Sum ## policy_loss

    policy_loss.backward()                      # SGD
    optimizer.step()
    return policy_loss


def main():

    total_reward_list = []
    true_episode = 0
    npy_id = 0
    for _ in range(file_num):  #
        npy_id += 1
        # self._1280_eps = np.load("/opt/tiger/fanzhiyun/data/ami/prepare_for_RL/epoch_1.npy")
        _1280_eps = np.load("./data/arr_"+ str(npy_id) +".npy")

        for i in tqdm(range(len(_1280_eps)//batch_size)):
            batch_data = torch.from_numpy(_1280_eps[i:i+batch_size]).to(torch.float32)

            print("time begin::", time.asctime(time.localtime(time.time())))
            return_log_p, act, rewards = model(batch_data)
            print("time end::", time.asctime(time.localtime(time.time())))

            policy_loss = train_episode(return_log_p, rewards)
            rewards = torch.mean(rewards.to(torch.float32))
            return_log_p = torch.mean(return_log_p)

            writer.add_scalar('train/policy_loss', policy_loss.item(), true_episode*batch_size)
            writer.add_scalar('train/rewards', rewards, true_episode*batch_size)
            writer.add_scalar('train/return_log_p', return_log_p.item(), true_episode*batch_size)

            print(true_episode*batch_size, "rewards:", rewards, ", policy_loss:", policy_loss, "return_log_p", return_log_p)

            true_episode += 1

        torch.save(model.state_dict(), './SaveModel/Reinforce_Save_'+str(true_episode)+'.pth')

if __name__ == '__main__':
    main()
