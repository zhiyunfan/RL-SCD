import numpy as np
import gym
from metric.segmentation import SegmentationPurityCoverageFMeasure

class SCD_ENV(gym.Env):
    def __init__(self):
        self.action_shape = 2
        self.batch_size = 50
        self.last_action = np.zeros((self.batch_size, 1))
        self.rewards = np.zeros((self.batch_size, 1))

        self.npy_id = 1
        self._1280_eps = np.load("/opt/tiger/fanzhiyun/data/ami/prepare_for_RL/epoch_1.npy")
        #self._1280_eps = np.load("./data/arr_1.npy")
        self.current_episode = self._1280_eps[0:self.batch_size]
        self.cur_file_done = False
        self.cur_file_count = self._1280_eps.shape[0]
        self.cur_eps_id = 0

    def reset(self):
        if self.cur_file_done: # read next file
            self._1280_eps = self.read_next_file()
        self.current_episode = self.get_cur_episode()
        self.mean_state = np.mean(self.current_episode[:, :, :-1], axis=1)
        self.time_step = 0
        self.current_state = self.current_episode[:, self.time_step]
        return self.get_current_state()

    def read_next_file(self):
        self.npy_id += 1  ### 判别当前读取哪一个 npy
        _1280_eps = np.load("/opt/tiger/fanzhiyun/data/ami/prepare_for_RL/epoch_" + str(npy_id+1) + ".npy")
        #_1280_eps = np.load("./data/arr_" + str(self.npy_id) + ".npy")
        self.cur_file_count = _1280_eps.shape[0]
        return _1280_eps

    def get_cur_episode(self):
        if self.cur_eps_id == self.cur_file_count // self.batch_size - 1:
            self.cur_file_done = True
            self.cur_eps_id = 0 
        else:
            self.cur_file_done = False
            self.cur_eps_id += 1 

        begin = self.cur_eps_id * self.batch_size
        return self._1280_eps[begin:begin+self.batch_size]


    def get_current_state(self):
        cur_state = self.current_episode[:, self.time_step]
        last_action = self.last_action
        last_action_one_hot = np.where(last_action == 1, [0, 1], [1, 0])  # bs.2
        rewards = self.rewards
        self.global_state = np.concatenate((self.mean_state, last_action_one_hot, rewards), axis=1)

        return np.concatenate((cur_state, self.global_state), axis=1)


    def step(self, actions_lst):
        action = actions_lst[-1]
        info = {}
        self.time_step += 1
        done = False if self.time_step < 400 else True
        ### TODO 计算rewards
        metric = SegmentationPurityCoverageFMeasure()

        batch_size = len(action)
        f_lst, p_lst, c_lst = [], [], []
        for i in range(batch_size):
            p, c, f = metric.score2metric(np.array(actions_lst)[:,i], self.current_episode[:self.time_step,i][:,-1])
            p_lst.append([p])
            c_lst.append([c])
            f_lst.append([f])
        rewards = np.array(f_lst)
        ## 下一状态
        self.current_state = self.current_state if done else self.get_current_state()
        info['cover'] = c
        info['pure'] = p
        info['file_done'] = self.cur_file_done
        self.last_action = action
        self.rewards = rewards
        return self.current_state, rewards, done, info



