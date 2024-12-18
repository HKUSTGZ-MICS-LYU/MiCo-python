from MiCoSearch import MiCoSearch

import math
import torch
import numpy as np

from torch import nn
from copy import deepcopy

from searchers.DDPG import DDPG

class HAQSearcher:
    def __init__(self, search: MiCoSearch, qbits: list, seed, org_acc) -> None:
        self.mpq = search
        self.qbits = qbits
        self.layer_macs = self.mpq.layer_macs
        self.layer_params = self.mpq.layer_params
        self.n_layers = self.mpq.n_layers
        self.org_acc = org_acc

        self.build_embedding()
        nb_states = self.layer_embedding.shape[1]
        nb_actions = 1

        self.cur_ind = 0
        self.best_reward = -math.inf
        self.max_bit = max(self.qbits)
        self.min_bit = min(self.qbits)
        self.strategy = []

        self.last_weight_action = self.max_bit
        self.last_activation_action = self.max_bit
        self.action_radio_button = True

        self.agent = DDPG(nb_states, nb_actions, seed=seed, 
                          n_layers=self.n_layers)

        return
    
    def build_embedding(self):
        layer_embedding = []
        for i in range(len(self.mpq.layers)):
            this_state = []
            m = self.mpq.layers[i]
            if isinstance(m, nn.Conv2d):
                this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
            elif isinstance(m, nn.Linear):
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
            this_state.append([i])  # index
            this_state.append([1.])  # bits, 1 is the max bit
            this_state.append([1.])  # action radio button, 1 is the weight action
            layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding
        return

    def _is_final_layer(self):
        return self.cur_ind == self.n_layers - 1
    
    def _keep_first_last_layer(self):
        self.strategy[0][0] = 8
        self.strategy[0][1] = 8
        self.strategy[-1][0] = 8
        self.strategy[-1][1] = 8
        return

    def _cur_bops(self):
        sample = self.mpq.tuple_to_sample(self.strategy)
        cur_bops = self.mpq.get_bops(sample)
        return cur_bops
    
    def _final_action_wall(self, target_bops):

        min_bops = self.mpq.get_bops([[self.min_bit] * self.n_layers, 
                           [self.min_bit] * self.n_layers])
        
        print('before action_wall: ', self.strategy, min_bops, self._cur_bops())

        while min_bops < self._cur_bops() and target_bops < self._cur_bops():
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit[1] > self.min_bit:
                    self.strategy[-(i+1)][1] -= 1
                # self._keep_first_last_layer()
                if target_bops >= self._cur_bops():
                    break
                if n_bit[0] > self.min_bit:
                    self.strategy[-(i+1)][0] -= 1
                # self._keep_first_last_layer()
                if target_bops >= self._cur_bops():
                    break
        print('after action_wall: ', self.strategy, min_bops, self._cur_bops())
        return
    
    def reward(self, acc):
        reward = (acc - self.org_acc) * 0.1
        return reward
    
    def mpq_reset(self):
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def mpq_step(self, action, ptq = False, target_bops=None):
        action = float(action)
        lbound, rbound = self.min_bit - 0.5, self.max_bit + 0.5
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        if self.action_radio_button:
            self.last_weight_action = action
        else:
            self.last_activation_action = action
            self.strategy.append([self.last_weight_action, 
                self.last_activation_action])  # save action to strategy
            
        if self._is_final_layer() and (not self.action_radio_button):
            self._final_action_wall(target_bops)
            assert len(self.strategy) == self.n_layers
            sample = self.mpq.tuple_to_sample(self.strategy)
            res = self.mpq.eval_scheme(sample, ptq=ptq)
            name = self.mpq.get_scheme_str(sample)
            acc = res['Accuracy']
            reward = self.reward(acc)
            if reward > self.best_reward:
                self.best_reward = reward
                print('New best policy: {}, reward: {:.3f}, acc: {:.3f}'.format(
                    self.strategy, self.best_reward, acc))
                self.best_res = {name:res}
            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            self.action_radio_button = not self.action_radio_button
            info_set = {'accuracy': acc}
            return obs, reward, done, info_set
        info_set = {}
        reward = 0
        done = False

        if self.action_radio_button:
            self.layer_embedding[self.cur_ind][-1] = 0.0
        else:
            self.cur_ind += 1  # the index of next layer
            self.layer_embedding[self.cur_ind][-1] = 1.0
        self.layer_embedding[self.cur_ind][-2] = float(action) / float(self.max_bit)
        self.layer_embedding[self.cur_ind][-1] = float(self.action_radio_button)
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        self.action_radio_button = not self.action_radio_button

        return obs, reward, done, info_set


    def search(self, search_budget, 
            constr_bops=None, constr_size=None,
            n_init : int = None, use_max_q = False,
            ptq = False):
        
        num_episode = search_budget
        warm_up = n_init

        best_reward = -math.inf
        best_policy = []
        self.agent.is_training = True
        step = episode = episode_steps = 0
        episode_reward = 0.
        observation = None
        T = []  # trajectory
        while episode < num_episode:  # counting based on episode
            if observation is None:
                observation = deepcopy(self.mpq_reset())
                self.agent.reset(observation)
            # agent pick action ...
            if episode <= warm_up:
                action = self.agent.random_action()
            else:
                action = self.agent.select_action(observation, episode=episode)

            # env response with next_observation, reward, terminate_info
            observation2, reward, done, info = self.mpq_step(action, ptq=ptq, 
                                                             target_bops=constr_bops)
            observation2 = deepcopy(observation2)

            T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

            # update
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            if done:  # end of episode
                print('#{}: episode_reward:{:.4f} acc: {:.4f}, cost: {:.4f}'.format(
                    episode, episode_reward, info['accuracy'], 0.0))
                final_reward = T[-1][0]
                # agent observe and update policy
                for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                    self.agent.observe(final_reward, s_t, s_t1, a_t, done)
                    if episode > warm_up:
                        self.agent.update_policy()

                self.agent.memory.append(
                    observation,
                    self.agent.select_action(observation, episode=episode),
                    0., False
                )

                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
                T = []

                if final_reward > best_reward:
                    best_reward = final_reward
                    best_policy = self.strategy

                value_loss = self.agent.get_value_loss()
                policy_loss = self.agent.get_policy_loss()
                delta = self.agent.get_delta()
        return self.best_res, None