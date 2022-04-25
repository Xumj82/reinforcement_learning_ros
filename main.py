import os
from datetime import datetime
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gazebo_env.hockey_task_env import HockeyTaskEnv
from utils import create_logger
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(5)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(steps_done, loss)
    if steps_done % 100 == 0:
        writer.add_scalar('Loss/train', loss, steps_done-INITIAL_MEMORY)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(env, n_episodes, render=False, chk_file=None):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            # print(done)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    
            if done:
                break
        
        if episode % 10 == 0:
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
            print('save check point to ',chk_file)
            torch.save(policy_net.state_dict(), chk_file)

    env.close()
    return

def test(env, n_episodes, policy, render=False):
    # env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    TRAIN = False
    TEST = True
    CHK_FILE = 'chkpt/2022-04-25_20-29-47/dqn_hockey_model.pt'
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    # create networks
    policy_net = DQN(in_channels=3, n_actions=5).to(device)
    target_net = DQN(in_channels=3,n_actions=5).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    logger = create_logger('ros')
    writer = SummaryWriter()
    # create environment
    # env = gym.make("PongNoFrameskip-v4")
    # env = make_env(env)
    action_names = ['forward','backward','left','right','stop']
    env = HockeyTaskEnv(action_names, 5, image_shape=[112,112],logger=logger)
    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    try:
        chk_dir = os.path.join('chkpt',datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        if CHK_FILE is not None:
            chk_file = CHK_FILE
        else:
            chk_file = os.path.join(chk_dir,'dqn_hockey_model.pt')
        # print('save checkpoint to ',chk_file)
        if TRAIN:
            train(env, 400, chk_file=chk_file)
            torch.save(policy_net.state_dict(), chk_file)
        if TEST:
            checkpoint = torch.load(chk_file)
            policy_net.load_state_dict(checkpoint)
            policy_net.eval()
            test(env, 1, policy_net, render=False)
    except KeyboardInterrupt:
        env.reset()
        env.client.terminate()
        print('Close ROS connection')
    except Exception as e:
        env.reset()
        env.client.terminate()
        print(e)
        print('Close ROS connection')
