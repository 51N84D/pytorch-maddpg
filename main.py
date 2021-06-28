from make_env import make_env
from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
from params import scale_reward
import time

# do not render the scene
e_render = False

env = make_env("traffic")

reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
n_agents = env.n
n_states = env.observation_space[0].shape[0]
n_actions = env.action_space[0].n
capacity = 1000000
batch_size = 1000
n_units = 128

n_episode = 20000
max_steps = 120
episodes_before_train = 100
save_n = 200
n_update = 100
total_steps = 0

win = None
param = None

maddpg = MADDPG(
    n_agents,
    n_states,
    n_actions,
    batch_size,
    capacity,
    episodes_before_train,
    n_units=n_units,
)
FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

t_start = time.time()
for i_episode in range(n_episode):
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        if i_episode % 100 == 0 and e_render:
            env.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = env.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs
        total_steps += 1

        if total_steps % n_update == 0:
            c_loss, a_loss = maddpg.update_policy()

    reward_record.append(total_reward)
    maddpg.episode_done += 1

    if i_episode % save_n == 0 and i_episode > 0:
        avg_score = np.mean(reward_record[-save_n:])
        print(
            "episode",
            i_episode,
            "average score {:.1f}, time: {}".format(
                avg_score, round(time.time() - t_start, 3),
            ),
        )
        t_start = time.time()

    if maddpg.episode_done == maddpg.episodes_before_train:
        print("training now begins...")
