from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from wandb.integration.sb3 import WandbCallback

from algo.util import VecLogger, VecTranspose, register_envs, get_fn, WanDBCallBack
from algo.net import NatureCNN2X, MinAtarCNN, MinAtarCNN4X, IMPALACNN
# from algo.custom_ppo.ppo import CustomPPO
# from algo.custom_ppo.policy import CustomActorCriticPolicy
from algo.custom_vec_env import CustomVecEnv

from algo.qv_ppo.ppo import QVPPO
from algo.qv_ppo.policy import QVActorCriticPolicy

from time import time
from datetime import datetime

import wandb
import argparse
import numpy as np
import os
import hashlib
import json
import yaml
import torch.nn as nn
import gymnasium as gym
import random
import torch as th
from gymnasium.wrappers import (
    FlattenObservation, RecordEpisodeStatistics, ClipAction, 
    NormalizeObservation, TransformObservation,
    NormalizeReward, TransformReward,
)
from gymnasium import Wrapper
from con_algo.util import PeriodicCheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor


class ClipRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.clip(reward, -10.0, 10.0)

class RewardNormalizationWrapper(gym.RewardWrapper):
    def reward(self, reward, reward_max = 10.0):
        return reward / 10.0


class RawRewardMonitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_reward_sum = 0

    def reset(self, **kwargs):
        self.raw_reward_sum = 0.0
        obs, info = self.env.reset(**kwargs)

        # 保证字段存在
        info["raw_r"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.raw_reward_sum += reward

        # if done and "episode" in info:
        info["raw_r"] = self.raw_reward_sum
        print(f"raw_r is {info['raw_r']}")
        return obs, reward, terminated, truncated, info
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", type=str, default="PPO", help="Which algorithm to use"
    )
    parser.add_argument(
        "--envs", type=str, default=[], nargs="+", help="Environments to train"
    )
    parser.add_argument(
        "--steps", type=int, default=10000000, help="Number of agent steps to train"
    )
    parser.add_argument(
        "--save_model",
        default=False,
        action="store_true",
        help="Save the trained model",
    )
    parser.add_argument("--logging", default=False, action="store_true", help="Logging")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--hparam_file", type=str, required=True, help="YAML file for hyperparameters"
    )
    parser.add_argument(
        "--run_id", type=int, default=0, help="run ID, used for logging"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for asynchronous environment steps",
    )

    # NOTE(junweiluo): wandb param
    parser.add_argument("--use_wandb", default=False, action="store_true", help="use wandb to log")
    parser.add_argument("--project", default=None, type=str, help="wandb project name")
    
    # NOTE(junweiluo): mujoco implement
    parser.add_argument("--continous", default=False, action="store_true", help="continous task!")
    parser.add_argument("--commit_id", default="commit_id", type=str, help="commit id in git")
    
    
    return parser.parse_args()


def load_hparam(hfile):
    print(f"Loading hyperparameters from file {hfile}")

    sched = ["learning_rate", "clip_range", "temperature"]
    special = ["nenvs", "features_extractor"]

    hparam = {}
    with open(hfile, "r") as f:
        par = yaml.safe_load(f)
        for k, v in par.items():
            if k in sched:
                d = par[k]
                print(k, d)
                hparam[k] = get_fn(d["init"], d["final"], d["ftype"])
            elif k in special:
                continue
            else:
                hparam[k] = v
    # hparam['clip_range'] = 0.1
    if "features_extractor" in par:
        if hparam.get("policy_kwargs") is None:
            hparam["policy_kwargs"] = dict()
        if par["features_extractor"] == "nature":
            hparam["policy_kwargs"]["features_extractor_class"] = NatureCNN
        elif par["features_extractor"] == "nature2x":
            hparam["policy_kwargs"]["features_extractor_class"] = NatureCNN2X
        elif par["features_extractor"] == "minatar":
            hparam["policy_kwargs"]["features_extractor_class"] = MinAtarCNN
        elif par["features_extractor"] == "minatar4x":
            hparam["policy_kwargs"]["features_extractor_class"] = MinAtarCNN4X
        elif par["features_extractor"] == "impala":
            hparam["policy_kwargs"]["features_extractor_class"] = IMPALACNN
        else:
            # NOTE(junweiluo): mujoco use flatten observation
            pass
    
    # build an identity string for a specific hparam
    hparam2str = {}
    for k, v in hparam.items():
        if callable(v):
            hparam2str[k] = v.__name__   # 或直接 skip
        else:
            hparam2str[k] = v
    hparam2str = json.dumps(hparam2str, sort_keys=True)
    hparam_id = hashlib.md5(hparam2str.encode()).hexdigest()

    return hparam, par["nenvs"], hparam_id


def get_default_hparam(args):
    print("Using default hyperparameters")
    return load_hparam(f"params/{args.algo}.yml")




def custom_build_wrapper(raw_reward_monitor, reward_minmax_normalization):
    def wrap(env):
        # if raw_reward_monitor:
        env = ClipRewardWrapper(env)
        #     # env = Monitor(env, info_keywords=("raw_r",))
        if reward_minmax_normalization:
            env = RewardNormalizationWrapper(env)
        
        return env

    return wrap

# NOTE(junweiluo): add mujoco env maker
def get_mujoco_env(e, envs, args, logdir, rew_minmax_norm = False):
    if rew_minmax_norm == True:
        # 不使用rms归一化奖励
        norm_reward = False
    else:
        norm_reward = True
        
    wrapper = custom_build_wrapper(raw_reward_monitor = True, reward_minmax_normalization = rew_minmax_norm)
    env = make_vec_env(
        env_id=e,
        n_envs=envs,
        seed=args.seed,
        vec_env_cls=CustomVecEnv,
        vec_env_kwargs=dict(threads=args.threads),
        # wrapper_class = wrapper,
    )
    
    # env = ClipAction(env)
    from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    env = VecLogger(env, logdir)
    return env, 0

def get_discrete_env(e, envs, args, logdir):
    if "MinAtar" in e:
        register_envs()
        env = make_vec_env(
            env_id=e,
            n_envs=envs,
            seed=args.seed * envs,
            vec_env_cls=CustomVecEnv,
            vec_env_kwargs=dict(threads=args.threads),
        )
        env = VecLogger(VecTranspose(env), logdir=logdir)
        frameskip = 1
    else:
        env = make_atari_env(
            # env_id=f"{_env}NoFrameskip-v4",
            env_id=f"{_env}",
            n_envs=nenvs,
            seed=args.seed,
            vec_env_cls=CustomVecEnv,
            vec_env_kwargs=dict(threads=args.threads),
        )
        env = VecLogger(VecFrameStack(env, 4), logdir=logdir)
        frameskip = 4
    return env, frameskip


# def get_env(e, envs, args, logdir):

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)

def finish(env, algo, steps):
    print("Finishing up...")
    obs = algo._last_obs
    while env.steps < steps:
        actions, states = algo.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    if args.continous == False:
        from algo.custom_ppo.ppo import CustomPPO
        from algo.custom_ppo.policy import CustomActorCriticPolicy
        if args.algo == "CustomPPO":
            algo_cls = CustomPPO
            policy = CustomActorCriticPolicy
        elif args.algo == "PPO":
            algo_cls = PPO
            policy = "CnnPolicy"
        elif args.algo == "QVPPO":
            algo_cls = QVPPO
            policy = QVActorCriticPolicy
        else:
            raise NotImplementedError
        get_env = get_discrete_env
    else:
        from con_algo.dae.ppo import CustomPPO
        from con_algo.dae.policy import CustomActorCriticPolicy
        from con_algo.vanilla_ppo.policy import SimBaFeaturesExtractor
        if args.algo == "CustomPPO":
            algo_cls = CustomPPO
            policy = CustomActorCriticPolicy
        elif args.algo == "PPO":
            from con_algo.vanilla_ppo.ppo import VanillaPPO
            algo_cls = VanillaPPO
            policy = "MlpPolicy"
            # policy = SimBaFeaturesExtractor
        elif args.algo == "QVPPO":
            algo_cls = QVPPO
            policy = QVActorCriticPolicy
        elif args.algo == "A2C":
            from con_algo.a2c.a2c import CustomA2C
            from con_algo.a2c.policy import CustomActorCriticPolicy as A2CPolicy
            algo_cls = CustomA2C
            policy = A2CPolicy
        get_env = get_mujoco_env

    hparam, nenvs, hparam_id = load_hparam(args.hparam_file)

    for k, v in hparam.items():
        if not callable(v):
            print(k, v)
    
    
    print(f"N_ENVS: {nenvs}    SEED: {args.seed}")
    print("List of envs: ", args.envs)
    
    
    for _env in args.envs:
        print(f"Learning Env: {_env}")
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        cur_timestamp = int(time())
        logdir = f"./logs/{args.algo}/{_env}/{args.run_id}_seed{args.seed}_{time_str}_{cur_timestamp}" if args.logging else None

        if args.algo == "CustomPPO":
            rew_minmax_norm = True
        elif args.algo == "PPO":
            rew_minmax_norm = False

        env, frameskip = get_env(_env, nenvs, args, logdir, rew_minmax_norm)
        # env = VecMonitor(env)
        
        callbacks_list = []
        if args.use_wandb:
            if args.algo == "PPO":
                use_full_action = False
                nheads = 1
                discouple = "null"
                gae_like_coef = "0.95"
            else:
                use_full_action = hparam['full_action']
                nheads = hparam['nheads']
                discouple = hparam['dae_discouple_correction']
                gae_like_coef = f"{hparam['gae_like_lambda']}"
            commit_id = args.commit_id
            run_name = f"{args.algo}_{_env}_seed{args.seed}_nheads{nheads}_lambda{gae_like_coef}_fullact{use_full_action}_vf{hparam['vf_coef']}_epochs{hparam['n_epochs']}_{time_str}_{cur_timestamp}"
            # group_name = f"{args.algo}_{_env}_nheads{nheads}_fullact{use_full_action}_lambda{gae_like_coef}_vf{hparam['vf_coef']}_epochs{hparam['n_epochs']}_{commit_id}_discouple{discouple}"
            group_name = f"{args.algo}_{_env}_{hparam_id}_{commit_id}"
            
            wandb_run = wandb.init(
                project = args.project,
                name = run_name,
                group = group_name,
                sync_tensorboard = True,
                monitor_gym=True,
                save_code=True,
                config=hparam
            )
            artifact = wandb.Artifact("con_algo", type = "code")
            for root, dirs, files in os.walk("con_algo/dae"):
                for f in files:
                    if f.endswith(".py"):
                        artifact.add_file(os.path.join(root, f))
            wandb_run.log_artifact(artifact)
            wandb_callback = WandbCallback(
                verbose=2,
            )
            callbacks_list.append(wandb_callback)
        else:
            wandb_callback = None
        
        if args.algo == "PPO":
            policy_kwargs = dict(
                features_extractor_class=SimBaFeaturesExtractor,
                features_extractor_kwargs=dict(
                    block_num=2,
                    hidden_dim=256,
                    activation=nn.Tanh(),
                ),
                net_arch=dict(pi=[], vf=[]), 
                share_features_extractor=dict(),
                # features_extractor = False,
            )
        else:
            policy_kwargs = hparam.get("policy_kwargs", dict())
        hparam["policy_kwargs"] = policy_kwargs
        
        # if logdir is not None:
        #     checkpoints_callback = PeriodicCheckpointCallback(save_freq=1_000_000, save_path=logdir)
        #     eval_env = get_env(_env, envs=1, args=args, logdir=logdir)[0]
        #     checkpoints_callback = None
        #     # eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=16, eval_freq=200_000)
        #     callbacks_list.append(checkpoints_callback)
        #     # callbacks_list.append(eval_callback)
        # else:
        #     checkpoints_callback = None
        
        if len(callbacks_list) > 0: 
            callbacks_list = CallbackList(callbacks_list)
        else:
            callbacks_list = None    
            
            
        algo = algo_cls(
            policy, env, verbose=1, tensorboard_log=logdir, seed=args.seed, **hparam
        )
        print(f"===== actor-critic arch ======\n{algo.policy}")
        algo.learn(args.steps, callback=callbacks_list)
        finish(env, algo, args.steps * frameskip)
        overall = np.mean(env.scores)
        last = np.mean(env.scores[-100:])
        print(f"Overall: {overall:.2f}    Last: {last:.2f}")
        if args.save_model:
            savedir = os.path.join(logdir if logdir else ".", "model.zip")
            print(f"Saving model to {savedir}")
            algo.save(savedir)
