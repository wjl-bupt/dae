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
import yaml
import gymnasium as gym
from gymnasium.wrappers import (
    FlattenObservation, RecordEpisodeStatistics, ClipAction, 
    NormalizeObservation, TransformObservation,
    NormalizeReward, TransformReward
)



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
    return hparam, par["nenvs"]


def get_default_hparam(args):
    print("Using default hyperparameters")
    return load_hparam(f"params/{args.algo}.yml")

# NOTE(junweiluo): add mujoco env maker
def get_mujoco_env(e, envs, args, logdir):

    # def _make_env():
    #     env = gym.make(e)
    #     env = FlattenObservation(env)
    #     env = RecordEpisodeStatistics(env)
    #     env = ClipAction(env) 
    #     env = NormalizeObservation(env)
    #     env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space = env.observation_space)
    #     env = NormalizeReward(env)
    #     env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    #     return env
    
    # register_envs()
    env = make_vec_env(
        env_id=e,
        n_envs=envs,
        seed=args.seed,
        vec_env_cls=CustomVecEnv,
        vec_env_kwargs=dict(threads=args.threads),
    )
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)  # 可选但强烈推荐
    env = VecLogger(env, logdir)
    return env, 0

def get_discrete_env(e, envs, args, logdir):
    if "MinAtar" in e:
        register_envs()
        env = make_vec_env(
            env_id=e,
            n_envs=envs,
            seed=args.seed,
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



def finish(env, algo, steps):
    print("Finishing up...")
    obs = algo._last_obs
    while env.steps < steps:
        actions, states = algo.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)


if __name__ == "__main__":

    args = get_args()
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
        if args.algo == "CustomPPO":
            algo_cls = CustomPPO
            policy = CustomActorCriticPolicy
        elif args.algo == "PPO":
            algo_cls = PPO
            policy = "MlpPolicy"
        get_env = get_mujoco_env

    hparam, nenvs = load_hparam(args.hparam_file)

    for k, v in hparam.items():
        if not callable(v):
            print(k, v)
    
    
    print(f"N_ENVS: {nenvs}    SEED: {args.seed}")
    print("List of envs: ", args.envs)
    
    
    for _env in args.envs:
        print(f"Learning Env: {_env}")
        logdir = f"./logs/{_env}/{args.run_id}" if args.logging else None


        env, frameskip = get_env(_env, nenvs, args, logdir)
        env = VecMonitor(env)
        

        if args.use_wandb:
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            cur_timestamp = int(time())
            if args.algo == "PPO":
                use_full_action = False
            else:
                use_full_action = hparam['full_action']
            run_name = f"{args.algo}_{_env}_seed{args.seed}_fullact{use_full_action}_epochs{hparam['n_epochs']}_{time_str}_{cur_timestamp}"
            group_name = f"{args.algo}_{_env}_fullact{use_full_action}_epochs{hparam['n_epochs']}"

            
            wandb.init(
                project = args.project,
                name = run_name,
                group = group_name,
                sync_tensorboard = True,
                monitor_gym=True,
                save_code=True,
            )
            wandb_callback = WandbCallback(
                verbose=2,
            )
        else:
            wandb_callback = None
            
            
        algo = algo_cls(
            policy, env, verbose=1, tensorboard_log=logdir, seed=args.seed, **hparam
        )

        algo.learn(args.steps, callback=wandb_callback)
        finish(env, algo, args.steps * frameskip)
        overall = np.mean(env.scores)
        last = np.mean(env.scores[-100:])
        print(f"Overall: {overall:.2f}    Last: {last:.2f}")
        if args.save_model:
            savedir = os.path.join(logdir if logdir else ".", "model.zip")
            print(f"Saving model to {savedir}")
            algo.save(savedir)
