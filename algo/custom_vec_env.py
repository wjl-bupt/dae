import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from itertools import chain
import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env import DummyVecEnv


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import

    from algo.util import register_envs

    register_envs()

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(env.env_is_wrapped(data))
            # NOTE(junweiluo): test render_mode
            elif cmd == "get_render_mode":
                remote.send(env.render_mode)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class CustomVecEnv(VecEnv):
    """
    Asynchronous vectorized environment with which distributes the number of environments
    according to the specified threads. This is useful when the number of parallel
    environments is much larger than the number of physical processors.

    WARNING: ONLY CORE FUNCTIONALITY IS IMPLEMENTED

    Code adapted from:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param threads: number of parallel threads
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = None,
        threads: int = 1,
    ):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
            # NOTE(junweiluo): 继承虚拟环境，避免出现bug"AssertionError: OpenCV is not installed, you can do pip install opencv-python"
            # start_method = "fork"
        print(f"CustomVecEnv start_method: {start_method}")
        ctx = mp.get_context(start_method)

        self.threads = min(threads, len(env_fns))
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.threads)]
        )
        self.processes = []
        self.slicing = []
        q, r = divmod(len(env_fns), self.threads)
        l = 0
        for thread in range(self.threads):
            n = q + (1 if thread < r else 0)
            self.slicing.append(slice(l, l + n))
            l += n
        for work_remote, remote, sli in zip(
            self.work_remotes, self.remotes, self.slicing
        ):

            def wrapper():
                return DummyVecEnv(env_fns[sli])

            args = (work_remote, remote, CloudpickleWrapper(wrapper))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, sli in zip(self.remotes, self.slicing):
            remote.send(("step", actions[sli]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return (
            np.concatenate(obs),
            np.concatenate(rews),
            np.concatenate(dones),
            tuple(sum(infos, [])),
        ) 

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, (remote, sli) in enumerate(zip(self.remotes, self.slicing)):
            remote.send(("seed", seed + sli.start))
        return sum([remote.recv() for remote in self.remotes], [])

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.concatenate(obs)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return sum(imgs, [])

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """ TODO(junweiluo): implement get_attr method. """
        results = [None] * self.num_envs
        targets = self._get_target_remotes(indices = indices)
        for remote, rel_idx in targets:
            remote.send(("get_attr", attr_name))
            vals = remote.recv()
            # NOTE(junweiluo): 这里假设所有环境的属性值相同，如果不相同需要修改代码以支持返回不同的值
            # 对齐较新版本的SB3
            if not isinstance(vals, (list, tuple)):
                vals = [vals] * len(rel_idx)
            for idx, val in zip(rel_idx, vals):
                results[idx] = val
        return results
        
        # raise NotImplementedError()

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:

        targets = self._get_target_remotes(indices)
        for remote, rel_idxs in targets:
            remote.send(("set_attr", (attr_name, value)))
            remote.recv()  # 确保同步
            
        # raise NotImplementedError()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:

        results = [None] * self.num_envs
        targets = self._get_target_remotes(indices)
        for remote, rel_idxs in targets:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
            vals = remote.recv()
            for idx, val in zip(rel_idxs, vals):
                results[idx] = val
        return results
    
        # raise NotImplementedError()

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        for remote in self.remotes:
            remote.send(("is_wrapped", wrapper_class))
        return sum([remote.recv() for remote in self.remotes], [])

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:

        """
        根据 indices 返回对应的远程进程和环境在子进程中的位置
        """
        if indices is None:
            # 默认所有环境
            target_remotes = list(self.remotes)
            env_indices = list(chain.from_iterable(range(sli.start, sli.stop) for sli in self.slicing))
        else:
            # 支持 int 或 list/array
            if isinstance(indices, int):
                indices = [indices]
            env_indices = list(indices)
            target_remotes = []
            for sli, remote in zip(self.slicing, self.remotes):
                # 获取当前远程负责的环境索引
                idxs = [i for i in env_indices if i in range(sli.start, sli.stop)]
                if idxs:
                    target_remotes.append((remote, [i - sli.start for i in idxs]))
            # 返回远程和相对索引
            return target_remotes
        return list(zip(self.remotes, [list(range(sli.start, sli.stop)) for sli in self.slicing]))
        # raise NotImplementedError()
