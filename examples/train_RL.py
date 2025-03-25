#!/usr/bin/env python3
"""
Module to train and test an RL agent for the bestest_hydronic_heatpump case.
Note: This case must be deployed to run this script.

IMPORTANT:
 - To load the ExpertDataset, comment the first line in 
   stable_baselines/gail/_init_.py:
       from stable_baselines3.gail.model import GAIL
 - If issues occur related to np.ndarray during pretraining, you may have 
   duplicate numpy installations. See:
   https://stackoverflow.com/questions/54943168/problem-with-tensorflow-tf-sessionrun-wrapper-expected-all-values-in-input-dic
"""

from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
from typing import Any, Tuple, List, Optional

import os
import random
import requests

from boptestGymEnv import (
    BoptestGymEnv,
    NormalizedActionWrapper,
    NormalizedObservationWrapper,
    SaveAndTestCallback,
    DiscretizedActionWrapper
)
# from stable_baselines3.gail import ExpertDataset
from stable_baselines3 import A2C, SAC, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from examples.test_and_plot import test_agent
from testing import utilities

# Global variables
URL: str = 'https://api.boptest.net
SEED: int = 123456
random.seed(SEED)


def train_RL(
    algorithm: str = 'SAC',
    start_time_tests: List[int] = [(23 - 7) * 24 * 3600, (115 - 7) * 24 * 3600],
    episode_length_test: int = 14 * 24 * 3600,
    warmup_period: int = 1 * 24 * 3600,
    max_episode_length: int = 7 * 24 * 3600,
    mode: str = 'train',  # 'train', 'load', or 'empty'
    case: str = 'simple',
    training_timesteps: float = 3e5,
    render: bool = False,
    expert_traj: Optional[str] = None,
    model_name: str = 'last_model'
) -> Tuple[Any, Any, List[int], str]:
    """
    Train (or load) an RL agent.

    Parameters
    ----------
    start_time_tests : list of int
        Times in seconds from the beginning of the year to be used for testing.
    episode_length_test : int
        Duration (in seconds) of each testing period.
    warmup_period : int
        Warmup period duration (in seconds).
    max_episode_length : int
        Maximum length of an episode (in seconds).
    mode : str
        One of 'train', 'load', or 'empty'.
    case : str
        Specifies which case to run.
    training_timesteps : float
        Total timesteps for training.
    render : bool
        Whether to render each episode during training.
    expert_traj : Optional[str]
        Path to an expert trajectory (.npz format) for behavior cloning pretraining.
    model_name : str
        Name under which the model will be saved/loaded.

    Returns
    -------
    env, model, start_time_tests, log_dir
    """
    # Define periods to exclude from training
    excluding_periods: List[Tuple[int, int]] = [
        (start, start + episode_length_test) for start in start_time_tests
    ]
    # Exclude summer period (June 21 to September 22; no heating needed)
    excluding_periods.append((173 * 24 * 3600, 266 * 24 * 3600))

    # Create a log directory using pathlib for modern path handling.
    root_path = Path(utilities.get_root_path())
    log_dir_path = root_path / 'examples' / 'agents' / f"{algorithm}_{case}_{training_timesteps:.0e}_logdir"
    # Remove any '+' from the string (if present)
    log_dir = str(log_dir_path).replace('+', '')
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Redefine reward function via a custom environment subclass.
    class BoptestGymEnvCustomReward(BoptestGymEnv):
        """Custom environment with a tailored reward function."""

        def get_reward(self) -> float:
            # Get core KPIs from the BOPTEST server
            response = requests.get(f"{self.url}/kpi/{self.testid}")
            response.raise_for_status()
            kpis = response.json()['payload']

            # Calculate the objective integrand at this time step
            objective_integrand = kpis['cost_tot'] * 12.0 * 16.0 + 100 * kpis['tdis_tot']
            # Compute reward as the negative difference from the previous objective integrand
            reward = -(objective_integrand - self.objective_integrand)
            self.objective_integrand = objective_integrand
            return reward

    # Select environment configuration based on case
    if case == 'simple':
        env = BoptestGymEnvCustomReward(
            url=URL,
            testcase='bestest_hydronic_heat_pump',
            actions=['oveHeaPumY_u'],
            observations=OrderedDict([('reaTZon_y', (280.0, 310.0))]),
            random_start_time=True,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir
        )
    elif case == 'A':
        env = BoptestGymEnvCustomReward(
            url=URL,
            testcase='bestest_hydronic_heat_pump',
            actions=['oveHeaPumY_u'],
            observations=OrderedDict([
                ('time', (0, 604800)),
                ('reaTZon_y', (280.0, 310.0)),
                ('PriceElectricPowerHighlyDynamic', (-0.4, 0.4))
            ]),
            scenario={'electricity_price': 'highly_dynamic'},
            predictive_period=0,
            random_start_time=True,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir
        )
    elif case == 'B':
        env = BoptestGymEnvCustomReward(
            url=URL,
            testcase='bestest_hydronic_heat_pump',
            actions=['oveHeaPumY_u'],
            observations=OrderedDict([
                ('time', (0, 604800)),
                ('reaTZon_y', (280.0, 310.0)),
                ('PriceElectricPowerHighlyDynamic', (-0.4, 0.4)),
                ('LowerSetp[1]', (280.0, 310.0)),
                ('UpperSetp[1]', (280.0, 310.0))
            ]),
            predictive_period=0,
            scenario={'electricity_price': 'highly_dynamic'},
            random_start_time=True,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir
        )
    elif case == 'C':
        env = BoptestGymEnvCustomReward(
            url=URL,
            testcase='bestest_hydronic_heat_pump',
            actions=['oveHeaPumY_u'],
            observations=OrderedDict([
                ('time', (0, 604800)),
                ('reaTZon_y', (280.0, 310.0)),
                ('PriceElectricPowerHighlyDynamic', (-0.4, 0.4)),
                ('LowerSetp[1]', (280.0, 310.0)),
                ('UpperSetp[1]', (280.0, 310.0))
            ]),
            predictive_period=3 * 3600,
            scenario={'electricity_price': 'highly_dynamic'},
            random_start_time=True,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=1800,
            render_episodes=render,
            log_dir=log_dir
        )
    elif case == 'D':
        env = BoptestGymEnvCustomReward(
            url=URL,
            testcase='bestest_hydronic_heat_pump',
            actions=['oveHeaPumY_u'],
            observations=OrderedDict([
                ('time', (0, 604800)),
                ('reaTZon_y', (280.0, 310.0)),
                ('TDryBul', (265, 303)),
                ('HDirNor', (0, 862)),
                ('InternalGainsRad[1]', (0, 219)),
                ('PriceElectricPowerHighlyDynamic', (-0.4, 0.4)),
                ('LowerSetp[1]', (280.0, 310.0)),
                ('UpperSetp[1]', (280.0, 310.0))
            ]),
            predictive_period=24 * 3600,
            regressive_period=6 * 3600,
            scenario={'electricity_price': 'highly_dynamic'},
            random_start_time=True,
            excluding_periods=excluding_periods,
            max_episode_length=max_episode_length,
            warmup_period=warmup_period,
            step_period=900,
            render_episodes=render,
            log_dir=log_dir
        )
    else:
        raise ValueError(f"Unknown case '{case}'. Choose from 'simple', 'A', 'B', 'C', or 'D'.")

    # Wrap environment with normalization
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)

    # Add monitoring
    monitor_filepath = os.path.join(log_dir, 'monitor.csv')
    env = Monitor(env=env, filename=monitor_filepath)

    # Define and initialize the RL model
    model: Optional[Any] = None
    if mode == 'train':
        if algorithm.upper() == 'SAC':
            model = SAC(
                'MlpPolicy', env, verbose=1, gamma=0.99, seed=SEED,
                learning_rate=3e-4, batch_size=96, ent_coef='auto',
                buffer_size=365 * 96, learning_starts=96, train_freq=1,
                tensorboard_log=log_dir
            )
        elif algorithm.upper() == 'A2C':
            model = A2C(
                'MlpPolicy', env, verbose=1, gamma=0.99, seed=SEED,
                learning_rate=1e-6, n_steps=4, ent_coef=0,
                tensorboard_log=log_dir
            )
        elif algorithm.upper() == 'DQN':
            env = DiscretizedActionWrapper(env, n_bins_act=10)
            model = DQN(
                'MlpPolicy', env, verbose=1, gamma=0.99, seed=SEED,
                learning_rate=5e-4, batch_size=24,
                buffer_size=365 * 24, learning_starts=24, train_freq=1,
                tensorboard_log=log_dir
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if expert_traj is not None:
            # To use ExpertDataset, ensure you have modified the stable_baselines3/gail __init__ accordingly.
            from stable_baselines3.gail import ExpertDataset  # noqa: F401
            dataset = ExpertDataset(
                expert_path=expert_traj,
                randomize=False,
                traj_limitation=1,
                batch_size=96
            )
            model.pretrain(dataset, n_epochs=1000)

        # Create callback for testing and saving during training
        callback = SaveAndTestCallback(
            env, check_freq=int(1e10), save_freq=int(1e4),
            log_dir=log_dir, test=False
        )

        # Set up logger and begin training
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        model.save(os.path.join(log_dir, model_name))

    elif mode == 'load':
        if algorithm.upper() == 'SAC':
            model = SAC.load(os.path.join(log_dir, model_name))
        elif algorithm.upper() == 'A2C':
            model = A2C.load(os.path.join(log_dir, model_name))
        elif algorithm.upper() == 'DQN':
            env = DiscretizedActionWrapper(env, n_bins_act=10)
            model = DQN.load(os.path.join(log_dir, model_name))
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    elif mode == 'empty':
        model = None
    else:
        raise ValueError("mode should be either 'train', 'load', or 'empty'.")

    return env, model, start_time_tests, log_dir


def test_peak(
    env: Any,
    model: Any,
    start_time_tests: List[int],
    episode_length_test: int,
    warmup_period_test: int,
    log_dir: str = os.getcwd(),
    model_name: str = 'last_model',
    save_to_file: bool = False,
    plot: bool = False
) -> Tuple[Any, Any, Any, Any]:
    """
    Test the agent during the peak heating period.
    """
    return test_agent(
        env, model,
        start_time=start_time_tests[0],
        episode_length=episode_length_test,
        warmup_period=warmup_period_test,
        log_dir=log_dir,
        model_name=model_name,
        save_to_file=save_to_file,
        plot=plot
    )


def test_typi(
    env: Any,
    model: Any,
    start_time_tests: List[int],
    episode_length_test: int,
    warmup_period_test: int,
    log_dir: str = os.getcwd(),
    model_name: str = 'last_model',
    save_to_file: bool = False,
    plot: bool = False
) -> Tuple[Any, Any, Any, Any]:
    """
    Test the agent during a typical heating period.
    """
    return test_agent(
        env, model,
        start_time=start_time_tests[1],
        episode_length=episode_length_test,
        warmup_period=warmup_period_test,
        log_dir=log_dir,
        model_name=model_name,
        save_to_file=save_to_file,
        plot=plot
    )


def main() -> None:
    # Set rendering and plotting options
    render = True
    plot = not render  # Note: plot does not work together with render

    # Example: train an A2C agent for case 'D' with pretraining from an expert trajectory.
    expert_trajectory_path = os.path.join('trajectories', 'expert_traj_cont_28.npz')
    env, model, start_time_tests, log_dir = train_RL(
        algorithm='A2C',
        mode='train',
        case='D',
        training_timesteps=1e6,
        render=render,
        expert_traj=expert_trajectory_path
    )

    warmup_period_test = 7 * 24 * 3600
    episode_length_test = 14 * 24 * 3600
    save_to_file = True

    # Run testing for both peak and typical periods
    test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file=save_to_file, plot=plot)
    test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file=save_to_file, plot=plot)


if __name__ == "__main__":
    main()
