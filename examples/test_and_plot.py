"""
Common functionality to test and plot an agent
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import interpolate
from gymnasium.core import Wrapper


def test_agent(env, model, start_time, episode_length, warmup_period,
               log_dir: str = os.getcwd(), model_name: str = 'last_model',
               save_to_file: bool = False, plot: bool = False):
    """
    Test model agent in the given environment.

    Parameters:
        env: Environment object.
        model: Trained model for predictions.
        start_time: Start time for simulation (in seconds).
        episode_length: Maximum length of the episode (in seconds).
        warmup_period: Warmup period (in seconds).
        log_dir: Directory to log results.
        model_name: Name of the model (used for folder/file names).
        save_to_file: If True, results are saved to file.
        plot: If True, simulation results are plotted.

    Returns:
        Tuple of (observations, actions, rewards, kpis)
    """
    # Set a fixed start time in the environment
    if isinstance(env, Wrapper):
        env.unwrapped.random_start_time = False
        env.unwrapped.start_time = start_time
        env.unwrapped.max_episode_length = episode_length
        env.unwrapped.warmup_period = warmup_period
    else:
        env.random_start_time = False
        env.start_time = start_time
        env.max_episode_length = episode_length
        env.warmup_period = warmup_period

    # Reset environment
    obs, _ = env.reset()

    # Simulation loop
    done = False
    observations = [obs]
    actions = []
    rewards = []
    print("Simulating...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        done = terminated or truncated

    kpis = env.get_kpis()

    # Save KPIs to file if required
    if save_to_file:
        folder_name = os.path.join(log_dir, f"results_tests_{model_name}_{env.scenario['electricity_price']}")
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, f"kpis_{int(start_time / 3600 / 24)}.json")
        with open(file_path, "w") as f:
            json.dump(kpis, f)

    if plot:
        plot_results(env, rewards, save_to_file=save_to_file, log_dir=log_dir, model_name=model_name)

    # Restore random start time after testing
    if isinstance(env, Wrapper):
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True

    return observations, actions, rewards, kpis


def plot_results(env, rewards, points: list | None = None,
                 log_dir: str = os.getcwd(), model_name: str = 'last_model', save_to_file: bool = False):
    """
    Plot simulation results.

    Parameters:
        env: Environment object.
        rewards: List of reward values.
        points: List of point names to retrieve from simulation results. If None, defaults to all.
        log_dir: Directory to log results.
        model_name: Name of the model (used for folder/file names).
        save_to_file: If True, the plot is saved to file.
    """
    if points is None:
        points = list(env.all_measurement_vars.keys()) + list(env.all_input_vars.keys())

    # Retrieve all simulation data via API call
    url = f"{env.url}/results/{env.testid}"
    payload = {
        "point_names": points,
        "start_time": env.start_time + 1,
        "final_time": 3.1536e7  # This value could represent one year in seconds, adjust if needed
    }
    response = requests.put(url, json=payload)
    res = response.json()["payload"]

    df = pd.DataFrame(res)
    df = create_datetime_index(df)
    df.dropna(axis=0, inplace=True)
    scenario = env.scenario

    # Save results to CSV if required
    if save_to_file:
        folder_name = os.path.join(log_dir, f"results_tests_{model_name}_{scenario['electricity_price']}")
        os.makedirs(folder_name, exist_ok=True)
        csv_path = os.path.join(folder_name, f"results_sim_{int(df['time'].iloc[0] / 3600 / 24)}.csv")
        df.to_csv(csv_path)

    # Project rewards into results index
    rewards_time_days = np.arange(df["time"].iloc[0],
                                  env.start_time + env.max_episode_length,
                                  env.step_period) / 3600.0 / 24.0
    interp_func = interpolate.interp1d(rewards_time_days, rewards, kind="zero", fill_value="extrapolate")
    res_time_days = np.array(df["time"]) / 3600.0 / 24.0
    rewards_reindexed = interp_func(res_time_days)

    # Create subplots (4 vertically aligned)
    if not plt.get_fignums():
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
    else:
        fig = plt.gcf()
        axs = fig.subplots(nrows=4, ncols=1, sharex=True)

    x_time = df.index.to_pydatetime()

    # Plot operative temperature and comfort setpoint temperatures
    axs[0].plot(x_time, df['reaTZon_y'] - 273.15, color='darkorange', linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['reaTSetHea_y'] - 273.15, color='gray', linewidth=1, label='Comfort setp.')
    axs[0].plot(x_time, df['reaTSetCoo_y'] - 273.15, color='gray', linewidth=1, label='_nolegend_')
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel("Operative\ntemperature\n($^\\circ$C)")

    # Plot heat pump modulation signal
    axs[1].plot(x_time, df['oveHeaPumY_u'], color='darkorange', linestyle='-', linewidth=1, label='_nolegend_')
    axs[1].set_ylabel("Heat pump\nmodulation\nsignal\n(-)")

    # Plot rewards
    axs[2].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='Rewards')
    axs[2].set_ylabel("Rewards\n(-)")

    # Plot ambient temperature and solar irradiation on a twin axis
    axs[3].plot(x_time, df['weaSta_reaWeaTDryBul_y'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
    axs[3].set_ylabel("Ambient\ntemperature\n($^\circ$C)")
    axs[3].set_yticks(np.arange(-5, 16, 5))
    axt = axs[3].twinx()
    axt.plot(x_time, df['weaSta_reaWeaHDirNor_y'], color='gold', linestyle='-', linewidth=1, label='$\\dot{Q}_rad$')
    axt.set_ylabel("Solar\nirradiation\n(W)")

    # Adding dummy lines for legend
    axs[3].plot([], [], color='darkorange', linestyle='-', linewidth=1, label='RL')
    axs[3].plot([], [], color='dimgray', linestyle='dotted', linewidth=1, label='Price')
    axs[3].plot([], [], color='royalblue', linestyle='-', linewidth=1, label='$T_a$')
    axs[3].plot([], [], color='gold', linestyle='-', linewidth=1, label='$\\dot{Q}_{rad}$')
    axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3))

    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.tight_layout()

    # Save the figure if required, otherwise show it
    if save_to_file:
        folder_name = os.path.join(log_dir, f"results_tests_{model_name}_{scenario['electricity_price']}")
        os.makedirs(folder_name, exist_ok=True)
        pdf_path = os.path.join(folder_name, f"results_sim_{int(df['time'].iloc[0] / 3600 / 24)}.pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    else:
        plt.pause(0.001)
        plt.show()


def reindex(df: pd.DataFrame, interval: int = 60, start: int | None = None, stop: int | None = None) -> pd.DataFrame:
    """
    Reindex the DataFrame to a regular interval.

    Parameters:
        df: DataFrame containing simulation data with a 'time' column.
        interval: Desired time interval (in seconds).
        start: Start time for reindexing (in seconds); defaults to first time in df.
        stop: Stop time for reindexing (in seconds); defaults to last time in df.

    Returns:
        Reindexed DataFrame with linear interpolation.
    """
    if start is None:
        start = df["time"].iloc[0]
    if stop is None:
        stop = df["time"].iloc[-1]
    index = np.arange(start, stop + 0.1, interval).astype(int)
    df_reindexed = df.reindex(index)

    # Avoid duplicates that can cause interpolation errors
    df.drop_duplicates(subset="time", inplace=True)

    for key in df_reindexed.columns:
        interp_func = interpolate.interp1d(df["time"], df[key], kind="linear", fill_value="extrapolate")
        df_reindexed[key] = interp_func(index)

    return df_reindexed


def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a datetime index for the DataFrame based on a fixed starting timestamp.

    Parameters:
        df: DataFrame containing simulation data with a 'time' column (in seconds).

    Returns:
        DataFrame with a new datetime index.
    """
    base_timestamp = pd.Timestamp("2023-01-01")
    df["datetime"] = df["time"].apply(lambda t: base_timestamp + pd.Timedelta(seconds=t))
    df.set_index("datetime", inplace=True)
    return df


