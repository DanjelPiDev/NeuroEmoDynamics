import torch
import numpy as np


def generate_synthetic_data(profile: str,
                            timesteps: int,
                            batch_size: int,
                            input_size: int,
                            reward_size: int,
                            device: str = "cpu"):
    """
    Generate synthetic sensory input and reward signal based on a psychological profile.
    :param profile: One of 'healthy', 'depressed', or 'anxious'.
    :param timesteps: Number of time steps (e.g., simulation steps).
    :param batch_size: Number of samples per batch.
    :param input_size: Number of neurons in the sensory input (e.g., for the prefrontal layer).
    :param reward_size: Dimensionality of the reward signal (e.g., for neuromodulation).
    :param device: 'cpu' or 'cuda'.
    :return: sensory_input (timesteps, batch_size, input_size),
             reward_signal (batch_size, reward_size)
    """
    sensory_input = torch.randn(timesteps, batch_size, input_size, device=device)
    reward_signal = torch.randn(batch_size, reward_size, device=device)
    p = profile.lower()
    if p == 'healthy':
        sensory_input *= 1.0
        reward_signal *= 0.5
    elif p == 'depressed':
        sensory_input *= 0.8
        reward_signal = reward_signal * 0.2 - 0.5
    elif p == 'anxious':
        sensory_input *= 1.2
        reward_signal = reward_signal * 1.0 + 0.5
    elif p == 'impulsive':
        sensory_input *= 1.4
        reward_signal = reward_signal * 0.8 + 0.3
    elif p == 'resilient':
        sensory_input *= 1.0
        reward_signal *= 0.4
    else:
        raise ValueError("Options: healthy, depressed, anxious, impulsive, resilient.")
    t = torch.linspace(0, 2 * np.pi, timesteps, device=device).view(timesteps, 1, 1)
    sensory_input += 0.1 * torch.sin(t)
    return sensory_input, reward_signal
