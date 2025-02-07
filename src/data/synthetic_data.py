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

    if profile.lower() == 'healthy':
        # Healthy state: moderate sensory variability and reward signals near zero mean.
        sensory_input = sensory_input * 1.0
        reward_signal = reward_signal * 0.5
    elif profile.lower() == 'depressed':
        # Depressed state:
        # - Sensory input could be somewhat subdued.
        # - Reward signal might have a consistently lower (negative) mean and lower variance.
        sensory_input = sensory_input * 0.8
        reward_signal = reward_signal * 0.2 - 0.5
    elif profile.lower() == 'anxious':
        # Anxious state:
        # - Sensory input might be more erratic or “bursty” (here simulated by higher variance).
        # - Reward signal might be higher and more volatile.
        sensory_input = sensory_input * 1.2
        reward_signal = reward_signal * 1.0 + 0.5
    else:
        raise ValueError("Unknown profile. Choose from 'healthy', 'depressed', or 'anxious'.")

    t = torch.linspace(0, 2 * np.pi, timesteps, device=device).unsqueeze(1).unsqueeze(2)
    oscillation = 0.1 * torch.sin(t)
    sensory_input += oscillation

    return sensory_input, reward_signal


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timesteps = 50
    batch_size = 16
    input_size = 512   # for the prefrontal layer
    reward_size = 1024  # for neuromodulatory input to the striatum

    # Profiles: "healthy", "depressed", or "anxious"
    profile = "depressed"

    sensory_input, reward_signal = generate_synthetic_data(profile, timesteps, batch_size, input_size, reward_size, device)

    print("Sensory input shape:", sensory_input.shape)
    print("Reward signal shape:", reward_signal.shape)

    print("Sensory input mean/std:", sensory_input.mean().item(), sensory_input.std().item())
    print("Reward signal mean/std:", reward_signal.mean().item(), reward_signal.std().item())
