import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from layers.torch_layers import LIFLayer

from data.synthetic_data import generate_synthetic_data


class PsychProfileNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.prefrontal = LIFLayer(512, V_th=1.2, tau=30.0,
                                   neuromod_transform=lambda x: torch.sigmoid(3 * x), device=self.device)
        self.amygdala = LIFLayer(256, noise_std=0.3, use_adaptive_threshold=False, device=self.device)
        self.striatum = LIFLayer(1024, base_alpha=4.0, recovery_rate=0.15, device=self.device,
                                 neuromod_transform=lambda x: torch.sigmoid(3 * x))

        self.pfc_amyg = nn.Linear(512, 256)
        self.amyg_striatum = nn.Linear(256, 1024, bias=False)

        self.to(self.device)

    def forward(self, sensory_input, reward_signal):
        """
        sensory_input: Tensor of shape (timesteps, batch, 512)
        reward_signal: Tensor of shape (batch, 1024) – one modulation vector per sample
        """
        pfc_spikes, _ = self.prefrontal(sensory_input)
        pfc_activ = self.pfc_amyg(pfc_spikes.float())

        amyg_spikes, _ = self.amygdala(pfc_activ)
        amyg_activ = self.amyg_striatum(amyg_spikes.float())

        neuromod = self.striatum.lif_group.neuromod_transform(reward_signal)
        neuromod = neuromod.unsqueeze(0)
        striatum_input = amyg_activ * neuromod

        striatum_spikes, striatum_voltages = self.striatum(striatum_input)
        return striatum_spikes, striatum_voltages

    @staticmethod
    def compute_depression_score(voltages: torch.Tensor) -> torch.Tensor:
        """
        Computes a depression score from the striatal voltage trace.
        Here we assume that lower variance (over time) in striatal activity correlates with depression.
        voltages: Tensor of shape (timesteps, batch, neurons)
        """
        var_over_time = voltages.var(dim=0)
        return var_over_time.mean()


def get_target_depression_score(profile: str) -> float:
    """
    A lower depression score (voltage variance) is assumed to simulate depression.
    :param profile: One of 'healthy', 'depressed', 'anxious', 'impulsive', 'resilient'.
    :return:
    """
    if profile == 'healthy':
        return 0.1
    elif profile == 'depressed':
        return 0.05
    elif profile == 'anxious':
        return 0.15
    elif profile == 'impulsive':
        return 0.2
    elif profile == 'resilient':
        return 0.08
    else:
        raise ValueError("Unknown profile. Choose from 'healthy', 'depressed', 'anxious', 'impulsive', or 'resilient'")


def train_model(num_epochs=100, batch_size=16, timesteps=50, target_profile="depressed", learning_rate=1e-3):
    """
    Trains the network using synthetic data to achieve a target depression score.
    A lower depression score (voltage variance) is assumed to simulate depression.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PsychProfileNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    target_depression = get_target_depression_score(target_profile)

    for epoch in range(num_epochs):
        sensory_input, reward_signal = generate_synthetic_data(
            profile=target_profile,
            timesteps=timesteps,
            batch_size=batch_size,
            input_size=512,
            reward_size=1024,
            device=device
        )
        optimizer.zero_grad()
        spikes, voltages = model(sensory_input, reward_signal)
        depression_score = PsychProfileNetwork.compute_depression_score(voltages)
        loss = (depression_score - target_depression) ** 2
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}, Depression Score = {depression_score.item():.4f}")

    torch.save(model.state_dict(), f"../checkpoints/model_{target_profile}.pt")


if __name__ == "__main__":
    num_epochs = 100
    batch_size = 16
    timesteps = 50
    target_profiles = ["healthy", "depressed", "anxious", "impulsive", "resilient"]
    learning_rate = 1e-3

    train_model(num_epochs, batch_size, timesteps, target_profiles[4], learning_rate)
