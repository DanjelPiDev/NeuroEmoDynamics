import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader
from data.emotion_dataset import EmotionDataset
from data.synthetic_data import generate_synthetic_data
from models.neuro_emotional_dynamics import NeuroEmoDynamics
from utils.helper_functions import build_vocab
from sklearn.metrics import roc_auc_score
from safetensors.torch import save_file


def hybrid_loss(logits, targets, neurotransmitters, profile_ids, self_ref_scores,
                alpha=0.8, beta=0.1, gamma=0.1):
    ce_loss = F.cross_entropy(logits, targets)

    # Neuromodulatory consistency loss
    serotonin, dopamine, norepinephrine = neurotransmitters
    profile_factor = F.one_hot(profile_ids, num_classes=5).float()
    expected_levels = torch.tensor([
        [0.3, 0.4, 0.6],
        [0.6, 0.5, 0.8],
        [0.8, 0.7, 0.5],
        [0.5, 0.6, 0.7],
        [0.7, 0.5, 0.6]
    ], device=profile_ids.device)

    neuromod_loss = F.mse_loss(
        torch.stack([serotonin.mean(1), dopamine.mean(1), norepinephrine.mean(1)], dim=1),
        torch.matmul(profile_factor, expected_levels)
    )

    # Emotional coherence penalty
    eps = 1e-8
    prob_emotions = F.softmax(logits, dim=1) + eps
    emotion_bias = torch.tensor([
        # Depressed (allow more joy/surprise)
        [0.5, 0.2, 0.1, 0.0, 0.1, 0.1],
        # Anxious (allow calm/joy)
        [0.1, 0.3, 0.1, 0.4, 0.0, 0.1],
        # Healthy
        [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],
        # Impulsive
        [0.0, 0.2, 0.1, 0.1, 0.6, 0.0],
        # Resilient (allow surprise)
        [0.1, 0.2, 0.1, 0.1, 0.1, 0.4]
    ], device=profile_ids.device)

    kl_per_sample = F.kl_div(
        prob_emotions.log(),
        emotion_bias[profile_ids],
        reduction='batchmean'
    )
    coherence_loss = (kl_per_sample * (1 - self_ref_scores.squeeze())).mean()

    return alpha * ce_loss + beta * neuromod_loss + gamma * coherence_loss


def generate_profile_signals(profile: str, batch_size: int, device: str):
    profile_map = {
        'depressed': 0,
        'anxious': 1,
        'healthy': 2,
        'impulsive': 3,
        'resilient': 4
    }
    return torch.full((batch_size,), profile_map[profile], device=device)


def train_model(num_epochs=10, batch_size=16, timesteps=50, lr=1e-3, lambda_aux=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset (using Hugging Face's datasets)
    dataset = load_dataset("dair-ai/emotion", "split")
    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    # Map emotions to psychological profiles
    idx_to_profile = {
        0: 'depressed',
        1: 'anxious',
        2: 'healthy',
        3: 'impulsive',
        4: 'resilient'
    }

    # Prepare datasets and vocabulary
    vocab = build_vocab(texts, min_freq=2, max_size=30000)
    train_dataset = EmotionDataset(split="train", vocab=vocab, max_len=32)
    val_dataset = EmotionDataset(split="validation", vocab=vocab, max_len=32)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NeuroEmoDynamics(
        vocab,
        num_classes=6,
        num_profiles=5,
        batch_size=batch_size
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        _total_loss = 0.0

        for batch in train_dataloader:
            text_in = batch["text"].to(device)
            batch_labels = batch["label"].to(device)

            # Generate profile signals for the batch
            profile_ids = torch.randint(0, 5, (batch_size,), device=device)

            # Generate synthetic inputs per sample
            sensory_inputs = []
            reward_signals = []
            for pid in profile_ids:
                profile = idx_to_profile[int(pid)]
                si, rs = generate_synthetic_data(
                    profile=profile,
                    timesteps=timesteps,
                    batch_size=1,
                    input_size=512,
                    reward_size=1024,
                    device=device
                )
                sensory_inputs.append(si)
                reward_signals.append(rs)
            sensory_input = torch.cat(sensory_inputs, dim=1)
            reward_signal = torch.cat(reward_signals, dim=0)

            optimizer.zero_grad()
            spks, volts, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score = model(
                sensory_input, reward_signal, text_in, profile_ids
            )

            primary_loss = hybrid_loss(
                logits=logits,
                targets=batch_labels,
                neurotransmitters=(serotonin, dopamine, norepinephrine),
                profile_ids=profile_ids,
                self_ref_scores=self_ref_score
            )
            aux_loss = F.cross_entropy(aux_logits, batch_labels)
            total_loss = primary_loss + lambda_aux * aux_loss

            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _total_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = _total_loss / len(train_dataloader)

        # ================ Validation ================
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_dataloader:
                text_in = batch["text"].to(device)
                batch_labels = batch["label"].to(device)

                profile_ids = torch.randint(0, 5, (batch_size,), device=device)

                sensory_inputs = []
                reward_signals = []
                for pid in profile_ids:
                    profile = idx_to_profile[int(pid)]
                    si, rs = generate_synthetic_data(
                        profile=profile,
                        timesteps=timesteps,
                        batch_size=1,
                        input_size=512,
                        reward_size=1024,
                        device=device
                    )
                    sensory_inputs.append(si)
                    reward_signals.append(rs)
                sensory_input = torch.cat(sensory_inputs, dim=1)
                reward_signal = torch.cat(reward_signals, dim=0)

                spks, volts, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score = model(
                    sensory_input, reward_signal, text_in, profile_ids
                )

                primary_loss = hybrid_loss(
                    logits=logits,
                    targets=batch_labels,
                    neurotransmitters=(serotonin, dopamine, norepinephrine),
                    profile_ids=profile_ids,
                    self_ref_scores=self_ref_score
                )
                aux_loss = F.cross_entropy(aux_logits, batch_labels)
                loss = primary_loss + lambda_aux * aux_loss
                val_losses.append(loss.item())

                probs = F.softmax(logits, dim=1)
                all_preds.append(probs.cpu())
                all_targets.append(batch_labels.cpu())

        avg_val_loss = sum(val_losses) / len(val_losses)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        try:
            val_auc = roc_auc_score(all_targets, all_preds, multi_class="ovr")
        except Exception as e:
            val_auc = float('nan')
            print(f"Error computing AUC: {e}")

        print(
            f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val AUC {val_auc:.4f}")
        if (epoch + 1) % 10 == 0:
            print(f"Saving model at epoch [{epoch + 1}]...")
            save_file(model.state_dict(), f"../../checkpoints/neuro_emo_dynamics_v8_{epoch + 1}.safetensors")
            with open(f"../../checkpoints/neuro_emo_dynamics_v8_{epoch + 1}.json", "w") as f:
                import json
                json.dump({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_auc": val_auc
                }, f)
            print("Model saved!")

    save_file(model.state_dict(), "../../checkpoints/neuro_emo_dynamics_v8.safetensors")


if __name__ == "__main__":
    # Because of random profile selection more epochs are needed [default: 10 -> 50]
    train_model(num_epochs=50, lambda_aux=0.5, timesteps=16)
