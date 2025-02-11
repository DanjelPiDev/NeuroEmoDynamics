import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from layers.torch_layers import LIFLayer
from torch import optim
from torch.utils.data import DataLoader
from data.emotion_dataset import EmotionDataset
from data.synthetic_data import generate_synthetic_data
from models.text_encoder import TextEncoder
from utils.helper_functions import build_vocab
from sklearn.metrics import roc_auc_score, roc_curve, auc
from safetensors.torch import save_file


class NeuroEmoDynamics(nn.Module):
    def __init__(self, vocab, embed_dim=100, text_hidden_dim=128,
                 modulation_dim=1024, num_classes=6, num_profiles=5, batch_size=16):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Profile modulation system
        self.num_profiles = num_profiles
        self.profile_embedding = nn.Embedding(num_profiles, 256)
        self.neuromod_gates = nn.ModuleDict({
            'serotonin': nn.Linear(256, 512),
            'norepinephrine': nn.Linear(256, 256),
            'dopamine': nn.Linear(256, modulation_dim)
        })

        # Neural components
        self.prefrontal = LIFLayer(512, V_th=1.2, tau=30.0,
                                   neuromod_transform=lambda x: torch.sigmoid(3 * x),
                                   device=self.device, batch_size=batch_size)

        self.amygdala = LIFLayer(256, noise_std=0.3, use_adaptive_threshold=False,
                                 device=self.device, batch_size=batch_size)
        self.hippocampus = LIFLayer(256, V_th=1.1, tau=25.0,
                                    neuromod_transform=lambda x: torch.sigmoid(2.5 * x),
                                    device=self.device, batch_size=batch_size)
        self.thalamus = LIFLayer(256, V_th=1.0, tau=20.0,
                                 neuromod_transform=lambda x: torch.sigmoid(3 * x),
                                 device=self.device, batch_size=batch_size)

        # Connectivity
        self.pfc_amyg = nn.Linear(512, 256)
        self.pfc_hipp = nn.Linear(512, 256)
        self.pfc_thal = nn.Linear(512, 256)

        self.cross_amyg_hipp = nn.Linear(512, 256)
        self.cross_hipp_thal = nn.Linear(512, 256)
        self.cross_thal_amyg = nn.Linear(512, 256)

        self.integrator = nn.Linear(1792, 1024)
        self.striatum = LIFLayer(1024, base_alpha=4.0, recovery_rate=0.15,
                                 device=self.device, batch_size=batch_size,
                                 neuromod_transform=lambda x: torch.sigmoid(3 * x))

        self.feedback = nn.Linear(1024, 512)
        # STILL IN WORK...
        # Text processing (I want, that the text can "override" the sensory input)
        # E.g.: Profile is "depressed", but the text is "positive" -> the model should output "joy/surprise", if the text is strong enough (I feel happy, even though I'm depressed)
        # E.g.: Profile is "anxious", but the text is "calm" -> the model should output "joy", if the text is strong enough (I feel calm, even though I'm anxious)
        # So if the text is "self confident" and the profile is "anxious", the model should output "joy" (or "surprise" if the text is strong enough)
        self.scale_net = nn.Sequential(
            nn.Linear(modulation_dim + 1, 1),
            nn.Sigmoid()
        )

        # Is the text "self-referential"?
        self.self_ref_classifier = nn.Sequential(
            nn.Linear(modulation_dim, 1),
            nn.Sigmoid()
        )

        self.text_override_layer = nn.Linear(modulation_dim, 1)

        self.text_encoder = TextEncoder(len(vocab), embed_dim, text_hidden_dim, modulation_dim)
        # Learns a gating vector from the text encoding.
        self.text_gate = nn.Linear(modulation_dim, modulation_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(modulation_dim + 1024, modulation_dim),
            nn.ReLU(),
            nn.Linear(modulation_dim, modulation_dim)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2048, 1),  # 1024 + 1024 if modulation_dim == 1024
            nn.Sigmoid()
        )
        self.text_norm = nn.LayerNorm(modulation_dim)
        self.integ_norm = nn.LayerNorm(1024)

        self.final_classifier = nn.Linear(modulation_dim, num_classes)

        self.id_to_word = {i: token for token, i in vocab.items()}

        self.register_buffer("prev_feedback", torch.zeros(batch_size, 512))
        self.to(self.device)

    def forward(self, sensory_input, reward_signal, text_input, profile_ids=None, profile_vec=None):
        # Profile-based modulation
        profile_vec = self.profile_embedding(profile_ids)

        serotonin = torch.sigmoid(self.neuromod_gates['serotonin'](profile_vec))  # PFC modulation
        norepinephrine = 1 + torch.sigmoid(self.neuromod_gates['norepinephrine'](profile_vec))  # Amygdala gain
        dopamine = torch.sigmoid(self.neuromod_gates['dopamine'](profile_vec))  # Reward

        # Emotion-text processing with dopamine modulation
        text_encoding = self.text_encoder(text_input)
        aux_logits = self.final_classifier(text_encoding)
        aux_probs = F.softmax(aux_logits, dim=1)
        positive_scores = (aux_probs[:, 1] + aux_probs[:, 5]).unsqueeze(1)

        # Gate for text encoding, modulated by dopamine
        gate = torch.sigmoid(self.text_gate(text_encoding))

        self_ref_flag = self.get_self_reference_flag(text_input, self.id_to_word).to(self.device)
        augmented_text_encoding = torch.cat([text_encoding, self_ref_flag], dim=-1)
        sample_scale = self.scale_net(augmented_text_encoding)
        self_ref_score = self.self_ref_classifier(text_encoding)  # high if the text uses "I", etc.
        final_scale = sample_scale * self_ref_score

        text_mod = text_encoding * (gate + dopamine * final_scale)

        # Process inputs with profile modulation
        pfc_spikes, _ = self.prefrontal(sensory_input)
        pfc_out = pfc_spikes[-1].float() * serotonin + self.prev_feedback

        # Profile-modulated pathway processing
        amyg_in = self.pfc_amyg(pfc_out) * norepinephrine
        hipp_in = self.pfc_hipp(pfc_out)
        thal_in = self.pfc_thal(pfc_out)

        # Cross-connections with emotional bias
        cross_ah = self.cross_amyg_hipp(torch.cat([amyg_in, hipp_in], dim=-1))
        cross_ht = self.cross_hipp_thal(torch.cat([hipp_in, thal_in], dim=-1))
        cross_ta = self.cross_thal_amyg(torch.cat([thal_in, amyg_in], dim=-1))

        # Integrate with feedback
        integ_input = torch.cat([
            amyg_in, hipp_in, thal_in,
            cross_ah, cross_ht, cross_ta,
            self.prev_feedback[:, :256]
        ], dim=-1)
        integ_out = self.integrator(integ_input)

        new_feedback = self.feedback(integ_out).detach()
        new_feedback = torch.clamp(new_feedback, min=-10.0, max=10.0)
        self.prev_feedback = new_feedback

        text_mod = self.text_norm(text_mod)
        integ_out = self.integ_norm(integ_out)

        override_score = torch.sigmoid(self.text_override_layer(text_mod))
        override_boost = (positive_scores * self_ref_score) * 1.5
        override_score = torch.clamp(override_score + override_boost, 0, 1.5)

        w = self.fusion_gate(torch.cat([text_mod, integ_out], dim=-1))
        w = torch.max(w, override_score)
        fused_rep = w * text_mod + (1 - w) * integ_out

        logits = self.final_classifier(fused_rep)

        # Striatal processing with modulated reward
        striatum_input = integ_out.unsqueeze(0).repeat(10, 1, 1) * \
                         reward_signal.unsqueeze(0) * dopamine.unsqueeze(0)
        spks, volts = self.striatum(striatum_input)

        return spks, volts, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score

    @staticmethod
    def decode_tokens(token_tensor, id_to_word):
        tokens = [id_to_word[int(token)] for token in token_tensor]
        return " ".join(tokens)

    def get_self_reference_flag(self, token_tensor, id_to_word):
        first_person_pronouns = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
        flags = []
        for sample in token_tensor:
            text = self.decode_tokens(sample, id_to_word)
            tokens = text.lower().split()
            flag = 1.0 if any(token in first_person_pronouns for token in tokens) else 0.0
            flags.append(flag)
        return torch.tensor(flags, dtype=torch.float32).unsqueeze(1)

    @staticmethod
    def compute_depression_score(voltages):
        return voltages.var(dim=0, unbiased=False).mean()


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
    emotion_to_profile = {
        0: 'depressed',  # sadness
        1: 'healthy',  # joy
        2: 'healthy',  # love
        3: 'impulsive',  # anger
        4: 'anxious',  # fear
        5: 'healthy'  # surprise - now mapped to healthy instead of resilient
    }

    profile_to_idx = {
        'depressed': 0,
        'anxious': 1,
        'healthy': 2,
        'impulsive': 3,
        'resilient': 4
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

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        _total_loss = 0.0

        for batch in train_dataloader:
            text_in = batch["text"].to(device)
            batch_labels = batch["label"].to(device)

            # Generate profile signals for the batch
            profile_ids = torch.tensor(
                [profile_to_idx[emotion_to_profile[l.item()]] for l in batch_labels],
                device=device
            )

            # Generate synthetic inputs per sample
            sensory_inputs = []
            reward_signals = []
            for l in batch_labels:
                profile = emotion_to_profile[l.item()]
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

                profile_ids = torch.tensor(
                    [profile_to_idx[emotion_to_profile[l.item()]] for l in batch_labels],
                    device=device
                )

                sensory_inputs = []
                reward_signals = []
                for l in batch_labels:
                    profile = emotion_to_profile[l.item()]
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

                # Get probabilities and store predictions
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

    save_file(model.state_dict(), "../checkpoints/neuro_emo_dynamics_v7.safetensors")


if __name__ == "__main__":
    train_model(num_epochs=10, lambda_aux=0.8)
