import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.torch_layers import LIFLayer
from models.text_encoder import TextEncoder
from utils.emotion_override_gate import EmotionOverrideGate


class NeuroEmoDynamics(nn.Module):
    def __init__(self, vocab, embed_dim=100, text_hidden_dim=128,
                 modulation_dim=1024, num_classes=6, num_profiles=5, batch_size=16):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_profiles = num_profiles
        self.profile_embedding = nn.Embedding(num_profiles, 256)
        self.neuromod_gates = nn.ModuleDict({
            'serotonin': nn.Linear(256, 512),
            'norepinephrine': nn.Linear(256, 256),
            'dopamine': nn.Linear(256, modulation_dim)
        })

        self.prefrontal = LIFLayer(512, V_th=1.2, tau=30.0,
                                   neuromod_transform=lambda x: torch.sigmoid(3 * x),
                                   device=self.device, batch_size=batch_size, learnable_threshold=True, learnable_eta=True, learnable_tau=True)

        self.amygdala = LIFLayer(256, noise_std=0.3, use_adaptive_threshold=False,
                                 device=self.device, batch_size=batch_size, learnable_threshold=True, learnable_eta=True, learnable_tau=True)

        self.hippocampus = LIFLayer(256, V_th=1.1, tau=25.0,
                                    neuromod_transform=lambda x: torch.sigmoid(2.5 * x),
                                    device=self.device, batch_size=batch_size, learnable_threshold=True, learnable_eta=True, learnable_tau=True)

        self.thalamus = LIFLayer(256, V_th=1.0, tau=20.0,
                                 neuromod_transform=lambda x: torch.sigmoid(3 * x),
                                 device=self.device, batch_size=batch_size, learnable_threshold=True, learnable_eta=True, learnable_tau=True)

        self.pfc_amyg = nn.Linear(512, 256)
        self.pfc_hipp = nn.Linear(512, 256)
        self.pfc_thal = nn.Linear(512, 256)
        self.cross_amyg_hipp = nn.Linear(512, 256)
        self.cross_hipp_thal = nn.Linear(512, 256)
        self.cross_thal_amyg = nn.Linear(512, 256)

        self.integrator = nn.Linear(1792, 1024)
        self.striatum = LIFLayer(1024, base_alpha=4.0, recovery_rate=0.15,
                                 device=self.device, batch_size=batch_size,
                                 neuromod_transform=lambda x: torch.sigmoid(3 * x), learnable_threshold=True, learnable_eta=True, learnable_tau=True)

        self.feedback = nn.Linear(1024, 512)

        self.scale_net = nn.Sequential(
            nn.Linear(modulation_dim + 1, 1),
            nn.Sigmoid()
        )

        self.self_ref_classifier = nn.Sequential(
            nn.Linear(modulation_dim, 1),
            nn.Sigmoid()
        )

        self.override_gate = EmotionOverrideGate(modulation_dim).to(self.device)

        self.text_override_layer = nn.Linear(modulation_dim, 1)
        self.text_encoder = TextEncoder(len(vocab), embed_dim, text_hidden_dim, modulation_dim)

        self.text_gate = nn.Linear(modulation_dim, modulation_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(modulation_dim + 1024, modulation_dim),
            nn.ReLU(),
            nn.Linear(modulation_dim, modulation_dim)
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

        self.text_norm = nn.LayerNorm(modulation_dim)
        self.integ_norm = nn.LayerNorm(1024)
        self.final_classifier = nn.Linear(modulation_dim, num_classes)

        self.id_to_word = {i: token for token, i in vocab.items()}
        self.register_buffer("prev_feedback", torch.zeros(batch_size, 512))
        self.to(self.device)

    def forward(self, sensory_input, reward_signal, text_input, profile_ids=None, profile_vec=None):
        profile_vec = self.profile_embedding(profile_ids)

        serotonin = torch.sigmoid(self.neuromod_gates['serotonin'](profile_vec))
        norepinephrine = 1 + torch.sigmoid(self.neuromod_gates['norepinephrine'](profile_vec))
        dopamine = torch.sigmoid(self.neuromod_gates['dopamine'](profile_vec))

        text_encoding = self.text_encoder(text_input)
        aux_logits = self.final_classifier(text_encoding)
        aux_probs = F.softmax(aux_logits, dim=1)
        positive_scores = (aux_probs[:, 1] + aux_probs[:, 5]).unsqueeze(1)

        gate = torch.sigmoid(self.text_gate(text_encoding))
        self_ref_flag = self.get_self_reference_flag(text_input, self.id_to_word).to(self.device)
        augmented_text_encoding = torch.cat([text_encoding, self_ref_flag], dim=-1)
        sample_scale = self.scale_net(augmented_text_encoding)
        self_ref_score = self.self_ref_classifier(text_encoding)
        final_scale = sample_scale * self_ref_score

        text_mod = text_encoding * (gate + dopamine * final_scale)
        pfc_spikes, _ = self.prefrontal(sensory_input)
        pfc_out = pfc_spikes[-1].float() * serotonin + self.prev_feedback

        amyg_in = self.pfc_amyg(pfc_out) * norepinephrine
        hipp_in = self.pfc_hipp(pfc_out)
        thal_in = self.pfc_thal(pfc_out)

        cross_ah = self.cross_amyg_hipp(torch.cat([amyg_in, hipp_in], dim=-1))
        cross_ht = self.cross_hipp_thal(torch.cat([hipp_in, thal_in], dim=-1))
        cross_ta = self.cross_thal_amyg(torch.cat([thal_in, amyg_in], dim=-1))

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

        override_score = self.override_gate(
            text_encoding,
            profile_vec,
            self_ref_score,
            positive_scores
        )

        w = self.fusion_gate(torch.cat([text_mod, integ_out], dim=-1))
        w = torch.max(w, override_score)
        fused_rep = w * text_mod + (1 - w) * integ_out

        logits = self.final_classifier(fused_rep)

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
