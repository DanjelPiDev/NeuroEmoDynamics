import numpy as np
import plotly.graph_objects as go
import torch
from datasets import load_dataset
from safetensors.torch import load_file

from models.neuro_emotional_dynamics import NeuroEmoDynamics
from utils.helper_functions import build_vocab

dataset = load_dataset("emotion")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

# Map emotions to psychological profiles
# 0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"
emotion_to_profile = {
    0: 'depressed',  # sadness
    1: 'healthy',  # joy
    2: 'healthy',  # love
    3: 'impulsive',  # anger
    4: 'anxious',  # fear
    5: 'resilient'  # surprise / healthy too?
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

# -------------------------------
model = NeuroEmoDynamics(vocab)
state_dict = load_file("../checkpoints/neuro_emo_dynamics_v4.safetensors")
model.load_state_dict(state_dict)
model.eval()
# -------------------------------

# Prefrontal -> Amygdala
pfc_amyg_weights = model.pfc_amyg.weight.detach().cpu().numpy()
# Prefrontal -> Hippocampus
pfc_hipp_weights = model.pfc_hipp.weight.detach().cpu().numpy()
# Prefrontal -> Thalamus
pfc_thal_weights = model.pfc_thal.weight.detach().cpu().numpy()

# For the integrated output into Striatum, we extract the integrator weights.
integrator_weights = model.integrator.weight.detach().cpu().numpy()
#  256 (Amygdala), 256 (Hippocampus), 256 (Thalamus),
#  256 (cross from Amygdala-Hippocampus), 256 (cross from Hippocampus-Thalamus),
#  256 (cross from Thalamus-Amygdala) and remaining for feedback.)
cross_amyg_hipp_weights = model.cross_amyg_hipp.weight.detach().cpu().numpy()
cross_hipp_thal_weights = model.cross_hipp_thal.weight.detach().cpu().numpy()
cross_thal_amyg_weights = model.cross_thal_amyg.weight.detach().cpu().numpy()

serotonin_weights = model.neuromod_gates['serotonin'].weight.detach().cpu().numpy()
norepinephrine_weights = model.neuromod_gates['norepinephrine'].weight.detach().cpu().numpy()
dopamine_weights = model.neuromod_gates['dopamine'].weight.detach().cpu().numpy()

# -------------------------------
# Define sampling sizes (number of neurons to visualize from each region)
sample_size = 200

# For the PFC layer (512 neurons) we sample indices
pfc_indices = np.sort(np.random.choice(512, sample_size, replace=False))
# For the other regions (each has 256 neurons), we sample indices
amyg_indices = np.sort(np.random.choice(256, sample_size, replace=False))
hipp_indices = np.sort(np.random.choice(256, sample_size, replace=False))
thal_indices = np.sort(np.random.choice(256, sample_size, replace=False))

# Extract submatrices corresponding to sampled neurons.
pfc_amyg_sub = pfc_amyg_weights[np.ix_(amyg_indices, pfc_indices)]
pfc_hipp_sub = pfc_hipp_weights[np.ix_(hipp_indices, pfc_indices)]
pfc_thal_sub = pfc_thal_weights[np.ix_(thal_indices, pfc_indices)]

region_centers = {
    "Prefrontal": np.array([0.0, 2.0, 0.0]),
    "Amygdala": np.array([-1.0, 0.5, -0.5]),
    "Hippocampus": np.array([0.0, -1.0, 0.5]),
    "Thalamus": np.array([0.0, 0.0, 0.5]),
    "Hypothalamus": np.array([0.0, -0.2, -0.1]),
    "Striatum": np.array([1.2, 0.3, -0.3]),
    "Raphe Nuclei": np.array([0.0, -1.8, -0.6]),
    "Locus Coeruleus": np.array([0.0, -1.5, -0.4]),
    "VTA": np.array([0.0, -1.2, -0.5]),
    "Insula": np.array([1.0, 0.0, 0.0]),
    "ACC": np.array([0.0, 1.5, 0.7]),
}


def create_brain_surfaces():
    surfaces = []

    # Cerebral hemispheres
    for hemisphere in [-1, 1]:
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50))
        x = 1.2 * np.cos(u) * np.sin(v) + hemisphere * 0.7
        y = 1.5 * np.sin(u) * np.sin(v) + 0.5
        z = 1.2 * np.cos(v)
        surfaces.append(go.Surface(
            x=x, y=y, z=z,
            opacity=0.2,
            colorscale=[[0, '#546dc7' if hemisphere < 0 else '#8f3950'],
                        [1, '#546dc7' if hemisphere < 0 else '#8f3950']],
            showscale=False,
            name=f'Cerebrum {"Left" if hemisphere < 0 else "Right"}',
        ))

    # Cerebellum
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50))
    x = 0.8 * np.cos(u) * np.sin(v)
    y = 1.0 * np.sin(u) * np.sin(v) - 1.2
    z = 1.2 * np.cos(v)
    surfaces.append(go.Surface(
        x=x, y=y, z=z,
        opacity=0.2,
        colorscale=[[0, '#e8d5c4'], [1, '#e8d5c4']],
        showscale=False,
        name='Cerebellum'
    ))

    return surfaces


def build_connection_lines(source_positions, target_positions, weight_submatrix, threshold, source_name, target_name):
    x_lines, y_lines, z_lines = [], [], []
    for i in range(weight_submatrix.shape[0]):
        for j in range(weight_submatrix.shape[1]):
            weight = weight_submatrix[i, j]
            if np.abs(weight) > threshold:
                start = source_positions[j]
                end = target_positions[i]

                mid_points = []
                if "Amygdala" in [source_name, target_name]:
                    mid_points.append(start + [0, -0.5, 0])
                elif "Hippocampus" in [source_name, target_name]:
                    mid_points.append(start + [0, 0.3, 0.2])

                t = np.linspace(0, 1, 20)
                if len(mid_points) == 1:
                    x = (1-t)**2 * start[0] + 2*(1-t)*t*mid_points[0][0] + t**2*end[0]
                    y = (1-t)**2 * start[1] + 2*(1-t)*t*mid_points[0][1] + t**2*end[1]
                    z = (1-t)**2 * start[2] + 2*(1-t)*t*mid_points[0][2] + t**2*end[2]
                else:
                    curve_dir = np.cross(end-start, np.random.randn(3))
                    curve_dir /= np.linalg.norm(curve_dir)
                    curve_mag = 0.3 * np.linalg.norm(end-start)
                    mid = 0.5*(start+end) + curve_dir*curve_mag

                    x = (1-t)**2 * start[0] + 2*(1-t)*t*mid[0] + t**2*end[0]
                    y = (1-t)**2 * start[1] + 2*(1-t)*t*mid[1] + t**2*end[1]
                    z = (1-t)**2 * start[2] + 2*(1-t)*t*mid[2] + t**2*end[2]

                x_lines.extend(x)
                y_lines.extend(y)
                z_lines.extend(z)
                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)
    return x_lines, y_lines, z_lines


def build_cross_connection_lines(source_positions, target_positions, weight_submatrix, threshold):
    x_lines, y_lines, z_lines = [], [], []
    for i in range(weight_submatrix.shape[0]):
        for j in range(weight_submatrix.shape[1]):
            weight = weight_submatrix[i, j]
            if np.abs(weight) > threshold:
                start = source_positions[j]
                end = target_positions[i]

                mid1 = start + (end - start) * 0.3 + np.random.normal(0, 0.1, 3)
                mid2 = start + (end - start) * 0.7 + np.random.normal(0, 0.1, 3)

                t = np.linspace(0, 1, 10)
                x = (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * mid1[0] + 3 * (1 - t) * t ** 2 * mid2[0] + t ** 3 * \
                    end[0]
                y = (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * mid1[1] + 3 * (1 - t) * t ** 2 * mid2[1] + t ** 3 * \
                    end[1]
                z = (1 - t) ** 3 * start[2] + 3 * (1 - t) ** 2 * t * mid1[2] + 3 * (1 - t) * t ** 2 * mid2[2] + t ** 3 * \
                    end[2]

                x_lines.extend(x)
                y_lines.extend(y)
                z_lines.extend(z)
                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)
    return x_lines, y_lines, z_lines


def build_striatum_connections(integrator_weights, region_positions, striatum_positions, threshold):
    x_lines, y_lines, z_lines = [], [], []

    for striatum_idx, striatum_pos in enumerate(striatum_positions):
        weights = integrator_weights[striatum_idx]

        # Split weights into their components. Direct inputs from each region
        amyg_weights = weights[:256]
        hipp_weights = weights[256:512]
        thal_weights = weights[512:768]
        # Cross connections (assumed to represent combined influences):
        cross_ah_weights = weights[768:1024]
        cross_ht_weights = weights[1024:1280]
        cross_ta_weights = weights[1280:1536]
        # Feedback connection (from striatum back to Prefrontal):
        feedback_weights = weights[1536:]

        t = np.linspace(0, 1, 8)

        # --- Direct connections ---
        for src_region, region_weights in zip(['Amygdala', 'Hippocampus', 'Thalamus'],
                                              [amyg_weights, hipp_weights, thal_weights]):
            max_idx = np.argmax(np.abs(region_weights))
            if max_idx < len(region_positions[src_region]) and np.abs(region_weights[max_idx]) > threshold:
                start = region_positions[src_region][max_idx]
                mid = (start + striatum_pos) / 2 + np.random.normal(0, 0.1, 3)
                x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid[0] + t ** 2 * striatum_pos[0]
                y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid[1] + t ** 2 * striatum_pos[1]
                z = (1 - t) ** 2 * start[2] + 2 * (1 - t) * t * mid[2] + t ** 2 * striatum_pos[2]
                x_lines.extend(x)
                y_lines.extend(y)
                z_lines.extend(z)

        # cross_ah: from Amygdala and Hippocampus
        max_cross_ah = np.argmax(np.abs(cross_ah_weights))
        if (max_cross_ah < len(region_positions['Amygdala']) and
                max_cross_ah < len(region_positions['Hippocampus']) and
                np.abs(cross_ah_weights[max_cross_ah]) > threshold):
            pos_a = region_positions['Amygdala'][max_cross_ah]
            pos_h = region_positions['Hippocampus'][max_cross_ah]
            cross_start = (pos_a + pos_h) / 2
            mid = (cross_start + striatum_pos) / 2 + np.random.normal(0, 0.1, 3)
            x = (1 - t) ** 2 * cross_start[0] + 2 * (1 - t) * t * mid[0] + t ** 2 * striatum_pos[0]
            y = (1 - t) ** 2 * cross_start[1] + 2 * (1 - t) * t * mid[1] + t ** 2 * striatum_pos[1]
            z = (1 - t) ** 2 * cross_start[2] + 2 * (1 - t) * t * mid[2] + t ** 2 * striatum_pos[2]
            x_lines.extend(x)
            y_lines.extend(y)
            z_lines.extend(z)

        # cross_ht: from Hippocampus and Thalamus
        max_cross_ht = np.argmax(np.abs(cross_ht_weights))
        if (max_cross_ht < len(region_positions['Hippocampus']) and
                max_cross_ht < len(region_positions['Thalamus']) and
                np.abs(cross_ht_weights[max_cross_ht]) > threshold):
            pos_h = region_positions['Hippocampus'][max_cross_ht]
            pos_t = region_positions['Thalamus'][max_cross_ht]
            cross_start = (pos_h + pos_t) / 2
            mid = (cross_start + striatum_pos) / 2 + np.random.normal(0, 0.1, 3)
            x = (1 - t) ** 2 * cross_start[0] + 2 * (1 - t) * t * mid[0] + t ** 2 * striatum_pos[0]
            y = (1 - t) ** 2 * cross_start[1] + 2 * (1 - t) * t * mid[1] + t ** 2 * striatum_pos[1]
            z = (1 - t) ** 2 * cross_start[2] + 2 * (1 - t) * t * mid[2] + t ** 2 * striatum_pos[2]
            x_lines.extend(x)
            y_lines.extend(y)
            z_lines.extend(z)

        # cross_ta: from Thalamus and Amygdala
        max_cross_ta = np.argmax(np.abs(cross_ta_weights))
        if (max_cross_ta < len(region_positions['Thalamus']) and
                max_cross_ta < len(region_positions['Amygdala']) and
                np.abs(cross_ta_weights[max_cross_ta]) > threshold):
            pos_t = region_positions['Thalamus'][max_cross_ta]
            pos_a = region_positions['Amygdala'][max_cross_ta]
            cross_start = (pos_t + pos_a) / 2
            mid = (cross_start + striatum_pos) / 2 + np.random.normal(0, 0.1, 3)
            x = (1 - t) ** 2 * cross_start[0] + 2 * (1 - t) * t * mid[0] + t ** 2 * striatum_pos[0]
            y = (1 - t) ** 2 * cross_start[1] + 2 * (1 - t) * t * mid[1] + t ** 2 * striatum_pos[1]
            z = (1 - t) ** 2 * cross_start[2] + 2 * (1 - t) * t * mid[2] + t ** 2 * striatum_pos[2]
            x_lines.extend(x)
            y_lines.extend(y)
            z_lines.extend(z)

        max_feedback = np.argmax(np.abs(feedback_weights))
        if max_feedback < len(region_positions['Prefrontal']) and np.abs(feedback_weights[max_feedback]) > threshold:
            target = region_positions['Prefrontal'][max_feedback]
            mid = (striatum_pos + target) / 2 + np.random.normal(0, 0.1, 3)
            x = (1 - t) ** 2 * striatum_pos[0] + 2 * (1 - t) * t * mid[0] + t ** 2 * target[0]
            y = (1 - t) ** 2 * striatum_pos[1] + 2 * (1 - t) * t * mid[1] + t ** 2 * target[1]
            z = (1 - t) ** 2 * striatum_pos[2] + 2 * (1 - t) * t * mid[2] + t ** 2 * target[2]
            x_lines.extend(x)
            y_lines.extend(y)
            z_lines.extend(z)

    return x_lines, y_lines, z_lines


def generate_neuron_positions(region_name, center, num_neurons, radius=0.3, layers=4):
    positions = []
    base_noise = radius * 0.2

    if region_name == "Raphe Nuclei":
        for _ in range(num_neurons):
            x = 0.0 + np.random.normal(0, 0.05)
            y = -1.5 + np.random.normal(0, 0.1)
            z = -0.5 + np.random.normal(0, 0.05)
            positions.append([x, y, z])
    elif region_name == "Locus Coeruleus":
        for _ in range(num_neurons):
            theta = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
            r = radius * 0.5 * np.random.rand()
            x = r * np.cos(theta)
            y = -1.2 + r * np.sin(theta)
            z = -0.3 + np.random.normal(0, 0.05)
            positions.append([x, y, z])
    elif region_name == "VTA":
        for _ in range(num_neurons):
            x = np.random.normal(0, 0.1)
            y = -1.0 + np.random.normal(0, 0.1)
            z = -0.4 + np.random.normal(0, 0.05)
            positions.append([x, y, z])
    elif region_name == "Insula":
        for _ in range(num_neurons):
            theta = np.random.uniform(np.pi / 2, 3 * np.pi / 2)
            r = radius * np.random.rand() ** 0.5
            x = -0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            z = 0.0 + np.random.normal(0, 0.1)
            positions.append([x, y, z])
    elif region_name == "ACC":
        for _ in range(num_neurons):
            x = 0.0 + np.random.normal(0, 0.1)
            y = 1.0 + np.random.normal(0, 0.1)
            z = 0.5 + np.random.normal(0, 0.1)
            positions.append([x, y, z])
    elif region_name == "Prefrontal":
        for i in range(num_neurons):
            layer = i % 6
            hemisphere = np.random.choice([-1, 1])
            angle = np.random.uniform(-np.pi/3, np.pi/3)

            x = hemisphere * (0.5 + layer*0.05) + np.random.normal(0, 0.05)
            y = 2.0 - layer*0.1 + np.random.normal(0, 0.05)
            z = np.sin(angle)*(0.3 + layer*0.05) + np.random.normal(0, 0.05)
            positions.append([x, y, z])
    elif region_name == "Amygdala":
        # Nuclei-based clustering
        nuclei_centers = [
            center + np.array([-0.2, 0.1, -0.1]),
            center + np.array([0.3, -0.1, 0.2]),
            center + np.array([0.0, 0.2, -0.3])
        ]
        for i in range(num_neurons):
            nucleus = nuclei_centers[np.random.choice(len(nuclei_centers))]
            pos = nucleus + np.random.normal(0, 0.15, 3)
            positions.append(pos)
    elif region_name == "Hippocampus":
        # Curved structure along anterior-posterior axis
        theta = np.linspace(-np.pi/2, np.pi/2, num_neurons)
        for t in theta:
            x = 0.5 * np.cos(t) + np.random.normal(0, 0.05)
            y = -1.0 + 0.8 * np.sin(t) + np.random.normal(0, 0.05)
            z = 0.5 * np.cos(t) + np.random.normal(0, 0.05)
            positions.append([x, y, z])
    elif region_name == "Thalamus":
        # Medial-lateral nuclear groups
        for i in range(num_neurons):
            if i % 2 == 0:  # Medial nuclei
                x = np.random.normal(0, 0.1)
            else:           # Lateral nuclei
                x = np.random.normal(0.4 * np.random.choice([-1,1]), 0.15)
            y = np.random.normal(0, 0.15)
            z = 0.5 + np.random.normal(0, 0.2)
            positions.append([x, y, z])
    elif region_name == "Hypothalamus":
        # Cluster around third ventricle
        for _ in range(num_neurons):
            pos = center + np.random.normal(0, 0.1, 3)
            positions.append(pos)
    elif region_name == "Striatum":
        # Split into caudate and putamen
        for i in range(num_neurons):
            hemisphere = np.random.choice([-1, 1])
            if i % 2 == 0:  # Putamen-like
                x = hemisphere * (1.0 + np.random.normal(0, 0.1))
                y = 0.3 + np.random.normal(0, 0.1)
                z = -0.3 + np.random.normal(0, 0.1)
            else:           # Caudate-like
                x = hemisphere * (0.8 + np.random.normal(0, 0.1))
                y = 0.5 + np.random.normal(0, 0.1)
                z = 0.1 + np.random.normal(0, 0.1)
            positions.append([x, y, z])
    else:
        for _ in range(num_neurons):
            vec = np.random.randn(3)
            vec /= np.linalg.norm(vec)
            r = radius * np.random.rand() ** (1 / 3)
            positions.append(center + r * vec)

    return np.clip(np.array(positions), center-radius, center+radius)


if __name__ == '__main__':
    pfc_positions = generate_neuron_positions("Prefrontal", region_centers["Prefrontal"], sample_size)
    amyg_positions = generate_neuron_positions("Amygdala", region_centers["Amygdala"], sample_size)
    hipp_positions = generate_neuron_positions("Hippocampus", region_centers["Hippocampus"], sample_size)
    thal_positions = generate_neuron_positions("Thalamus", region_centers["Thalamus"], sample_size)
    striatum_positions = generate_neuron_positions("Striatum", region_centers["Striatum"], sample_size)
    raphe_positions = generate_neuron_positions("Raphe Nuclei", region_centers["Raphe Nuclei"], sample_size)
    locus_positions = generate_neuron_positions("Locus Coeruleus", region_centers["Locus Coeruleus"], sample_size)
    vta_positions = generate_neuron_positions("VTA", region_centers["VTA"], sample_size)
    insula_positions = generate_neuron_positions("Insula", region_centers["Insula"], sample_size)
    acc_positions = generate_neuron_positions("ACC", region_centers["ACC"], sample_size)

    region_position_map = {
        'Amygdala': amyg_positions,
        'Hippocampus': hipp_positions,
        'Thalamus': thal_positions,
        'Striatum': striatum_positions,
        'Prefrontal': pfc_positions,
        'Raphe Nuclei': raphe_positions,
        'Locus Coeruleus': locus_positions,
        'VTA': vta_positions,
        'Insula': insula_positions,
        'ACC': acc_positions
    }

    threshold = 0.01

    x_striatum, y_striatum, z_striatum = build_striatum_connections(
        integrator_weights[:sample_size],
        region_position_map,
        striatum_positions,
        threshold=threshold
    )

    x_amyg, y_amyg, z_amyg = build_connection_lines(
        pfc_positions, amyg_positions, pfc_amyg_sub, threshold,
        source_name="Prefrontal", target_name="Amygdala"
    )
    x_hipp, y_hipp, z_hipp = build_connection_lines(
        pfc_positions, hipp_positions, pfc_hipp_sub, threshold,
        source_name="Prefrontal", target_name="Hippocampus"
    )
    x_thal, y_thal, z_thal = build_connection_lines(
        pfc_positions, thal_positions, pfc_thal_sub, threshold,
        source_name="Prefrontal", target_name="Thalamus"
    )

    raphe_indices = np.sort(np.random.choice(256, sample_size, replace=False))
    pfc_serotonin_indices = np.sort(np.random.choice(pfc_positions.shape[0], sample_size, replace=False))
    serotonin_sub = serotonin_weights[pfc_serotonin_indices][:, raphe_indices]
    x_serotonin, y_serotonin, z_serotonin = build_connection_lines(
        raphe_positions,
        pfc_positions[pfc_serotonin_indices],
        serotonin_sub.T,
        threshold=0.01,
        source_name="Raphe Nuclei",
        target_name="Prefrontal"
    )

    locus_indices = np.sort(np.random.choice(256, sample_size, replace=False))
    amyg_ne_indices = np.sort(np.random.choice(pfc_positions.shape[0], sample_size, replace=False))
    norepinephrine_sub = norepinephrine_weights[amyg_ne_indices][:, locus_indices]
    x_norepinephrine, y_norepinephrine, z_norepinephrine = build_connection_lines(
        locus_positions,
        amyg_positions[amyg_ne_indices],
        norepinephrine_sub.T,
        threshold=0.01,
        source_name="Locus Coeruleus",
        target_name="Amygdala"
    )

    vta_indices = np.sort(np.random.choice(256, sample_size, replace=False))
    striatum_da_indices = np.sort(np.random.choice(pfc_positions.shape[0], sample_size, replace=False))
    dopamine_sub = dopamine_weights[striatum_da_indices][:, vta_indices]
    x_dopamine, y_dopamine, z_dopamine = build_connection_lines(
        vta_positions,
        striatum_positions[striatum_da_indices % sample_size],
        dopamine_sub.T,
        threshold=0.01,
        source_name="VTA",
        target_name="Striatum"
    )

    conn_traces = [
        go.Scatter3d(
            x=x_amyg, y=y_amyg, z=z_amyg,
            mode='lines',
            line=dict(color='rgba(255,50,50,0.4)', width=1.5),
            name='PFC→Amygdala'
        ),
        go.Scatter3d(
            x=x_hipp, y=y_hipp, z=z_hipp,
            mode='lines',
            line=dict(color='rgba(50,150,255,0.4)', width=1.5),
            name='PFC→Hippocampus'
        ),
        go.Scatter3d(
            x=x_thal, y=y_thal, z=z_thal,
            mode='lines',
            line=dict(color='rgba(100,200,100,0.4)', width=1.5),
            name='PFC→Thalamus'
        ),
        go.Scatter3d(
            x=x_striatum, y=y_striatum, z=z_striatum,
            mode='lines',
            line=dict(color='rgba(200,100,200,0.6)', width=2),
            name='Striatal Integration\n(direct, cross, & feedback)'
        ),
        go.Scatter3d(
            x=x_serotonin, y=y_serotonin, z=z_serotonin,
            mode='lines',
            line=dict(color='rgba(255,180,0,0.3)', width=1.2),
            name='Serotonin Pathways'
        ),
        go.Scatter3d(
            x=x_norepinephrine, y=y_norepinephrine, z=z_norepinephrine,
            mode='lines',
            line=dict(color='rgba(0,150,255,0.3)', width=1.2),
            name='Norepinephrine Pathways'
        ),
        go.Scatter3d(
            x=x_dopamine, y=y_dopamine, z=z_dopamine,
            mode='lines',
            line=dict(color='rgba(0,200,50,0.3)', width=1.2),
            name='Dopamine Pathways'
        )
    ]

    def create_neuron_trace(positions, region_name, color):
        return go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=5, color=color, opacity=0.8),
            name=f'{region_name} Neurons'
        )


    trace_pfc = create_neuron_trace(pfc_positions, "Prefrontal", "blue")
    trace_amyg = create_neuron_trace(amyg_positions, "Amygdala", "red")
    trace_hipp = create_neuron_trace(hipp_positions, "Hippocampus", "purple")
    trace_thal = create_neuron_trace(thal_positions, "Thalamus", "orange")
    trace_striatum = create_neuron_trace(striatum_positions, "Striatum", "brown")
    trace_raphe = create_neuron_trace(raphe_positions, "Raphe Nuclei", "gold")
    trace_locus = create_neuron_trace(locus_positions, "Locus Coeruleus", "cyan")
    trace_vta = create_neuron_trace(vta_positions, "VTA", "limegreen")
    trace_insula = create_neuron_trace(insula_positions, "Insula", "magenta")
    trace_acc = create_neuron_trace(acc_positions, "ACC", "orange")

    fig = go.Figure(data=[
        *create_brain_surfaces(),
        trace_pfc, trace_amyg, trace_hipp, trace_thal, trace_striatum,
        trace_raphe, trace_locus, trace_vta, trace_insula, trace_acc,
        *conn_traces,
        go.Scatter3d(
            x=[c[0] for c in region_centers.values()],
            y=[c[1] for c in region_centers.values()],
            z=[c[2] for c in region_centers.values()],
            mode='markers+text',
            marker=dict(size=8, color='black'),
            text=list(region_centers.keys()),
            textposition="top center",
            name='Region Labels'
        )
    ])

    fig.update_layout(
        title="Neurocognitive Architecture Visualization",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.6))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255,255,255,0.5)')
    )
    fig.write_html("interactive_viz.html")
    fig.show()
