# NeuroEmoDynamics: A Neurocognitive Profiling Spiking Neural Network

NeuroEmoDynamics is a biologically inspired spiking neural network (SNN) designed to simulate complex cognitive and emotional states. The project uses by own [Extended LIF Neurons](https://github.com/NullPointerExcy/Extended_LIF_Neurons), by using feedback and cross-connections among key brain regions, including the prefrontal cortex, amygdala, hippocampus, thalamus, and striatum, while integrating text-based emotion analysis.

Open the interactive_viz.html, to see the visualization (Only 200 neurons for each region, because of performance reasons).

## Features

- **Extended LIF Neuron Model**
- **Biologically Inspired Architecture:**
    - **Prefrontal Cortex (PFC):** Receives sensory inputs and modulates downstream regions.
    - **Downstream Regions:** Separate LIF layers for the amygdala, hippocampus, and thalamus.
    - **Feedback and Cross-Connections:**  
      Incorporates feedback from integrated outputs back to the PFC and cross-connections among downstream regions.
    - **Striatum Integration:**  
      A final integrative layer (striatum) processes the combined signal for state estimation.

- **Text-Based Emotion Analysis:**  
  A text processing branch (using an embedding layer, LSTM, and linear layers) converts text input (e.g., sentences from SST-2) into a neuromodulatory signal that influences the spiking network.

- **Synthetic Data Generation:**  
  Tools to generate synthetic sensory and reward signals that simulate different psychological profiles (e.g., healthy, depressed, anxious, impulsive, resilient).

- **Visualization Tools:**  
  Scripts to visualize the network’s connectivity and even to display neuron distributions in a 3D brain-like layout.
