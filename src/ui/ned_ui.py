import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt

from src.ui.style import set_dark_mode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from models.neuro_emotional_dynamics import NeuroEmoDynamics
from datasets import load_dataset
from data.synthetic_data import generate_synthetic_data
from utils.helper_functions import build_vocab
from safetensors.torch import load_file


def simple_tokenize(text):
    return text.lower().split()


def encode_text(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in simple_tokenize(text)]


def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    else:
        return seq[:max_len]


vocab = None

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
timesteps = 10
input_size = 512
reward_size = 1024
max_text_len = 32
num_of_classes = 6

label_to_emotion = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

profile_to_idx = {
    'depressed': 0,
    'anxious': 1,
    'healthy': 2,
    'impulsive': 3,
    'resilient': 4
}


def load_vocab():
    global vocab
    try:
        ds = load_dataset("dair-ai/emotion", split="train")
        texts = ds["text"]
        vocab = build_vocab(texts, min_freq=2, max_size=30000)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset and build vocab: {e}")


# ================= Main Window =================
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Neuro Emotional Dynamics UI")
        self.resize(800, 600)

        self.model = None
        self.figure_canvas = None

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # Load Model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model_file)
        top_layout.addWidget(self.load_model_btn)

        # Profile selection
        profile_label = QLabel("Select Profile:")
        top_layout.addWidget(profile_label)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(profile_to_idx.keys()))
        top_layout.addWidget(self.profile_combo)

        # Text input
        text_label = QLabel("Enter text:")
        top_layout.addWidget(text_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type your text here...")
        top_layout.addWidget(self.text_input)

        # Run processing button
        self.run_processing_btn = QPushButton("Run Processing")
        self.run_processing_btn.clicked.connect(self.run_processing)
        top_layout.addWidget(self.run_processing_btn)

        main_layout.addLayout(top_layout)

        # Label for displaying predicted emotions
        self.result_label = QLabel("Predicted emotions: ")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        # Widget for embedding the matplotlib plot
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_widget.setLayout(self.plot_layout)
        main_layout.addWidget(self.plot_widget)

        self.setLayout(main_layout)

        try:
            load_vocab()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_model_file(self):
        # Open a file dialog to select the model checkpoint (.safetensors file)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "checkpoints",
            "SafeTensor files (*.safetensors);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # Instantiate and load the model
            model_instance = NeuroEmoDynamics(vocab, num_classes=num_of_classes, batch_size=batch_size)
            state_dict = load_file(file_path)
            model_instance.load_state_dict(state_dict)
            model_instance.to(device)
            model_instance.eval()
            self.model = model_instance
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def run_processing(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return

        input_text = self.text_input.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Warning", "Please enter some text!")
            return

        # Get the selected profile from the combo box.
        selected_profile = self.profile_combo.currentText()
        try:
            profile_id = profile_to_idx[selected_profile]
        except KeyError:
            QMessageBox.critical(self, "Error", f"Unknown profile: {selected_profile}")
            return

        try:
            sensory_input, reward_signal = generate_synthetic_data(
                selected_profile, timesteps, batch_size, input_size, reward_size, device=device
            )
            profile_ids = torch.full((batch_size,), profile_id, dtype=torch.long, device=device)

            encoded = encode_text(input_text, vocab)
            encoded = pad_sequence(encoded, max_text_len, pad_value=vocab["<pad>"])
            text_input_tensor = torch.tensor([encoded] * batch_size, dtype=torch.long, device=device)

            with torch.no_grad():
                spikes, voltages, logits, aux_logits, serotonin, dopamine, norepinephrine = self.model(
                    sensory_input, reward_signal, text_input_tensor, profile_ids
                )

            predicted_classes = logits.argmax(dim=1).cpu().numpy()
            predicted_emotions = [label_to_emotion.get(idx, "unknown") for idx in predicted_classes]
            self.result_label.setText("Predicted emotions: " + ", ".join(predicted_emotions))

            avg_voltage = voltages[:, 0, :].mean(dim=1).cpu().numpy()
            time_steps = np.arange(timesteps)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(time_steps, avg_voltage, label="Average Voltage")
            ax.set_title("Average Membrane Voltage Over Time")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Voltage")
            ax.legend()
            fig.tight_layout()

            self.update_plot(fig)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during processing: {e}")

    def update_plot(self, fig):
        if self.figure_canvas is not None:
            self.plot_layout.removeWidget(self.figure_canvas)
            self.figure_canvas.setParent(None)

        self.figure_canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.figure_canvas)
        self.figure_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    set_dark_mode(app)

    window.show()
    sys.exit(app.exec_())
