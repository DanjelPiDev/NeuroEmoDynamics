import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from utils.helper_functions import pad_sequence, encode_text


class SST2Dataset(Dataset):
    def __init__(self, split, vocab, max_len=32):
        dataset = load_dataset("glue", "sst2", split=split)
        self.texts = dataset["sentence"]
        self.labels = dataset["label"]
        self.vocab = vocab
        self.max_len = max_len
        self.encoded_texts = [pad_sequence(encode_text(text, vocab), max_len) for text in self.texts]

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
