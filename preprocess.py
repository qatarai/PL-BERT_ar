#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import yaml
from tqdm import tqdm
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from camel_tools.tokenizers.word import simple_word_tokenize
from datasets import load_dataset

root_directory = "/mnt/datasets/wiki_phoneme"
os.makedirs(root_directory, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("google/wiki40b", data_dir="ar", split="train", cache_dir="/mnt/tmp")

global_phonemizer = EspeakBackend(
    language='ar',
    preserve_punctuation=True,
    with_stress=True,
    language_switch='remove-flags'
)

vocab = {}
vocab_index = 0

def convert_batch(batch):
    global vocab, vocab_index
    input_ids_list = []
    phonemes_list = []

    for text in batch["text"]:
        text = re.sub(r'\b_[A-Za-z0-9_-]*_\b', '', text).strip().replace('\n', ' ')

        tokens = simple_word_tokenize(text, split_digits=False)

        ids = []
        phonemes = []
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = vocab_index
                vocab_index += 1
            ids.append(vocab[tok])
            
            ph = global_phonemizer.phonemize([tok], strip=True)[0]
            phonemes.append(ph)
        
        input_ids_list.append(ids)
        phonemes_list.append(phonemes)

    return {"input_ids": input_ids_list, "phonemes": phonemes_list}

print("Processing dataset with phonemizer...")
processed_dataset = dataset.map(
    convert_batch,
    batched=True,
    batch_size=128,
    num_proc=32,
    remove_columns=["text"],
    desc="Processing Wiki40B Arabic"
)

save_path = os.path.join(root_directory, "processed_dataset")
processed_dataset.save_to_disk(save_path)
print(f"Saved processed dataset to: {save_path}")

vocab_file = os.path.join(root_directory, "vocab.txt")
with open(vocab_file, "w", encoding="utf-8") as f:
    for w, idx in vocab.items():
        f.write(f"{w}\t{idx}\n")
print(f"Saved vocabulary to: {vocab_file}")
