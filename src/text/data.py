from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def try_load_sms_spam():
    """Try a few common dataset identifiers from HuggingFace `datasets`."""
    candidates = [
        "sms_spam",
        "sms_spam_collection",
        "SMS-Spam-Collection",
        "ucirvine/sms_spam",
        "SetFit/sms_spam",
    ]
    last_err = None
    for name in candidates:
        try:
            ds = load_dataset(name)
            return ds, name
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Could not load SMS spam dataset automatically. " 
        "Edit src/text/data.py and set your dataset name or load from local CSV."
    ) from last_err

def normalize_dataset(ds):
    """Return (texts, labels) list for train/val/test from whatever fields exist."""
    # Common field names
    text_keys = ["text", "sms", "message"]
    label_keys = ["label", "labels", "spam"]

    def pick_key(example, keys):
        for k in keys:
            if k in example:
                return k
        return None

    split_names = list(ds.keys())
    if "train" not in split_names:
        # some datasets have only 'train' and 'test' already, or a single split
        pass

    # If there's only train, we will split later.
    split = ds["train"] if "train" in ds else ds[split_names[0]]
    sample = split[0]
    tk = pick_key(sample, text_keys)
    lk = pick_key(sample, label_keys)
    if tk is None or lk is None:
        raise RuntimeError(f"Unknown dataset schema. Available keys: {list(sample.keys())}")

    return tk, lk

def train_val_test_split(texts: List[str], labels: List[int], seed: int = 42,
                         val_ratio: float = 0.15, test_ratio: float = 0.15):
    n = len(texts)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]

    def take(idxs):
        return [texts[i] for i in idxs], [labels[i] for i in idxs]

    return take(train_idx), take(val_idx), take(test_idx)

def build_vocab(texts: List[str], max_size: int = 20000, min_freq: int = 2) -> Dict[str, int]:
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(simple_tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in c.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab

def encode(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids

@dataclass
class TextBatch:
    x: np.ndarray  # (B, T)
    y: np.ndarray  # (B,)
    mask: np.ndarray  # (B, T) 1 for real tokens

def make_batches(texts: List[str], labels: List[int], vocab: Dict[str,int], batch_size: int = 32,
                 max_len: int = 64, shuffle: bool = True, seed: int = 42):
    n = len(texts)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start+batch_size]
        x = np.array([encode(texts[i], vocab, max_len=max_len) for i in batch_idx], dtype=np.int64)
        y = np.array([labels[i] for i in batch_idx], dtype=np.int64)
        mask = (x != 0).astype(np.int64)
        yield TextBatch(x=x, y=y, mask=mask)
