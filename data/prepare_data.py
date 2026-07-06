"""
TinyStories 数据准备脚本
- 自动从 HuggingFace 下载数据集
- 抽取固定 100 条验证样本（seed=1，与论文一致）
- 保存为 PyTorch tensor 供下游实验使用
"""

import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, "cache")
VAL_SET_PATH = os.path.join(DATA_DIR, "val_set_100.pt")
VAL_TEXTS_PATH = os.path.join(DATA_DIR, "val_texts_100.pt")


def download_tinystories():
    """下载 TinyStories 数据集。"""
    print("=" * 60)
    print("Downloading TinyStories from HuggingFace...")
    print("=" * 60)
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=CACHE_DIR)
    print(f"  Train size: {len(dataset['train'])}")
    print(f"  Valid size: {len(dataset['validation'])}")
    return dataset


def prepare_validation_set(dataset, n_samples=100, seed=1):
    """抽取固定验证集。"""
    torch.manual_seed(seed)
    val_data = dataset['validation']
    indices = torch.randperm(len(val_data))[:n_samples].tolist()
    texts = [val_data[i]['text'] for i in indices]
    torch.save(texts, VAL_TEXTS_PATH)
    print(f"  Saved {n_samples} validation texts -> {VAL_TEXTS_PATH}")
    return texts


def prepare_tokenized_validation(tokenizer, texts, max_length=256):
    """将验证文本编码为 token IDs。"""
    encoded = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
        encoded.append(tokens)
    torch.save(encoded, VAL_SET_PATH)
    print(f"  Saved tokenized validation set -> {VAL_SET_PATH}")
    return encoded


def prepare_tokenizer():
    """准备 GPT-2 tokenizer。"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    return tokenizer


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset = download_tinystories()
    tokenizer = prepare_tokenizer()
    val_texts = prepare_validation_set(dataset)
    prepare_tokenized_validation(tokenizer, val_texts)
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"  Raw texts:       {VAL_TEXTS_PATH}")
    print(f"  Tokenized:       {VAL_SET_PATH}")
    print(f"  Dataset cache:   {CACHE_DIR}")
    print("\nUse in experiments:")
    print("  val_texts = torch.load('data/val_texts_100.pt')")


if __name__ == '__main__':
    main()
