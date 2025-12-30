import pdb
from typing import Dict, Iterable, Callable
from collections import Counter
from itertools import chain
import numpy as np
import torch
import seaborn as sns
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import wandb


class Attention(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._attention = None

        model.option_selector.option_dt.register_forward_hook(self.save_attention_hook())

    def save_attention_hook(self) -> Callable:
        def fn(model, input, output):
            self._attention = output[-1]
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._attention


def get_tokens(inputs, tokenizer):
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)[1:]
    return tokens


def viz_matrix(words_dict, num_options, step, skip_words, figure_title=''):
    # skip_words = ['go', 'to', 'the', 'a', '[SEP]']
    words = sorted(set(chain(*words_dict.values())) - set(skip_words))

    def w_to_ind(word):
        return words.index(word)

    matrix = np.zeros([len(words), num_options])

    for o in range(num_options):
        for w in words_dict[o]:
            if w not in skip_words:
                matrix[w_to_ind(w), o] += 1

    # plot co-occurence matrix (words x options)
    plt.figure(figsize=(30, 10))
    sns.heatmap(matrix, yticklabels=words)
    # plt.plot()
    wandb.log({f"Correlation Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # Now if we normalize it by column (word freq for each option)
    plt.figure(figsize=(30, 10))
    matrix_norm_col = (matrix)/(matrix.sum(axis=0, keepdims=True) + 1e-6)
    sns.heatmap(matrix_norm_col, yticklabels=words)
    wandb.log({f"Word Freq Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # Now if we normalize it by row (option freq for each word)
    plt.figure(figsize=(30, 10))
    matrix_norm_row = (matrix)/(matrix.sum(axis=1, keepdims=True) + 1e-6)
    sns.heatmap(matrix_norm_row, yticklabels=words)
    wandb.log({f"Option Freq Matrix {figure_title}": wandb.Image(plt)}, step=step)
    plt.close()


def viz_matrix2(words_dict, num_options, step, skip_words, figure_title='', labelsize=15):  # labelsize=12
    # skip_words = ['go', 'to', 'the', 'a', '[SEP]']
    skip_words.extend(['and',',','.','t'])
    words_set = set(chain(*words_dict.values()))
    for item in words_set:
        if item.startswith('##'):
            skip_words.append(item)

    words = sorted(set(chain(*words_dict.values())) - set(skip_words))
    # "/home/zxliang/new-code/LISA-hot/lisa"

    def w_to_ind(word):
        return words.index(word)

    matrix = np.zeros([len(words), num_options])

    for o in range(num_options):
        for w in words_dict[o]:
            if w not in skip_words:
                matrix[w_to_ind(w), o] += 1

    # pdb.set_trace()

    words = ["faucet" if x == "fa" else x for x in words]

    # plot co-occurence matrix (words x options)
    plt.figure(figsize=(20, 10))
    plt.tick_params(labelsize=labelsize)
    ax=sns.heatmap(matrix, yticklabels=words)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=labelsize)
    # plt.plot()
    wandb.log({f"Correlation Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # Now if we normalize it by column (word freq for each option)
    plt.figure(figsize=(20, 10))
    plt.tick_params(labelsize=labelsize)
    matrix_norm_col = (matrix)/(matrix.sum(axis=0, keepdims=True) + 1e-6)
    ax=sns.heatmap(matrix_norm_col, yticklabels=words)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=labelsize)
    wandb.log({f"Word Freq Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # Now if we normalize it by row (option freq for each word)
    plt.figure(figsize=(20, 10))
    plt.tick_params(labelsize=labelsize)
    matrix_norm_row = (matrix)/(matrix.sum(axis=1, keepdims=True) + 1e-6)
    ax=sns.heatmap(matrix_norm_row, yticklabels=words)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=labelsize)
    wandb.log({f"Option Freq Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # plt.show()
    plt.close()

def viz_matrix2_weighted(words_dict, num_options, step, skip_words, figure_title='',labelsize=15):  # labelsize=12
    skip_words.extend(['and',',','.','t'])
    
    # 1. 从新的数据结构中提取所有词语
    all_words_with_scores = [item for sublist in words_dict.values() for item in sublist]
    words_set = {list(d.keys())[0] for d in all_words_with_scores}

    for item in list(words_set): # Iterate over a copy
        if item.startswith('##'):
            skip_words.append(item)

    words = sorted(words_set - set(skip_words))
    # "/home/zxliang/new-code/LISA-hot/lisa"

    def w_to_ind(word):
        # 使用 try-except 避免因词语被跳过而引发错误
        try:
            return words.index(word)
        except ValueError:
            return -1

    matrix = np.zeros([len(words), num_options])

    # 2. 累加每个词语在对应技能下的注意力分数
    for o in range(num_options):
        for item_dict in words_dict[o]:
            # 每个 item_dict 是一个 {word: score} 形式的字典
            word, score = list(item_dict.items())[0]
            if word not in skip_words:
                idx = w_to_ind(word)
                if idx != -1:
                    matrix[idx, o] += score

    # pdb.set_trace()

    words = ["faucet" if x == "fa" else x for x in words]

    # plot co-occurence matrix (words x options)
    plt.figure(figsize=(20, 10))
    plt.tick_params(labelsize=labelsize)
    ax=sns.heatmap(matrix, yticklabels=words)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=labelsize)
    # plt.plot()
    wandb.log({f"Correlation Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # Now if we normalize it by column (word freq for each option)
    plt.figure(figsize=(20, 10))
    plt.tick_params(labelsize=labelsize)
    matrix_norm_col = (matrix)/(matrix.sum(axis=0, keepdims=True) + 1e-6)
    ax=sns.heatmap(matrix_norm_col, yticklabels=words)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=labelsize)
    wandb.log({f"Word Freq Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # Now if we normalize it by row (option freq for each word)
    plt.figure(figsize=(20, 10))
    plt.tick_params(labelsize=labelsize)
    matrix_norm_row = (matrix)/(matrix.sum(axis=1, keepdims=True) + 1e-6)
    ax=sns.heatmap(matrix_norm_row, yticklabels=words)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=labelsize)
    wandb.log({f"Option Freq Matrix {figure_title}": wandb.Image(plt)}, step=step)

    # plt.show()
    plt.close()


def plot_hist(stats):
    plt.clf()
    plt.bar(stats.keys(), stats.values())
    plt.xticks(rotation='vertical')
    return wandb.Image(plt)
