import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset

# ---------------------- 词表类 ----------------------
class Vocabulary:
    def __init__(self, word2idx=None, max_size=100000):
        if word2idx is None:
            self.word2idx = {'<pad>': 0, '<unk>': 1}
            self.idx2word = {0: '<pad>', 1: '<unk>'}
            self.max_size = max_size
            self.freq = {}
        else:
            # 从字典初始化（用于测试）
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def build(self, texts):
        """基于训练文本构建词表（仅训练时使用）"""
        for text in texts:
            tokens = text.lower().split()  # 可替换为更复杂分词（如jieba）
            for token in tokens:
                self.freq[token] = self.freq.get(token, 0) + 1

        # 按词频排序并截断
        sorted_words = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        for idx, (word, _) in enumerate(sorted_words[:self.max_size - 2]):
            self.word2idx[word] = idx + 2
            self.idx2word[idx + 2] = word

    def encode(self, text, max_len=200):
        """文本转索引（训练/测试共用）"""
        tokens = text.lower().split()
        ids = [self.word2idx.get(token, 1) for token in tokens]  # 未知词用<unk>
        # 前填充+后截断（与训练时一致）
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = [0] * (max_len - len(ids)) + ids
        return ids


# ---------------------- 位置编码 ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]  # 动态匹配输入长度


# ---------------------- 注意力分类模型 ----------------------
class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, max_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x)  # (B, L, D)
        x = x + self.pos_encoder(x)
        attn_output, _ = self.attention(x, x, x)  # 自注意力
        pooled = torch.mean(attn_output, dim=1)  # 全局平均池化
        return self.classifier(pooled).squeeze(-1)


# ---------------------- 训练数据集类（含标签） ----------------------
class TextDataset(Dataset):
    """用于训练/验证的带标签数据集"""

    def __init__(self, df, vocab, max_len=200):
        self.texts = df['text'].tolist()  # 文本列表
        self.labels = df['label'].tolist()  # 标签列表
        self.vocab = vocab  # 词表对象
        self.max_len = max_len  # 最大序列长度

    def __len__(self):
        return len(self.texts)  # 数据集大小

    def __getitem__(self, idx):
        """获取单个样本（编码后的文本+标签）"""
        text = self.texts[idx]
        label = self.labels[idx]
        # 使用词表的encode方法将文本转换为索引
        ids = self.vocab.encode(text, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# ---------------------- 测试数据集类（无标签） ----------------------
class TestDataset(Dataset):
    """用于测试的无标签数据集"""

    def __init__(self, texts, vocab, max_len=200):
        self.texts = texts  # 文本列表（无标签）
        self.vocab = vocab  # 词表对象
        self.max_len = max_len  # 最大序列长度

    def __len__(self):
        return len(self.texts)  # 数据集大小

    def __getitem__(self, idx):
        """获取单个样本（仅编码后的文本）"""
        text = self.texts[idx]
        ids = self.vocab.encode(text, self.max_len)
        return torch.tensor(ids, dtype=torch.long)

