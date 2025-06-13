import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer, BertConfig


# ---------------------- 词表类 (保留但不使用) ----------------------
class Vocabulary:
    def __init__(self, word2idx=None, max_size=100000):
        if word2idx is None:
            self.word2idx = {'<pad>': 0, '<unk>': 1}
            self.idx2word = {0: '<pad>', 1: '<unk>'}
            self.max_size = max_size
            self.freq = {}
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def build(self, texts):
        for text in texts:
            tokens = text.lower().split()
            for token in tokens:
                self.freq[token] = self.freq.get(token, 0) + 1

        sorted_words = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        for idx, (word, _) in enumerate(sorted_words[:self.max_size - 2]):
            self.word2idx[word] = idx + 2
            self.idx2word[idx + 2] = word

    def encode(self, text, max_len=200):
        tokens = text.lower().split()
        ids = [self.word2idx.get(token, 1) for token in tokens]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = [0] * (max_len - len(ids)) + ids
        return ids


# ---------------------- BERT分类模型 ----------------------
class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', freeze_bert=False):
        super().__init__()
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        self.config = BertConfig.from_pretrained(model_name)

        # 冻结BERT参数（可选）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 增加dropout概率
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 添加额外dropout层
            nn.Linear(256, 1)
        )

        # 初始化分类器权重
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask):
        # 通过BERT模型
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取[CLS]标记的隐藏状态作为整个序列的表示
        pooled_output = outputs.pooler_output

        # 分类层
        logits = self.classifier(pooled_output)
        return logits.squeeze(-1)  # 从(batch_size, 1)变为(batch_size,)


# ---------------------- BERT数据集类 ----------------------
class BERTTextDataset(Dataset):
    """用于BERT训练/验证的带标签数据集"""

    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist() if 'label' in df else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 使用tokenizer编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class BERTTestDataset(Dataset):
    """用于BERT测试的无标签数据集"""

    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
