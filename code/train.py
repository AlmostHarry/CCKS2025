import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BERTClassifier, BERTTextDataset, BERTTestDataset
from transformers import BertTokenizer
from torch.optim import AdamW
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, labels):
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(outputs, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


def evaluate_test(model, test_loader, device='cuda'):
    """评估测试集，并计算正负类比例差异"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
    all_preds = np.array(all_preds)
    pos_ratio = np.mean(all_preds)
    return pos_ratio, all_preds


def train_model(train_loader, test_loader, model, criterion, optimizer, epochs=10, device='cuda'):
    best_pos_ratio_diff = float('inf')
    model.to(device)
    best_model_path = 'results/best_model.pth'

    # 确保保存目录存在
    os.makedirs('results/checkpoints', exist_ok=True)

    # 计算每个epoch的训练步数
    total_steps = len(train_loader)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练指标
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item() * input_ids.size(0)

            # 每50步保存一个checkpoint
            if (step + 1) % 50 == 0:
                checkpoint_path = f'results/checkpoints/model_epoch_{epoch + 1}_step_{step + 1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                # 在checkpoint处评估测试集
                pos_ratio, _ = evaluate_test(model, test_loader, device)
                current_diff = abs(pos_ratio - 0.5)
                print(f"Checkpoint evaluation: Positive ratio {pos_ratio:.4f}, Diff from 0.5: {current_diff:.4f}")

                # 如果当前checkpoint的正负类比例更接近1:1，则保存为最佳模型
                if current_diff < best_pos_ratio_diff:
                    best_pos_ratio_diff = current_diff
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with pos_ratio_diff: {best_pos_ratio_diff:.6f}")

        # Epoch结束时的评估
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}')

        # 每个epoch结束后也评估测试集
        pos_ratio, _ = evaluate_test(model, test_loader, device)
        current_diff = abs(pos_ratio - 0.5)
        print(f"Epoch evaluation: Positive ratio {pos_ratio:.4f}, Diff from 0.5: {current_diff:.4f}")

        # 保存当前epoch模型
        epoch_model_path = f'results/model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Epoch model saved to {epoch_model_path}")

        # 更新最佳模型
        if current_diff < best_pos_ratio_diff:
            best_pos_ratio_diff = current_diff
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with pos_ratio_diff: {best_pos_ratio_diff:.6f}")

    return best_pos_ratio_diff


def main():
    # 超参数
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 16
    LEARNING_RATE = 2e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "local_models/bert-base-uncased"

    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/checkpoints', exist_ok=True)

    # 加载数据
    train_df = pd.DataFrame([json.loads(line) for line in open('data/train.jsonl', 'r', encoding='utf-8')])
    test_texts = [json.loads(line)['text'] for line in open('data/test.jsonl', 'r', encoding='utf-8')]

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # 创建数据加载器
    train_dataset = BERTTextDataset(train_df, tokenizer, MAX_LEN)
    test_dataset = BERTTestDataset(test_texts, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = BERTClassifier(MODEL_PATH, freeze_bert=False)
    criterion = FocalLoss(alpha=0.25, gamma=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 训练模型
    train_model(train_loader, test_loader, model, criterion, optimizer, EPOCHS, DEVICE)

    # 加载最佳模型并生成最终预测
    best_model = BERTClassifier(MODEL_PATH)
    best_model.load_state_dict(torch.load('results/best_model.pth', map_location=DEVICE))
    best_model.to(DEVICE)
    best_model.eval()

    probabilities = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = best_model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            probabilities.extend(probs.flatten().tolist())

    # 根据概率生成0/1预测（超过0.5的为1，否则为0）
    predictions = [1 if p >= 0.5 else 0 for p in probabilities]

    # 保存预测结果
    with open('submit/final_predictions.txt', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"最终预测结果已保存至 submit/final_predictions.txt，共 {len(predictions)} 条")


if __name__ == "__main__":
    main()
