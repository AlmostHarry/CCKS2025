import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BERTClassifier, BERTTextDataset
from transformers import BertTokenizer
from torch.optim import AdamW
import os


def train_model(train_loader, val_loader, model, criterion, optimizer, epochs=10, device='cuda'):
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        # 训练循环
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item() * input_ids.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total

        # 验证循环
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * input_ids.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total

        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')

        # 保存最佳模型（基于验证准确率）
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/best_model.pth')
            print(f'已保存最佳模型（验证准确率：{val_accuracy:.4f}, 验证损失：{val_loss:.4f}）')
            # 保存tokenizer
            tokenizer.save_pretrained('results/bert_tokenizer')

    print(f'\n训练完成！最佳验证准确率: {best_val_accuracy:.4f}, 对应验证损失: {best_val_loss:.4f}')
    return best_val_accuracy


def evaluate_model(loader, model, device='cuda'):
    """评估模型性能，返回准确率、损失和详细分类报告"""
    from sklearn.metrics import classification_report
    import numpy as np

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    # 生成分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=['Human', 'AI'],
        output_dict=True
    )

    # 打印易读的报告
    print("\n详细分类报告:")
    print(f"准确率: {accuracy:.4f}")
    print(f"平均损失: {avg_loss:.4f}")
    print(
        f"人类生成文本识别准确率: {report['Human']['precision']:.4f} (精确率), {report['Human']['recall']:.4f} (召回率)")
    print(f"AI生成文本识别准确率: {report['AI']['precision']:.4f} (精确率), {report['AI']['recall']:.4f} (召回率)")
    print(f"F1分数: {report['macro avg']['f1-score']:.4f} (宏平均)")

    return accuracy, avg_loss, report


if __name__ == "__main__":
    # 超参数
    MAX_LEN = 256  # BERT最大长度限制
    BATCH_SIZE = 32  # BERT需要更小的batch size
    EPOCHS = 4  # BERT通常需要较少的epoch
    LEARNING_RATE = 2e-5  # BERT推荐学习率
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 使用本地模型路径
    MODEL_PATH = "local_models/bert-base-uncased"  # 修改为您的本地路径

    # 确保结果文件夹存在
    os.makedirs('results', exist_ok=True)

    # 1. 加载并划分数据
    df = pd.DataFrame([json.loads(line) for line in open('data/train.jsonl', 'r', encoding='utf-8')])
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 2. 从本地加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    print(f"BERT tokenizer加载完成，词汇表大小: {tokenizer.vocab_size}")

    # 3. 创建数据加载器
    train_dataset = BERTTextDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = BERTTextDataset(val_df, tokenizer, MAX_LEN)
    test_dataset = BERTTextDataset(test_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型、损失函数和优化器
    model = BERTClassifier(MODEL_PATH, freeze_bert=False)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 5. 训练并保存最佳模型
    best_val_acc = train_model(train_loader, val_loader, model, criterion, optimizer, EPOCHS, DEVICE)

    # 6. 加载最佳模型并评估测试集性能
    model.load_state_dict(torch.load('results/best_model.pth'))
    print("\n在测试集上评估最佳模型性能...")
    test_accuracy, test_loss, test_report = evaluate_model(test_loader, model, DEVICE)

    # 7. 保存最终评估结果
    with open('results/performance_report.json', 'w') as f:
        json.dump({
            'best_validation_accuracy': best_val_acc,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': test_report
        }, f, indent=2)

    print(f"\n最终测试集准确率: {test_accuracy:.4f}，结果已保存至 performance_report.json")
