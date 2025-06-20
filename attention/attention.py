import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Vocabulary, AttentionClassifier, TextDataset  # 导入公共模块


def train_model(train_loader, val_loader, model, criterion, optimizer, vocab, epochs=10,
                device='cuda'):  # Added vocab to save
    best_val_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        # 训练循环
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'已保存最佳模型（验证损失：{val_loss:.4f}）')
            with open('best_vocab.json', 'w', encoding='utf-8') as f:
                json.dump(vocab.word2idx, f, ensure_ascii=False)  # vocab is now passed to train_model

    return best_val_loss


def evaluate_accuracy(loader, model, device='cuda'):
    """计算指定数据加载器的分类准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 二分类场景使用sigmoid激活+0.5阈值判断
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()  # 转换为0/1预测值
            correct += (preds == labels).sum().item()
            total += labels.size(0)  # 累计总样本数
    return correct / total  # 返回准确率


if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")  # 输出应为 True（表示支持 CUDA）
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")  # 输出 CUDA 版本号（如 11.7）

    # 超参数（可根据需求调整）
    MAX_VOCAB_SIZE = 10000
    MAX_LEN = 300
    BATCH_SIZE = 128
    D_MODEL = 64
    N_HEADS = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 1. 加载并划分数据
    df = pd.DataFrame([json.loads(line) for line in open('train.jsonl', 'r', encoding='utf-8')])
    # 只划分为训练集和验证集 (e.g., 80% train, 20% val)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2. 构建词表（仅用训练集）
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
    vocab.build(train_df['text'].tolist())

    # 3. 创建数据加载器
    train_dataset = TextDataset(train_df, vocab, MAX_LEN)
    val_dataset = TextDataset(val_df, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型、损失函数和优化器
    model = AttentionClassifier(
        vocab_size=len(vocab.word2idx),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        max_len=MAX_LEN
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练并保存最佳模型
    print("开始训练模型...")
    train_model(train_loader, val_loader, model, criterion, optimizer, vocab, EPOCHS, DEVICE)  # Pass vocab

    # 6. 加载最佳模型并评估验证集准确率 (可选, 因为训练过程中已经打印了验证损失)
    model.load_state_dict(torch.load('best_model.pth'))
    val_accuracy = evaluate_accuracy(val_loader, model, DEVICE)
    print(f'\n最终验证集准确率 (使用加载的最佳模型): {val_accuracy:.4f}')




