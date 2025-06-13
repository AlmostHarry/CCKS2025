import json
import torch
from torch.utils.data import DataLoader
from model import Vocabulary, AttentionClassifier, TestDataset  # 导入公共模块


def predict_test(test_file, vocab_file, model_ckpt, output_file,
                 max_len=200, batch_size=32, device='cuda'):
    """
    对无标签测试集进行预测并保存结果
    :param test_file: 测试数据路径（test.jsonl）
    :param vocab_file: 训练时保存的词表路径（best_vocab.json）
    :param model_ckpt: 最佳模型权重路径（best_model.pth）
    :param output_file: 预测结果保存路径（submit.txt）
    """
    # 1. 加载词表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    vocab = Vocabulary(word2idx=word2idx)  # 从文件初始化词表

    # 2. 加载测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_texts = [json.loads(line)['text'] for line in f]

    # 3. 创建数据加载器
    test_dataset = TestDataset(test_texts, vocab, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 加载模型
    model = AttentionClassifier(
        vocab_size=len(vocab.word2idx),
        d_model=64,  # 需与训练时一致
        n_heads=16,  # 需与训练时一致
        max_len=max_len
    )
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # 5. 预测并保存结果
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()  # 转换为0/1标签
            predictions.extend(preds.astype(int).tolist())

    # 保存结果（每行一个标签）
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"预测完成，结果已保存至 {output_file}（共 {len(predictions)} 条）")


if __name__ == "__main__":
    predict_test(
        test_file='test.jsonl',
        vocab_file='best_vocab.json',
        model_ckpt='best_model.pth',
        output_file='submit.txt',
        max_len=300,  # 需与训练时一致
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

