import json
import torch
from torch.utils.data import DataLoader
from model import BERTClassifier, BERTTestDataset
from transformers import BertTokenizer


def predict_test(test_file, model_ckpt, tokenizer_path, output_file,
                 max_len=256, batch_size=16, device='cuda'):
    """
    对无标签测试集进行预测并保存结果，将概率最高的前1400个样本设置为1，其余设置为0
    """
    # 1. 加载BERT tokenizer - 使用本地路径
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # 2. 加载测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_texts = [json.loads(line)['text'] for line in f]

    # 3. 创建数据加载器
    test_dataset = BERTTestDataset(test_texts, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 加载模型 - 使用本地模型路径
    model = BERTClassifier(model_name=tokenizer_path)  # 使用与tokenizer相同的路径
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # 5. 预测并保存概率
    probabilities = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            probabilities.extend(probs.flatten().tolist())

    # 6. 将概率最高的前1400个设置为1，其余为0
    sorted_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    predictions = [1 if i in sorted_indices[:1400] else 0 for i in range(len(probabilities))]

    # 7. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"预测完成，结果已保存至 {output_file}（共 {len(predictions)} 条）")


if __name__ == "__main__":
    # 使用本地模型路径
    local_model_path = "local_models/bert-base-uncased"

    predict_test(
        test_file='data/test.jsonl',
        model_ckpt='results/best_model.pth',
        tokenizer_path=local_model_path,
        output_file='submit/submit.txt',
        max_len=256,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
