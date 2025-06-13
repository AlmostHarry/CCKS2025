import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments, # 我们仍然用它来配置一些预测参数，比如批处理大小
    Trainer,
)
from peft import PeftModel # 用于加载LoRA适配器
import transformers
print(f"Transformers version used by the script: {transformers.__version__}")
import evaluate # 即使不计算指标，Trainer内部也可能需要它
import inspect # 用于调试，可以保留或移除

# --- 1. 配置部分 (请根据您的设置修改) ---
# 基础模型的ID或本地路径 (在QLoRA训练时使用的原始模型)
#base_model_id_or_path = "mistralai/Mistral-7B-v0.1"
# 或者，如果您之前将基础模型下载到了特定路径，例如：
base_model_id_or_path = "/root/autodl-tmp/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"

# 微调后保存的LoRA适配器权重和分词器的路径 (根据您的截图)
adapter_and_tokenizer_path = "./mistral-7b-qlora-text-detection-output-with-validation-full/checkpoint-1575"

# 完整的、无标签的测试数据文件路径
predict_file_path = "/root/autodl-tmp/CCKs/data/test.jsonl"

# 预测结果输出目录和文件名
output_dir_predict = "./predictions_on_full_testset_1epoch" # 新的输出目录
submission_file_path = f"{output_dir_predict}/submit.txt"

# 可选: 自定义缓存目录 (与训练时保持一致，确保基础模型能被找到)
custom_cache_path = "/root/autodl-tmp/huggingface"
if custom_cache_path:
     os.environ['HUGGINGFACE_HUB_CACHE'] = custom_cache_path
     if not os.path.exists(custom_cache_path):
         os.makedirs(custom_cache_path)
     print(f"Hugging Face Hub cache directory is set to: {os.getenv('HUGGINGFACE_HUB_CACHE')}")

if not os.path.exists(output_dir_predict):
    os.makedirs(output_dir_predict)

TOKENIZATION_MAX_LENGTH = 512 # 与训练时使用的最大长度保持一致

# --- 2. 加载分词器 (从保存的适配器路径加载，以确保一致性) ---
print("--- Step 2: Loading Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(adapter_and_tokenizer_path)
# 如果训练时设置了pad_token，这里加载的tokenizer应该已经包含了
# 但以防万一，可以再次检查和设置
if tokenizer.pad_token is None:
    print("Tokenizer pad_token was None, setting to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

# --- 3. 加载基础模型并应用QLoRA适配器 ---
print("\n--- Step 3: Loading Base Model and Applying QLoRA Adapters ---")
# BNB配置需要与训练时一致
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {base_model_id_or_path} with quantization...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id_or_path,
    quantization_config=bnb_config,
    num_labels=2, # 确保与训练时一致
    device_map="auto", # 自动将模型放到可用GPU上
)

if base_model.config.pad_token_id is None:
    base_model.config.pad_token_id = tokenizer.eos_token_id
print("Base model loaded.")

print(f"Loading PEFT model from adapter path: {adapter_and_tokenizer_path}...")
# 加载LoRA适配器到基础模型上
model = PeftModel.from_pretrained(base_model, adapter_and_tokenizer_path)
model = model.eval() # 设置为评估模式，这很重要！
print("PEFT model with adapters loaded and set to evaluation mode.")

# --- 4. 加载和准备预测数据 (完整的测试集) ---
print("\n--- Step 4: Loading and Preparing Full Test Data for Prediction ---")
raw_predict_dataset = load_dataset("json", data_files={"predict": predict_file_path})["predict"]
print(f"Predict samples (no labels): {len(raw_predict_dataset)}")

def preprocess_function_for_predict(examples):
    return tokenizer(
        examples["text"], # 确保JSON中的文本字段是 "text"
        truncation=True,
        padding="max_length", # 或者 "longest" 如果你想动态填充
        max_length=TOKENIZATION_MAX_LENGTH,
        return_tensors="pt" # Trainer.predict 可能不需要这个，但手动处理时需要
    )

print("Tokenizing predict dataset...")
# 注意：如果数据集很大，直接 .map().set_format() 可能会占用较多内存
# Trainer.predict 内部会处理数据加载和批处理，所以这里主要确保分词正确
# 对于 Trainer.predict, remove_columns 最好在 map 中完成
tokenized_predict_dataset = raw_predict_dataset.map(
    preprocess_function_for_predict,
    batched=True,
    remove_columns=raw_predict_dataset.column_names # 移除原始列，只保留tokenizer输出
)
# Trainer.predict 会处理 set_format，但如果手动迭代则需要
# tokenized_predict_dataset.set_format("torch")
print("Predict data preparation complete.")

# --- 5. 设置Trainer (仅用于预测) 和进行预测 ---
print("\n--- Step 5: Setting up Trainer and Performing Prediction ---")
# 只需要最基本的TrainingArguments来进行预测
# 24GB 显存，可以尝试更大的 per_device_eval_batch_size
# 如果遇到OOM，可以减小这个值
PREDICT_BATCH_SIZE = 8 # 例如，可以根据您的24GB显存调整

predict_args = TrainingArguments(
    output_dir=output_dir_predict, # 临时目录，预测时也需要
    per_device_eval_batch_size=PREDICT_BATCH_SIZE,
    report_to="none", # 预测时不需要报告给tensorboard等
    # bf16=torch.cuda.is_bf16_supported(), # 可选，用于预测时的精度
    # fp16=False, # 如果 bf16 为 True，这个应为 False
)

trainer = Trainer(
    model=model, # 注意这里是已经加载了适配器的 peft_model
    args=predict_args,
    tokenizer=tokenizer, # 传递tokenizer给Trainer是很重要的
    # train_dataset, eval_dataset, compute_metrics 在这里不需要
)
print("Trainer for prediction setup complete.")

print("Starting prediction on the full test set...")
predictions_output = trainer.predict(test_dataset=tokenized_predict_dataset)
print("Prediction complete.")

# --- 6. 处理预测结果并保存 ---
print("\n--- Step 6: Processing Predictions and Saving Submission File ---")
logits = predictions_output.predictions
predicted_class_ids = np.argmax(logits, axis=1)

# (可选) 检查预测数量是否与原始测试集数量一致
if len(predicted_class_ids) != len(raw_predict_dataset):
    print(f"Warning: Number of predictions ({len(predicted_class_ids)}) does not match number of original test samples ({len(raw_predict_dataset)}). Please check.")

with open(submission_file_path, "w") as writer:
    for class_id in predicted_class_ids:
        writer.write(f"{class_id}\n")

print(f"Submission file saved to {submission_file_path}")
print("\n--- Script Finished ---")