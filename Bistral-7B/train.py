import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
import transformers
print(f"Transformers version used by the script: {transformers.__version__}")
from peft import LoraConfig, get_peft_model, TaskType
import evaluate # Hugging Face的评估库
import inspect

print(f"TrainingArguments class loaded from: {inspect.getfile(TrainingArguments)}")
sig = inspect.signature(TrainingArguments.__init__)
print(f"TrainingArguments __init__ signature: {sig}")

# --- 1. 配置部分 (请根据您的设置修改) ---
#model_name_or_path = "mistralai/Mistral-7B-v0.1" # 或者您的本地模型路径
model_name_or_path = "/root/autodl-tmp/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b" # 修改输出目录名以作区分

train_file_path = "/root/autodl-tmp/CCKs/data/train.jsonl"
predict_file_path = "/root/autodl-tmp/CCKs/data/test.jsonl" # 这是无标签的，用于最终预测
output_dir = "./mistral-7b-qlora-text-detection-output-with-validation-full"

custom_cache_path = "/root/autodl-tmp/huggingface"
if custom_cache_path:
     os.environ['HUGGINGFACE_HUB_CACHE'] = custom_cache_path
     if not os.path.exists(custom_cache_path):
         os.makedirs(custom_cache_path)
     print(f"Hugging Face Hub cache directory is set to: {os.getenv('HUGGINGFACE_HUB_CACHE')}")

# --- 2. 加载和准备数据 ---
print("--- Step 2: Loading and Preparing Data ---")

# 2.1 加载带标签的训练数据并划分为训练集和验证集
print("Loading training data for splitting into train/validation...")
raw_train_val_dataset = load_dataset("json", data_files={"train": train_file_path})["train"]
# 划分训练集和验证集 (例如 90% 训练, 10% 验证)
train_val_split = raw_train_val_dataset.train_test_split(test_size=0.2, seed=42) # seed保证可复现
train_dataset_for_trainer = train_val_split["train"]
eval_dataset_for_trainer = train_val_split["test"] # 这个是带标签的验证集

print(f"Original training samples: {len(raw_train_val_dataset)}")
print(f"New training samples: {len(train_dataset_for_trainer)}")
print(f"Validation samples: {len(eval_dataset_for_trainer)}")

# 2.2 加载无标签的测试数据 (仅用于最终预测)
print("\nLoading test data for final prediction...")
raw_predict_dataset = load_dataset("json", data_files={"predict": predict_file_path})["predict"]
print(f"Predict samples (no labels): {len(raw_predict_dataset)}")

# 2.3 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

TOKENIZATION_MAX_LENGTH = 512

# 2.4 定义预处理函数
# 这个函数用于处理带标签的数据 (训练集和验证集)
def preprocess_function_with_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=TOKENIZATION_MAX_LENGTH
    )
    if "label" in examples: # 确保只在有label时添加
        tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

# 这个函数用于处理无标签的数据 (最终预测的测试集)
def preprocess_function_for_predict(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=TOKENIZATION_MAX_LENGTH
    )
    return tokenized_inputs

# 2.5 应用预处理
print("\nTokenizing datasets...")
tokenized_train_dataset = train_dataset_for_trainer.map(preprocess_function_with_labels, batched=True, remove_columns=train_dataset_for_trainer.column_names)
tokenized_eval_dataset = eval_dataset_for_trainer.map(preprocess_function_with_labels, batched=True, remove_columns=eval_dataset_for_trainer.column_names)
tokenized_predict_dataset = raw_predict_dataset.map(preprocess_function_for_predict, batched=True, remove_columns=raw_predict_dataset.column_names)

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")
tokenized_predict_dataset.set_format("torch")

print("Data preparation complete.")
print(f"Tokenized training samples: {len(tokenized_train_dataset)}")
print(f"Tokenized validation samples: {len(tokenized_eval_dataset)}")
print(f"Tokenized predict samples: {len(tokenized_predict_dataset)}")


# --- 3. 配置QLoRA并加载模型 ---
print("\n--- Step 3: Configuring QLoRA and Loading Model ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    num_labels=2,
    device_map="auto",
)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
print("Model and QLoRA configuration complete.")

# --- 4. 设置训练参数和 Trainer ---
print("\n--- Step 4: Setting up Training Arguments and Trainer ---")
# 您的服务器有24GB显存，可以适当调整批处理大小
# PER_DEVICE_TRAIN_BATCH_SIZE 仍然可以从较小值开始，例如 2 或 4，然后根据显存占用调整
# GRADIENT_ACCUMULATION_STEPS 相应调整以保持有效批处理大小
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # 24GB显存可以尝试稍大一点
GRADIENT_ACCUMULATION_STEPS = 8 # 有效批处理大小 2 * 8 = 16
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 4 # 保持4个epoch，或者根据验证集表现调整

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=25, # 可以根据新的 (总步数/epoch数) * 0.01 来调整，或者保持一个固定值
    eval_strategy="epoch", # 每个epoch在验证集上评估
    save_strategy="epoch",
    load_best_model_at_end=True, # 加载在验证集上表现最佳的模型
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="tensorboard",
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    label_names=["labels"], # 显式告诉Trainer标签列的名称
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")["recall"]
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset, # 使用新的训练集
    eval_dataset=tokenized_eval_dataset,   # 使用新的、带标签的验证集
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Trainer setup complete.")

# --- 5. 开始训练 ---
print("\n--- Step 5: Starting Training ---")
try:
    trainer.train()
    print("Training complete.")
except Exception as e:
    print(f"训练过程中发生错误: {e}")
    print("请检查显存占用、批处理大小、序列长度等设置。")
    raise

# --- 6. 获取预测并保存以供提交 (使用原始的、无标签的测试集) ---
print("\n--- Step 6: Getting Predictions for Submission ---")

# 注意：此时 trainer.model 已经是训练过程中在验证集上表现最好的模型了
# (因为 load_best_model_at_end=True)

# 移除任何可能存在的 "labels" 列 (尽管 preprocess_function_for_predict 不应该添加它)
predict_dataset_for_submission = tokenized_predict_dataset
if "labels" in predict_dataset_for_submission.column_names:
    predict_dataset_for_submission = predict_dataset_for_submission.remove_columns(["labels"])

predictions_output = trainer.predict(test_dataset=predict_dataset_for_submission)
logits = predictions_output.predictions
predicted_class_ids = np.argmax(logits, axis=1)

submission_file_path = f"{output_dir}/submit.txt"
with open(submission_file_path, "w") as writer:
    for class_id in predicted_class_ids:
        writer.write(f"{class_id}\n")
print(f"Predictions saved to {submission_file_path}")

# （可选）保存最终由Trainer加载的最佳模型和分词器
final_best_model_path = f"{output_dir}/final_best_model"
trainer.save_model(final_best_model_path) # 这会保存PEFT适配器
tokenizer.save_pretrained(final_best_model_path)
print(f"Final best model and tokenizer saved to {final_best_model_path}")

print("\n--- Script Finished ---")
