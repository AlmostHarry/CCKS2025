import os
custom_cache_path = "/root/autodl-tmp/huggingface"
if custom_cache_path:
     os.environ['HUGGINGFACE_HUB_CACHE'] = custom_cache_path
     if not os.path.exists(custom_cache_path):
         os.makedirs(custom_cache_path)
     print(f"Hugging Face Hub cache directory is set to: {os.getenv('HUGGINGFACE_HUB_CACHE')}")
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

# --- 1. 配置部分 (请根据您的设置修改) ---

# --- 用于快速测试的参数 ---
USE_SMALL_SUBSET = False # 设置为 True 以使用小数据集进行测试
DATA_SUBSET_FRACTION = 20 # 使用 1/20 的数据
NUM_TRAIN_EPOCHS_FOR_TEST = 1 # 快速测试时只跑1个epoch
LOGGING_STEPS_FOR_TEST = 5 # 小数据集时，更频繁地记录日志
# --- 结束快速测试参数 ---

model_name_or_path = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" 
train_file_path = "/root/autodl-tmp/qwen/train.jsonl"
predict_file_path = "/root/autodl-tmp/CCKs/data/test.jsonl"
output_dir = "./qwen2-8b-qlora-text-detection-full-test-revise-train" # 修改输出目录名以作区分



# --- 2. 加载和准备数据 ---
print("--- Step 2: Loading and Preparing Data ---")

# --- 2. 加载和准备数据 ---
print("--- Step 2: Loading and Preparing Data ---")

# 2.1 加载带标签的训练数据
print("Loading training data...")
raw_train_val_dataset_full = load_dataset("json", data_files={"train": train_file_path})["train"]
print(f"Total samples loaded from train.jsonl: {len(raw_train_val_dataset_full)}")

# --- 新增：按您的要求手动创建训练集和验证集 ---

# 1. 首先，按标签将原始数据集筛选为两个子集
print("\nFiltering dataset by label...")
human_dataset = raw_train_val_dataset_full.filter(lambda example: example['label'] == 0)
ai_dataset = raw_train_val_dataset_full.filter(lambda example: example['label'] == 1)
print(f"Found {len(human_dataset)} samples for label 0 (Human).")
print(f"Found {len(ai_dataset)} samples for label 1 (AI).")

# 2. 为了随机抽样，先将两个子集各自打乱
human_dataset = human_dataset.shuffle(seed=42)
ai_dataset = ai_dataset.shuffle(seed=42)

# 3. 定义您希望在验证集中每个类别的样本数量
val_human_samples_count = 4480
val_ai_samples_count = 1120

# 4. 从打乱后的子集中抽取指定数量的样本作为验证集
print(f"\nSelecting {val_human_samples_count} human samples and {val_ai_samples_count} AI samples for validation set...")
val_human_subset = human_dataset.select(range(val_human_samples_count))
val_ai_subset = ai_dataset.select(range(val_ai_samples_count))

# 5. 将剩余的样本作为训练集
train_human_subset = human_dataset.select(range(val_human_samples_count, len(human_dataset)))
train_ai_subset = ai_dataset.select(range(val_ai_samples_count, len(ai_dataset)))

# 6. 将抽样出的子集合并，形成最终的训练集和验证集
from datasets import concatenate_datasets

train_dataset_for_trainer = concatenate_datasets([train_human_subset, train_ai_subset])
eval_dataset_for_trainer = concatenate_datasets([val_human_subset, val_ai_subset])

# 7. (可选但推荐) 将合并后的数据集再次打乱，避免模型按顺序学习
train_dataset_for_trainer = train_dataset_for_trainer.shuffle(seed=42)
eval_dataset_for_trainer = eval_dataset_for_trainer.shuffle(seed=42)

# --- 手动创建结束 ---

# 验证一下我们创建的数据集是否符合要求
print("\n--- Verifying new dataset splits ---")
print(f"Final new training samples: {len(train_dataset_for_trainer)}")
print(f"Final validation samples: {len(eval_dataset_for_trainer)}")

# 计算验证集中标签的分布
val_labels = eval_dataset_for_trainer['label']
val_label_0_count = val_labels.count(0)
val_label_1_count = val_labels.count(1)
print(f"Validation set distribution: {val_label_0_count} samples for label 0, {val_label_1_count} samples for label 1.")
print(f"Ratio of label 0 to label 1 in validation set: {val_label_0_count / val_label_1_count:.2f} : 1")

# 2.2 加载无标签的测试数据 (这部分保持不变)
print("\nLoading test data for final prediction...")
raw_predict_dataset = load_dataset("json", data_files={"predict": predict_file_path})["predict"]
print(f"Predict samples (no labels): {len(raw_predict_dataset)}")

# ... 后续的 tokenizer 加载和数据预处理代码保持不变 ...
# 2.2 加载无标签的测试数据
print("\nLoading test data for final prediction...")
raw_predict_dataset_full = load_dataset("json", data_files={"predict": predict_file_path})["predict"]

if USE_SMALL_SUBSET:
    num_total_predict_samples = len(raw_predict_dataset_full)
    subset_size_predict = num_total_predict_samples // DATA_SUBSET_FRACTION
    raw_predict_dataset = raw_predict_dataset_full.select(range(subset_size_predict))
    print(f"Using a subset of predict data for quick test: {len(raw_predict_dataset)} samples (1/{DATA_SUBSET_FRACTION} of original)")
else:
    raw_predict_dataset = raw_predict_dataset_full
print(f"Predict samples (no labels): {len(raw_predict_dataset)}")

# 2.3 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

TOKENIZATION_MAX_LENGTH = 512

# 2.4 定义预处理函数 (这部分不需要改变)
def preprocess_function_with_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=TOKENIZATION_MAX_LENGTH)
    if "label" in examples:
        tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

def preprocess_function_for_predict(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=TOKENIZATION_MAX_LENGTH)

# 2.5 应用预处理 (这部分不需要改变)
print("\nTokenizing datasets...")
tokenized_train_dataset = train_dataset_for_trainer.map(preprocess_function_with_labels, batched=True, remove_columns=train_dataset_for_trainer.column_names)
tokenized_eval_dataset = eval_dataset_for_trainer.map(preprocess_function_with_labels, batched=True, remove_columns=eval_dataset_for_trainer.column_names)
tokenized_predict_dataset = raw_predict_dataset.map(preprocess_function_for_predict, batched=True, remove_columns=raw_predict_dataset.column_names)

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")
tokenized_predict_dataset.set_format("torch")

print("Data preparation complete.")

# --- 3. 配置QLoRA并加载模型 ---
print("\n--- Step 3: Configuring QLoRA and Loading Model ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    num_labels=2,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
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
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4

# 如果是快速测试，使用测试的epoch数，否则使用正常的epoch数
current_num_train_epochs = NUM_TRAIN_EPOCHS_FOR_TEST if USE_SMALL_SUBSET else 4
current_logging_steps = LOGGING_STEPS_FOR_TEST if USE_SMALL_SUBSET else 25

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    num_train_epochs=current_num_train_epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=current_logging_steps,
    eval_strategy="steps",    # <--- 修改这里：从 "epoch" 改为 "steps"
    eval_steps=350,                 # <--- 新增这里：每100步评估一次
    save_strategy="steps",          # <--- 修改这里：从 "epoch" 改为 "steps"
    save_steps=350,                 # <--- 新增这里：每100步保存一次检查点
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard",
    bf16=True,
    fp16=False,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    label_names=["labels"],
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)["accuracy"]
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels, average="binary")["f1"]
    precision = evaluate.load("precision").compute(predictions=predictions, references=labels, average="binary")["precision"]
    recall = evaluate.load("recall").compute(predictions=predictions, references=labels, average="binary")["recall"]
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
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
    raise

# --- 6. 获取预测并保存以供提交 ---
print("\n--- Step 6: Getting Predictions for Submission ---")
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

final_best_model_path = f"{output_dir}/final_best_model"
trainer.save_model(final_best_model_path)
tokenizer.save_pretrained(final_best_model_path)
print(f"Final best model and tokenizer saved to {final_best_model_path}")

print("\n--- Script Finished ---")