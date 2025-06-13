import os
custom_cache_path = "/root/autodl-tmp/huggingface"
if custom_cache_path:
    os.environ['HUGGINGFACE_HUB_CACHE'] = custom_cache_path
    if not os.path.exists(custom_cache_path):
        os.makedirs(custom_cache_path)
    print(f"Hugging Face Hub cache directory is set to: {os.getenv('HUGGINGFACE_HUB_CACHE')}")
import torch
import numpy as np
import shutil
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback 
)
import transformers
print(f"Transformers version used by the script: {transformers.__version__}")
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate
import inspect

# --- 1. 配置部分 ---
model_name_or_path = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
train_file_path = "/root/autodl-tmp/CCKs/data/train.jsonl"
predict_file_path = "/root/autodl-tmp/CCKs/data/test.jsonl"
output_dir = "./qwen2-8b-qlora-custom-callback-v2-2" # 使用一个新的输出目录



# --- 2. 加载和准备数据 ---
print("--- Step 2: Loading and Preparing Data ---")
# 2.1 只加载训练数据
raw_train_dataset = load_dataset("json", data_files={"train": train_file_path})["train"]
print(f"Total training samples: {len(raw_train_dataset)}")

# 2.2 加载无标签的测试数据 (用于回调和最终预测)
raw_predict_dataset = load_dataset("json", data_files={"predict": predict_file_path})["predict"]
print(f"Predict samples (no labels): {len(raw_predict_dataset)}")

# 2.3 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
TOKENIZATION_MAX_LENGTH = 512

# 2.4 定义预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=TOKENIZATION_MAX_LENGTH)

# 2.5 应用预处理
print("\nTokenizing datasets...")
tokenized_train_dataset = raw_train_dataset.map(preprocess_function, batched=True, remove_columns=raw_train_dataset.column_names)
tokenized_predict_dataset = raw_predict_dataset.map(preprocess_function, batched=True, remove_columns=raw_predict_dataset.column_names)
tokenized_train_dataset = tokenized_train_dataset.add_column("labels", raw_train_dataset["label"]) 

tokenized_train_dataset.set_format("torch")
tokenized_predict_dataset.set_format("torch")
print("Data preparation complete.")

# --- 3. 配置QLoRA并加载模型 ---
print("\n--- Step 3: Configuring QLoRA and Loading Model ---")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, quantization_config=bnb_config, num_labels=2, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(r=128, lora_alpha=256, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
print("Model and QLoRA configuration complete.")

class SaveSubmissionsCallback(TrainerCallback):
    # 修改 1: 在构造函数中接收 model 和 tokenizer
    def __init__(self, predict_dataset, model, tokenizer, target_count=1400, output_dir_base=None):
        super().__init__()
        self.predict_dataset = predict_dataset
        self.model = model             # 保存 model
        self.tokenizer = tokenizer     # 保存 tokenizer
        self.target_count = target_count
        self.output_dir_base = output_dir_base
        self.closest_count_diff = float('inf')
        self.best_checkpoint_info = {}
        self.history_log_path = os.path.join(self.output_dir_base, "checkpoint_prediction_log.txt")
        with open(self.history_log_path, "w") as f:
            f.write("Checkpoint Prediction Log\n" + "="*50 + "\n")

    # 修改 2: 修正 on_save 的函数签名，并使用 self.model 和 self.tokenizer
    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        checkpoint_step = state.global_step
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{checkpoint_step}")
        
        print(f"\n--- [Callback] Running Prediction for Checkpoint {checkpoint_step} ---")
        
        # 直接使用保存在 self 中的 model 和 tokenizer
        temp_trainer = Trainer(model=self.model, args=args, tokenizer=self.tokenizer)
        
        predictions_output = temp_trainer.predict(test_dataset=self.predict_dataset)
        predicted_class_ids = np.argmax(predictions_output.predictions, axis=1)
        count_of_label_1 = np.sum(predicted_class_ids == 1)
        current_diff = abs(count_of_label_1 - self.target_count)
        
        submission_filename = f"submit_checkpoint_{checkpoint_step}.txt"
        submission_filepath = os.path.join(checkpoint_path, submission_filename)
        with open(submission_filepath, "w") as writer:
            for class_id in predicted_class_ids:
                writer.write(f"{class_id}\n")
        
        log_message = (
            f"Checkpoint: {checkpoint_path}\n"
            f"   - Predicted Label 1s: {count_of_label_1}\n"
            f"   - Difference from Target({self.target_count}): {current_diff}\n"
            f"   - Submission file saved to: {submission_filepath}\n"
        )
        print(log_message)
        
        if current_diff < self.closest_count_diff:
            self.closest_count_diff = current_diff
            self.best_checkpoint_info = {
                "path": checkpoint_path,
                "count": count_of_label_1,
                "diff": current_diff,
                "submission_file": submission_filepath
            }
            print(f"***** [Callback] New best checkpoint found: {self.best_checkpoint_info['path']} with a difference of {self.closest_count_diff} *****\n")
            log_message += "***** NEW BEST *****\n"

        with open(self.history_log_path, "a") as f:
            f.write(log_message + "-"*50 + "\n")


# --- 4. 设置训练参数和 Trainer ---
print("\n--- Step 4: Setting up Training Arguments and Trainer ---")
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 3

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",        
    save_steps=30,               # 每100步保存一次检查点，并触发回调
    save_total_limit=None,        # 保存所有检查点
    eval_strategy="no",     
    load_best_model_at_end=False, 
    report_to="tensorboard",
    bf16=True,
    fp16=False,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
)

# 实例化回调，并传入输出目录
# 修改 3: 在创建回调实例时，传入 model 和 tokenizer
custom_callback = SaveSubmissionsCallback(
    predict_dataset=tokenized_predict_dataset, 
    model=peft_model,      # 在这里传入 model
    tokenizer=tokenizer,   # 在这里传入 tokenizer
    target_count=1400, 
    output_dir_base=output_dir
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    compute_metrics=None,
    callbacks=[custom_callback],
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

# --- 6. 报告最终结果 ---
print("\n--- Step 6: Final Result Summary ---")
best_info = custom_callback.best_checkpoint_info

if best_info:
    print("="*60)
    print("           BEST CHECKPOINT FOUND DURING TRAINING")
    print("="*60)
    print(f"  Checkpoint Path: {best_info['path']}")
    print(f"  Predicted Label 1s: {best_info['count']} (Closest to target 1400)")
    print(f"  Difference from Target: {best_info['diff']}")
    print(f"  Corresponding Submission File: {best_info['submission_file']}")
    print("\nTraining process finished. You can now use the submission file from the best checkpoint.")
else:
    print("No checkpoints were saved, or the callback did not find a best checkpoint.")

print("\n--- Script Finished ---")