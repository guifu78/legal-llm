# -*- coding: utf-8 -*-
"""法律大模型CPU训练脚本"""

import os
import yaml
import torch
from pathlib import Path

project_root = Path(".")

def main():
    print("=" * 60)
    print("法律大模型微调训练 (CPU模式)")
    print("=" * 60)

    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n[1/6] 环境检查...")
    print(f"PyTorch: {torch.__version__}")

    print("\n[2/6] 加载模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_name = config["model"]["path"]
    print(f"模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    model = model.to("cpu")
    model.train()
    print("模型加载完成")

    print("\n[3/6] 应用LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    print("\n[4/6] 加载数据...")
    from datasets import load_dataset

    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
    val_dataset = load_dataset("json", data_files="data/val.jsonl", split="train")
    print(f"训练: {len(train_dataset)} 条, 验证: {len(val_dataset)} 条")

    print("\n[5/6] 预处理...")
    def preprocess(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            prompt = f"请根据以下法律问题提供专业解答：\n\n### 问题：\n{examples['instruction'][i]}\n\n### 解答："
            full = prompt + examples["output"][i] + tokenizer.eos_token
            texts.append(full)
        tokenized = tokenizer(texts, truncation=True, padding=True, max_length=1024)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

    print("\n[6/6] 开始训练...")
    from transformers import TrainingArguments, Trainer

    output_dir = str(project_root / "finetuned")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        logging_steps=2,
        save_steps=20,
        eval_strategy="steps",
        eval_steps=20,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        save_total_limit=2,
        use_cpu=True,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("开始训练...")
    result = trainer.train()

    print("\n保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"训练损失: {result.metrics['train_loss']:.4f}")
    print(f"训练时间: {result.metrics['train_runtime']:.2f} 秒")
    print(f"模型保存到: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
