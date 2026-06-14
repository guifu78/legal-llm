# -*- coding: utf-8 -*-
"""
法律大模型简化训练脚本 - 修复版
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path

project_root = Path(".")

def main():
    print("=" * 60)
    print("法律大模型微调训练")
    print("=" * 60)
    
    # 加载配置
    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 环境检查
    print("\n[1/6] 检查环境...")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到CUDA")
    
    # 加载模型
    print("\n[2/6] 加载模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    model_name = config["model"]["path"]
    print(f"模型路径: {model_name}")
    
    # 量化配置 - 使用fp16而不是4bit以避免兼容性问题
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer加载完成")
    
    # 加载模型 - 使用trust_remote_code
    print("正在加载模型（可能需要1-2分钟）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("模型加载完成")
    
    # 应用LoRA
    print("\n[3/6] 应用LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = config.get("lora", {})
    peft_config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
    
    # 加载数据
    print("\n[4/6] 加载训练数据...")
    from datasets import load_dataset
    
    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
    val_dataset = load_dataset("json", data_files="data/val.jsonl", split="train")
    print(f"训练数据: {len(train_dataset)} 条")
    print(f"验证数据: {len(val_dataset)} 条")
    
    # 预处理
    print("\n[5/6] 预处理数据...")
    def preprocess(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            prompt = f"请根据以下法律问题提供专业解答：\n\n### 问题：\n{examples['instruction'][i]}\n\n### 解答："
            full = prompt + examples["output"][i] + tokenizer.eos_token
            texts.append(full)
        
        tokenized = tokenizer(texts, truncation=True, padding=True, max_length=2048)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)
    print("数据预处理完成")
    
    # 训练
    print("\n[6/6] 开始训练...")
    from transformers import TrainingArguments, Trainer
    
    training_config = config.get("training", {})
    output_dir = str(project_root / training_config.get("output_dir", "finetuned"))
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        report_to="tensorboard",
        logging_dir=str(project_root / "logs"),
        dataloader_num_workers=0,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    print("开始训练...")
    result = trainer.train()
    
    # 保存模型
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
