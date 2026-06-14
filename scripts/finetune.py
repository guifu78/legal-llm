# -*- coding: utf-8 -*-
"""
法律大模型微调脚本
使用 QLoRA 进行高效微调
"""
import os
import json
import yaml
import glob
from pathlib import Path

project_root = Path(__file__).parent.parent

def load_config():
    config_path = project_root / "configs" / "train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_data(data_file):
    """加载 JSONL 数据"""
    data = []
    with open(project_root / data_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_instruction(sample):
    """将 instruction/input/output 格式化为对话格式"""
    if sample.get("input") and sample["input"].strip():
        text = f"<|user|>\n{sample['instruction']}\n{sample['input']}<|end|>\n<|assistant|>\n{sample['output']}<|end|>"
    else:
        text = f"<|user|>\n{sample['instruction']}<|end|>\n<|assistant|>\n{sample['output']}<|end|>"
    return text

def main():
    print("=" * 60)
    print("法律大模型 QLoRA 微调训练")
    print("=" * 60)

    # 加载配置
    config = load_config()
    print(f"\n基础模型: {config['model']['name']}")
    print(f"本地模型路径: {config['model']['path']}")

    # 检查模型是否存在
    model_path = project_root / config['model']['path']
    if not model_path.exists():
        print(f"\n错误: 模型路径不存在: {model_path}")
        print("请先运行 python scripts/download_model.py 下载模型")
        return

    # 加载库
    print("\n加载依赖库...")
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # GPU 检查
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("警告: 未检测到 CUDA，训练将非常缓慢！")

    # 量化配置
    print("\n配置 4-bit 量化...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 加载 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("加载模型（4-bit 量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    print("配置 LoRA...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type="CAUSAL_LM",
    )

    # 加载和处理数据
    print("\n加载训练数据...")
    train_data = load_data("data/train.jsonl")
    val_data = load_data("data/val.jsonl")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    # 格式化数据
    train_texts = [format_instruction(s) for s in train_data]
    val_texts = [format_instruction(s) for s in val_data]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    # 训练参数
    print("\n配置训练参数...")
    output_dir = project_root / config["training"]["output_dir"]

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        optim=config["training"]["optim"],
        max_grad_norm=config["training"]["max_grad_norm"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=config["model"]["max_length"],
    )

    # 查找最新 checkpoint 用于恢复训练
    resume_from_checkpoint = None
    checkpoint_dir = output_dir
    checkpoints = sorted(glob.glob(str(checkpoint_dir / "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
    if checkpoints:
        resume_from_checkpoint = checkpoints[-1]
        print(f"\n找到最新 checkpoint: {resume_from_checkpoint}")

    # 创建 Trainer
    print("\n开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 开始训练（自动恢复）
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型
    print("\n保存模型...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {final_dir}")
    print(f"训练损失: {result.training_loss:.4f}")
    print(f"训练步数: {result.global_step}")
    print("=" * 60)

if __name__ == "__main__":
    main()
