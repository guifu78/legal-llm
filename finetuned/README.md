---
library_name: transformers
model_name: legal-llm
tags:
- generated_from_trainer
- trl
- sft
- qlora
- legal
---

# 法律大模型微调 (Legal LLM)

基于 Qwen2.5-7B-Instruct，使用 QLoRA 4-bit 量化微调的中文法律问答模型。

## 训练结果

| 指标 | 数值 |
|------|------|
| 基础模型 | Qwen/Qwen2.5-7B-Instruct |
| 微调方法 | QLoRA (4-bit NF4) |
| LoRA rank | 64 |
| 训练数据 | 214 条法律问答（171 训练 / 43 验证） |
| 训练轮数 | 3 epochs |
| 最终训练损失 | 0.7741 |
| Token 准确率 | 95.1% |
| 训练时间 | ~46 分钟（RTX 5070 Laptop） |

## 数据覆盖

- 民法典（合同、物权、婚姻家庭、继承、侵权责任）
- 刑法（犯罪与刑罚、常见罪名）
- 劳动法（劳动合同、工资、工伤、社保）
- 知识产权（著作权、专利、商标）
- 公司法（治理、股权、人格否认）
- 行政法（处罚、许可、赔偿）
- 诉讼法（管辖、证据、执行）
- 商法（保险、票据、破产）
- 新兴领域（个人信息保护、网络安全、电子合同）

## Quick start

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, "./finetuned/final")
tokenizer = AutoTokenizer.from_pretrained("./finetuned/final")

prompt = "<|user|>\n什么是正当防卫？<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Framework versions

- TRL: 1.6.0
- Transformers: 4.57.1
- Pytorch: 2.11.0+cu128

## 许可证

本项目代码采用 MIT 许可证。微调模型使用需遵守 Qwen2.5 的许可协议。
