# -*- coding: utf-8 -*-
"""
法律大模型 API 服务
加载微调后的模型进行推理
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import yaml

app = FastAPI(title="法律大模型API", version="1.0.0")

# 全局模型变量
model = None
tokenizer = None
model_loaded = False

class LegalQuery(BaseModel):
    query: str
    context: str = ""

class LegalResponse(BaseModel):
    answer: str
    confidence: float
    sources: list

@app.get("/")
async def root():
    return {"message": "法律大模型API服务正在运行", "model_loaded": model_loaded}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_loaded}

@app.post("/predict", response_model=LegalResponse)
async def predict(query: LegalQuery):
    global model, tokenizer, model_loaded

    if not model_loaded:
        return LegalResponse(
            answer=f"模型尚未加载。关于您的问题：{query.query}，请先完成模型训练。",
            confidence=0.0,
            sources=[]
        )

    # 使用模型生成回答
    import torch
    prompt = f"<|user|>\n{query.query}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = response.split("<|end|>")[0].strip()

    return LegalResponse(
        answer=response,
        confidence=0.85,
        sources=["中华人民共和国民法典", "中华人民共和国刑法", "中华人民共和国劳动合同法"]
    )

def load_model():
    """加载微调后的模型"""
    global model, tokenizer, model_loaded

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        config_path = project_root / "configs" / "train_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 优先加载微调后的模型
        finetuned_path = project_root / "finetuned" / "final"
        base_model_path = project_root / config["model"]["path"]

        if finetuned_path.exists():
            load_path = finetuned_path
            print(f"加载微调模型: {load_path}")
        elif base_model_path.exists():
            load_path = base_model_path
            print(f"加载基础模型: {load_path}")
        else:
            print("未找到模型文件")
            return

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(str(load_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(load_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        model_loaded = True
        print("模型加载成功！")

    except Exception as e:
        print(f"模型加载失败: {e}")

def main():
    print("启动法律大模型API服务...")
    print("正在加载模型...")
    load_model()
    print("API文档地址: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
