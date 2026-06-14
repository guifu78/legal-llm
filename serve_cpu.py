# -*- coding: utf-8 -*-
"""法律大模型API服务 (CPU模式) - 带中文Web界面"""

import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import threading

app = FastAPI(title="法律大模型 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
tokenizer = None
model_loaded = False

class LegalQuery(BaseModel):
    query: str
    max_length: Optional[int] = 256
    temperature: Optional[float] = 0.7

class LegalResponse(BaseModel):
    answer: str
    status: str = "success"

def load_model():
    global model, tokenizer, model_loaded
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_name = "./models/Qwen/Qwen2___5-7B-Instruct"
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        print("Using GPU:", torch.cuda.get_device_name(0))
        print("Using 4-bit quantization to fit 8GB VRAM")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
    else:
        print("Using CPU (no GPU detected)")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=None, trust_remote_code=True)
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "./finetuned")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    model_loaded = True
    print("Model loaded!")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_model, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "法律大模型 API", "model_loaded": model_loaded}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loaded}

@app.post("/predict", response_model=LegalResponse)
async def predict(query: LegalQuery):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model loading...")
    try:
        prompt = f"请根据以下法律问题提供专业解答：\n\n### 问题：\n{query.query}\n\n### 解答："
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=query.max_length,
                temperature=query.temperature,
                do_sample=True,
                top_p=0.9
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### 解答：" in response:
            answer = response.split("### 解答：")[-1].strip()
        else:
            answer = response[-query.max_length:]
        return LegalResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print("=" * 50)
    print("  法律大模型 API 服务")
    print("  Web界面: http://localhost:8080")
    print("  API文档: http://localhost:8080/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8080)
