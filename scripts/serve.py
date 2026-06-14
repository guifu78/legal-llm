"""
法律大模型服务启动脚本
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = FastAPI(title="法律大模型API", version="1.0.0")

class LegalQuery(BaseModel):
    """法律查询请求模型"""
    query: str
    context: str = ""

class LegalResponse(BaseModel):
    """法律查询响应模型"""
    answer: str
    confidence: float
    sources: list

@app.get("/")
async def root():
    """根路径"""
    return {"message": "法律大模型API服务正在运行"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model": "Legal-LLM"}

@app.post("/predict", response_model=LegalResponse)
async def predict(query: LegalQuery):
    """法律问题预测接口"""
    # 这里将加载微调后的模型进行预测
    return LegalResponse(
        answer=f"关于您的法律问题：{query.query}，这是一个示例回答。",
        confidence=0.95,
        sources=["中华人民共和国民法典", "中华人民共和国刑法"]
    )

def main():
    """主函数"""
    print("启动法律大模型API服务...")
    print("API文档地址: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
