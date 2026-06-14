# 🏛️ 法律大模型项目 (Legal-LLM)

## 项目目标
打造一个比 ChatLaw 更强大的法律领域大语言模型

## 硬件环境
- GPU: NVIDIA RTX 5070 (8GB 显存)
- CPU: AMD Ryzen 7 7745HX
- 磁盘: 82GB 可用空间

## 技术方案

### 1. 基础模型选择
推荐使用以下模型之一：
- **Qwen2.5-7B-Instruct** - 通义千问7B版本（推荐）
- **GLM-4-9B-Chat** - 智谱清言9B
- **Baichuan2-13B-Chat** - 百川13B

### 2. 训练策略（适合8GB显存）
- 使用 **LoRA** 或 **QLoRA** 进行高效微调
- 采用 **4-bit 量化** 减少显存占用
- 使用 **DeepSpeed ZeRO-2** 优化

### 3. 数据准备
- 法律问答数据集
- 法律条文解释数据
- 判决文书摘要数据
- 法律咨询对话数据

### 4. 模型评估
- 法律知识准确性
- 法律推理能力
- 实际应用场景测试

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载基础模型
python scripts/download_model.py

# 准备数据
python scripts/prepare_data.py

# 开始微调
python scripts/finetune.py

# 启动服务
python scripts/serve.py
```

## 项目结构
```
legal-llm/
├── README.md              # 项目说明
├── requirements.txt       # 依赖列表
├── configs/              # 配置文件
├── data/                 # 训练数据
├── scripts/              # 脚本文件
├── models/               # 基础模型
├── finetuned/            # 微调后的模型
└── logs/                 # 训练日志
```

## 预期效果
- 法律知识问答准确率 > 90%
- 支持多种法律场景
- 响应速度快，适合实际应用
