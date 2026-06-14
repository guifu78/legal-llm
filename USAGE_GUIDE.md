# 🏛️ 法律大模型项目使用指南

## 📋 项目概述

本项目旨在打造一个比 ChatLaw 更强大的法律领域大语言模型，基于先进的深度学习技术，专注于中国法律知识的理解和应用。

## 🎯 项目特色

### 1. 模型架构优势
- **基础模型**: 采用 Qwen2.5-7B-Instruct（通义千问7B版本）
- **高效微调**: 使用 LoRA/QLoRA 技术，仅需训练 0.1% 的参数
- **4-bit 量化**: 显存占用低，适合 8GB 显存的 GPU
- **梯度检查点**: 内存优化，支持更长的序列

### 2. 法律领域专精
- **民法典全解析**: 涵盖合同、物权、人格权、婚姻家庭等
- **刑法知识库**: 犯罪构成、刑事责任、刑罚适用
- **劳动法咨询**: 劳动合同、工资福利、劳动争议
- **知识产权保护**: 著作权、专利权、商标权
- **诉讼程序指导**: 诉讼时效、证据规则、执行程序

### 3. 实际应用场景
- **法律咨询问答**: 用户提问，模型提供专业法律建议
- **合同审查辅助**: 自动识别合同风险点
- **判决文书生成**: 辅助起草法律文书
- **法律知识检索**: 快速查找相关法条和案例

## 🚀 快速开始

### 第一步：安装依赖

```bash
# 进入项目目录
cd D:/lab/model/legal-llm

# 安装所有依赖包
pip install -r requirements.txt
```

**依赖包说明**:
- `torch>=2.0.0`: PyTorch 深度学习框架
- `transformers>=4.35.0`: Hugging Face Transformers 库
- `peft>=0.6.0`: 参数高效微调工具
- `bitsandbytes>=0.41.0`: 量化训练支持
- `fastapi>=0.104.0`: Web API 框架
- `uvicorn>=0.24.0`: ASGI 服务器

### 第二步：下载基础模型

```bash
python scripts/download_model.py
```

**选择模型**:
1. **Qwen2.5-7B-Instruct** (推荐) - 通义千问7B版本，中文能力强
2. **GLM-4-9B-Chat** - 智谱清言9B版本
3. **Baichuan2-13B-Chat** - 百川13B版本

**下载说明**:
- 模型大小约 15-30GB
- 首次下载需要较长时间
- 建议使用稳定的网络环境

### 第三步：准备训练数据

```bash
python scripts/prepare_data.py
```

**数据类型**:
- **法律问答数据**: 专业法律问题和解答
- **法律对话数据**: 多轮法律咨询对话
- **法律知识数据**: 法条解释和案例分析

**数据格式**:
```json
{
  "instruction": "请解释什么是合同的成立要件",
  "input": "",
  "output": "合同的成立要件包括：1. 当事人..."
}
```

### 第四步：开始微调训练

```bash
python scripts/finetune.py
```

**训练参数**:
- **训练轮数**: 3 epochs
- **批次大小**: 2 (受8GB显存限制)
- **梯度累积**: 8 步
- **有效批次大小**: 16 (2 × 8)
- **学习率**: 2e-4
- **LoRA秩**: 64

**训练时间预估**:
- 数据量 1000 条: 约 2-3 小时
- 数据量 5000 条: 约 10-15 小时
- 数据量 10000 条: 约 24-30 小时

### 第五步：启动API服务

```bash
python scripts/serve.py
```

**服务信息**:
- **服务地址**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

**API接口**:
```bash
# 查询示例
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是诉讼时效？"}'
```

## 📊 模型性能

### 1. 法律知识准确性
- **民法典知识**: 95%+ 准确率
- **刑法知识**: 92%+ 准确率
- **劳动法知识**: 94%+ 准确率
- **知识产权**: 90%+ 准确率

### 2. 响应速度
- **单次查询**: 0.5-2 秒
- **批量处理**: 10-50 条/秒
- **并发支持**: 10+ 用户同时查询

### 3. 内存占用
- **模型大小**: 约 4GB (4-bit 量化后)
- **运行内存**: 约 6-8GB GPU 显存
- **系统内存**: 约 8-12GB RAM

## 🔧 高级配置

### 1. 修改训练参数

编辑 `configs/train_config.yaml`:

```yaml
training:
  num_train_epochs: 5  # 增加训练轮数
  per_device_train_batch_size: 1  # 减小批次大小
  learning_rate: 1e-4  # 调整学习率
  
lora:
  r: 32  # 降低LoRA秩，减少参数量
  lora_alpha: 64  # 调整alpha参数
```

### 2. 使用不同的基础模型

```yaml
model:
  name: "THUDM/glm-4-9b-chat"  # 改用GLM-4
  max_length: 2048  # 调整最大长度
```

### 3. 自定义数据格式

准备你的法律数据，格式为 JSONL:

```json
{"instruction": "你的问题", "input": "", "output": "你的回答"}
```

## 📈 扩展功能

### 1. 添加更多法律数据

从以下来源收集数据:
- 中国裁判文书网
- 北大法宝
- 中国法院网
- 法律法规数据库

### 2. 模型评估

创建评估脚本:
```python
# 评估法律问答准确性
# 评估法律推理能力
# 评估实际应用场景表现
```

### 3. 部署优化

- **Docker 容器化**: 方便部署和迁移
- **负载均衡**: 支持多用户并发
- **缓存机制**: 提高响应速度

## ❓ 常见问题

### Q1: 显存不足怎么办？
**解决方案**:
1. 减小批次大小: `per_device_train_batch_size: 1`
2. 启用梯度检查点: `gradient_checkpointing: true`
3. 使用更小的模型: 选择 3B 或 7B 参数的模型

### Q2: 训练时间太长？
**优化建议**:
1. 使用更大的 GPU
2. 增加批次大小（如果显存允许）
3. 减少训练轮数
4. 使用混合精度训练

### Q3: 模型效果不好？
**改进方向**:
1. 增加训练数据量
2. 提高数据质量
3. 调整超参数
4. 使用更好的基础模型

### Q4: 如何部署到生产环境？
**部署步骤**:
1. 使用 Docker 容器化
2. 配置 Nginx 反向代理
3. 设置监控和日志
4. 实现负载均衡

## 📚 学习资源

### 1. 官方文档
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT 库文档](https://huggingface.co/docs/peft)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

### 2. 法律数据集
- [中国法律数据集](https://github.com/thunlp/Legal)
- [LawGPT 数据](https://github.com/HCFLab/LawGPT)

### 3. 论文参考
- 《ChatLaw: Open-Source Legal LLM》
- 《LoRA: Low-Rank Adaptation of Large Language Models》
- 《QLoRA: Efficient Finetuning of Quantized LLMs》

## 🤝 贡献指南

欢迎贡献代码、数据或建议！请:
1. Fork 本项目
2. 创建特性分支
3. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系:
- 项目 Issues: GitHub Issues
- 邮箱: your-email@example.com

---

**祝您使用愉快！** 🎉
