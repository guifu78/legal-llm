"""
法律大模型微调脚本
使用LoRA/QLoRA进行高效微调
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class LegalModelTrainer:
    """法律模型训练器"""

    def __init__(self, config_path="configs/train_config.yaml"):
        self.config_path = project_root / config_path
        self.config = self.load_config()

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            print(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return {}

    def setup_training_environment(self):
        """设置训练环境"""
        import torch
        print("设置训练环境...")
        if torch.cuda.is_available():
            device = "cuda"
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU: {gpu_props.name}")
            print(f"显存: {gpu_props.total_memory / 1024**3:.2f} GB")
        else:
            device = "cpu"
            print("未检测到CUDA，将使用CPU训练")
        return device

    def train(self):
        """开始训练"""
        print("开始法律大模型微调训练")
        device = self.setup_training_environment()
        print("训练配置加载完成")

def main():
    print("法律大模型微调工具")
    trainer = LegalModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
