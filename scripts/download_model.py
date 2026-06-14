"""
下载基础模型脚本
使用 Hugging Face Hub 下载预训练模型
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_model(model_name: str, save_path: str):
    """
    下载指定的预训练模型
    
    参数:
        model_name: 模型名称（Hugging Face Hub上的名称）
        save_path: 保存路径
    """
    try:
        from huggingface_hub import snapshot_download
        from tqdm import tqdm
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"🚀 开始下载模型: {model_name}")
        print(f"📁 保存路径: {save_path}")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 检查CUDA是否可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  使用设备: {device}")
        
        if device == "cuda":
            # 检查显存
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"🎮 GPU: {gpu_props.name}")
            print(f"💾 显存: {gpu_props.total_memory / 1024**3:.2f} GB")
        
        # 下载tokenizer
        print("\n📥 下载Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=save_path,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        print("✅ Tokenizer下载完成")
        
        # 下载模型
        print("\n📥 下载模型（这可能需要一些时间）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=save_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 保存模型
        model.save_pretrained(os.path.join(save_path, "model"))
        print("✅ 模型下载完成")
        
        # 打印模型信息
        print("\n📊 模型信息:")
        print(f"  - 参数量: {model.num_parameters() / 1e9:.2f}B")
        print(f"  - 模型类型: {model.__class__.__name__}")
        print(f"  - 词汇表大小: {len(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False

def main():
    """主函数"""
    # 推荐的模型列表
    models = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "glm-4-9b": "THUDM/glm-4-9b-chat",
        "baichuan2-13b": "baichuan-inc/Baichuan2-13B-Chat"
    }
    
    print("🏛️ 法律大模型 - 模型下载工具")
    print("=" * 50)
    print("\n推荐模型:")
    for i, (key, name) in enumerate(models.items(), 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 50)
    choice = input("请选择要下载的模型 (1-3): ").strip()
    
    if choice not in ["1", "2", "3"]:
        print("❌ 无效的选择")
        return
    
    model_name = list(models.values())[int(choice) - 1]
    save_path = project_root / "models"
    
    success = download_model(model_name, str(save_path))
    
    if success:
        print("\n🎉 模型下载完成！")
        print("下一步：运行 python scripts/prepare_data.py 准备训练数据")
    else:
        print("\n💡 提示：如果下载失败，可以尝试以下方法：")
        print("  1. 使用镜像站点")
        print("  2. 手动下载模型文件")
        print("  3. 检查网络连接")

if __name__ == "__main__":
    main()
