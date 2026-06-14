from modelscope import snapshot_download
import os

model_name = 'Qwen/Qwen2.5-7B-Instruct'
save_path = './models'

print('Starting download from ModelScope:', model_name)
print('This may take 10-30 minutes...')

# 下载模型
model_dir = snapshot_download(
    model_name,
    cache_dir=save_path
)

print('Download completed!')
print('Model saved to:', model_dir)
