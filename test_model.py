#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型加载和基本功能
"""

import os
import sys
from chat_with_model import GemmaChatBot

def test_model_loading():
    """测试模型加载"""
    print("测试模型加载...")
    
    # 检查模型路径
    model_path = "models/gemma3-1b"
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查必要的文件
    required_files = ["config.json", "tokenizer.json", "model.safetensors"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            print(f"错误: 缺少必要文件: {file}")
            return False
    
    print("模型文件检查通过")
    return True

def test_imports():
    """测试依赖导入"""
    print("测试依赖导入...")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"PyTorch导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
    except ImportError as e:
        print(f"Transformers导入失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("="*50)
    print("Gemma3模型测试")
    print("="*50)
    
    # 测试依赖
    if not test_imports():
        print("依赖测试失败，请安装必要的包")
        print("运行: pip install -r requirements.txt")
        return
    
    # 测试模型文件
    if not test_model_loading():
        print("模型文件检查失败")
        return
    
    print("\n所有测试通过!")
    print("可以运行 python chat_with_model.py 开始对话")

if __name__ == "__main__":
    main()
