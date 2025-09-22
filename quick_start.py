#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本
一键运行完整的LoRA微调流程
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    try:
        # 使用UTF-8编码处理输出，避免GBK解码错误
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace'  # 遇到无法解码的字符时用替换字符代替
        )
        print("✅ 执行成功!")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        if e.stderr:
            print("错误信息:", e.stderr)
        return False

def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖包...")
    
    required_packages = [
        "torch", "transformers", "peft", "datasets", 
        "numpy", "accelerate", "safetensors"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_model_path():
    """检查模型路径"""
    model_path = "models/gemma3-1b"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请确保Gemma模型已下载到正确位置")
        return False
    
    required_files = ["config.json", "tokenizer.json", "model.safetensors"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"❌ 缺少必要文件: {file}")
            return False
    
    print(f"✅ 模型路径检查通过: {model_path}")
    return True

def main():
    """主函数"""
    print("🚀 Gemma LoRA微调快速开始")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装依赖包")
        return
    
    # 检查模型
    if not check_model_path():
        print("\n❌ 模型检查失败，请确保模型文件完整")
        return
    
    print("\n✅ 环境检查通过，开始执行微调流程...")
    
    # 步骤1: 创建数据集
    print("\n📊 步骤1: 创建银行领域数据集")
    if not run_command("python banking_dataset.py", "创建训练数据集"):
        print("❌ 数据集创建失败")
        return
    
    # 步骤2: 开始LoRA微调
    print("\n🔧 步骤2: 开始LoRA微调")
    output_dir = f"outputs/lora_banking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    finetune_cmd = f"""python lora_finetune.py \
        --model-path models/gemma3-1b \
        --output-dir {output_dir} \
        --epochs 20 \
        --batch-size 4 \
        --learning-rate 2e-4"""
    
    if not run_command(finetune_cmd, "LoRA微调训练"):
        print("❌ LoRA微调失败")
        return
    
    # 步骤3: 运行对比测试
    print("\n🧪 步骤3: 运行模型对比测试")
    comparison_file = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    compare_cmd = f"""python compare_models.py \
        --peft-model-path {output_dir} \
        --output-file {comparison_file}"""
    
    if not run_command(compare_cmd, "模型对比测试"):
        print("❌ 对比测试失败")
        return
    
    # 步骤4: 显示结果摘要
    print("\n📈 步骤4: 显示结果摘要")
    try:
        with open(comparison_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        summary = results.get('summary', {})
        print(f"平均改进程度: {summary.get('average_improvement', 0):.3f}")
        print(f"正向改进: {summary.get('positive_improvements', 0)}")
        print(f"负向改进: {summary.get('negative_improvements', 0)}")
        print(f"无变化: {summary.get('neutral_improvements', 0)}")
        
    except Exception as e:
        print(f"读取结果文件失败: {e}")
    
    # 步骤5: 提供使用建议
    print("\n🎉 微调完成!")
    print("="*60)
    print("接下来你可以:")
    print(f"1. 查看微调模型: {output_dir}")
    print(f"2. 查看对比结果: {comparison_file}")
    print(f"3. 使用微调模型对话:")
    print(f"   python chat_with_model.py --model-path {output_dir}")
    print("4. 查看详细文档: README_LoRA.md")
    
    print("\n✨ 快速开始完成!")

if __name__ == "__main__":
    main()
