#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
负责保存、加载和管理微调后的模型
"""

import os
import json
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings("ignore")

class ModelManager:
    def __init__(self, base_model_path: str = "models/gemma3-1b"):
        """
        初始化模型管理器
        
        Args:
            base_model_path (str): 基础模型路径
        """
        self.base_model_path = base_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        print(f"使用设备: {self.device}")
    
    def load_base_model(self):
        """加载基础模型"""
        try:
            print("正在加载基础模型...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("基础模型加载完成!")
            return True
            
        except Exception as e:
            print(f"基础模型加载失败: {e}")
            return False
    
    def load_peft_model(self, peft_model_path: str):
        """
        加载PEFT模型
        
        Args:
            peft_model_path (str): PEFT模型路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print(f"正在加载PEFT模型: {peft_model_path}")
            
            # 首先加载基础模型
            if not self.load_base_model():
                return False
            
            # 加载PEFT配置
            peft_config = PeftConfig.from_pretrained(peft_model_path)
            
            # 加载PEFT模型
            self.peft_model = PeftModel.from_pretrained(
                self.model,
                peft_model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            
            print("PEFT模型加载完成!")
            return True
            
        except Exception as e:
            print(f"PEFT模型加载失败: {e}")
            return False
    
    def save_peft_model(self, output_path: str, model_info: Dict[str, Any] = None):
        """
        保存PEFT模型
        
        Args:
            output_path (str): 保存路径
            model_info (Dict): 模型信息
        """
        try:
            if self.peft_model is None:
                print("没有可保存的PEFT模型!")
                return False
            
            print(f"正在保存PEFT模型到: {output_path}")
            
            # 创建输出目录
            os.makedirs(output_path, exist_ok=True)
            
            # 保存PEFT模型
            self.peft_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # 保存模型信息
            if model_info:
                info_path = os.path.join(output_path, "model_info.json")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print("PEFT模型保存完成!")
            return True
            
        except Exception as e:
            print(f"PEFT模型保存失败: {e}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 256, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成回复
        
        Args:
            prompt (str): 输入提示
            max_new_tokens (int): 最大新生成token数
            temperature (float): 温度参数
            top_p (float): top_p参数
            do_sample (bool): 是否采样
            
        Returns:
            str: 生成的回复
        """
        try:
            if self.peft_model is None:
                print("PEFT模型未加载!")
                return ""
            
            # 格式化输入
            formatted_prompt = f"### 指令:\n{prompt}\n### 回答:\n"
            
            # 编码输入
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"生成回复失败: {e}")
            return ""
    
    def generate_response_stream(
        self, 
        prompt: str, 
        max_new_tokens: int = 256, 
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        流式生成回复
        
        Args:
            prompt (str): 输入提示
            max_new_tokens (int): 最大新生成token数
            temperature (float): 温度参数
            top_p (float): top_p参数
            
        Yields:
            str: 流式生成的文本片段
        """
        try:
            if self.peft_model is None:
                print("PEFT模型未加载!")
                return
            
            # 格式化输入
            formatted_prompt = f"### 指令:\n{prompt}\n### 回答:\n"
            
            # 编码输入
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 流式生成
            with torch.no_grad():
                for output in self.peft_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                ):
                    # 解码新生成的token
                    input_length = inputs.input_ids.shape[1]
                    if output.shape[1] > input_length:
                        new_tokens = output[0][input_length:]
                        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        if new_text.strip():
                            yield new_text
                            
        except Exception as e:
            print(f"流式生成失败: {e}")
            yield f"生成失败: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.peft_model is None:
            return {"status": "模型未加载"}
        
        info = {
            "base_model_path": self.base_model_path,
            "device": self.device,
            "model_type": "PEFT模型",
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "loaded_at": datetime.now().isoformat()
        }
        
        return info
    
    def list_available_models(self, models_dir: str = "outputs") -> List[Dict[str, Any]]:
        """
        列出可用的模型
        
        Args:
            models_dir (str): 模型目录
            
        Returns:
            List[Dict]: 可用模型列表
        """
        available_models = []
        
        if not os.path.exists(models_dir):
            return available_models
        
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                # 检查是否是有效的PEFT模型
                if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                    model_info = {
                        "name": item,
                        "path": model_path,
                        "type": "PEFT模型"
                    }
                    
                    # 尝试读取模型信息
                    info_file = os.path.join(model_path, "model_info.json")
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, "r", encoding="utf-8") as f:
                                additional_info = json.load(f)
                                model_info.update(additional_info)
                        except:
                            pass
                    
                    available_models.append(model_info)
        
        return available_models

def main():
    """测试模型管理器"""
    print("测试模型管理器...")
    
    # 创建模型管理器
    manager = ModelManager()
    
    # 列出可用模型
    models = manager.list_available_models()
    print(f"\n可用模型数量: {len(models)}")
    
    for model in models:
        print(f"- {model['name']}: {model['path']}")
    
    # 测试基础模型加载
    if manager.load_base_model():
        print("基础模型加载成功!")
        
        # 测试生成
        response = manager.generate_response("什么是银行账户？")
        print(f"测试回复: {response[:100]}...")
    else:
        print("基础模型加载失败!")

if __name__ == "__main__":
    main()
