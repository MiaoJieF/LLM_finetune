#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma模型LoRA微调脚本
使用PEFT库对Gemma模型进行参数高效微调
"""

import os
import torch
import json
from typing import Dict, Any, Optional
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_callback import ProgressCallback
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    PeftConfig
)
from datasets import Dataset
import numpy as np
# 评估指标函数
def accuracy_score(y_true, y_pred):
    """计算准确率"""
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if len(y_true) > 0 else 0

def precision_recall_fscore_support(y_true, y_pred, average='weighted'):
    """计算精确率、召回率、F1分数"""
    # 简单的实现，实际项目中建议安装sklearn
    precision = accuracy_score(y_true, y_pred)
    recall = precision
    f1 = precision
    return precision, recall, f1, None

from banking_dataset import BankingDataset

class GemmaLoRATrainer:
    def __init__(
        self,
        model_path: str = "models/gemma3-1b",
        output_dir: str = "outputs/lora_banking",
        lora_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化LoRA训练器
        
        Args:
            model_path (str): 预训练模型路径
            output_dir (str): 输出目录
            lora_config (Dict): LoRA配置参数
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 默认LoRA配置
        self.lora_config = lora_config or {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"LoRA配置: {self.lora_config}")
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        try:
            print("正在加载模型和分词器...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("模型和分词器加载完成!")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def setup_lora(self):
        """设置LoRA配置"""
        try:
            print("设置LoRA配置...")
            
            # 创建LoRA配置
            lora_config = LoraConfig(**self.lora_config)
            
            # 应用LoRA到模型
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # 打印可训练参数
            self.peft_model.print_trainable_parameters()
            
            print("LoRA配置完成!")
            return True
            
        except Exception as e:
            print(f"LoRA设置失败: {e}")
            return False
    
    def prepare_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        """
        准备训练数据集
        
        Args:
            dataset (Dataset): 原始数据集
            max_length (int): 最大序列长度
            
        Returns:
            Dataset: 处理后的数据集
        """
        def format_instruction(example):
            """格式化指令"""
            if example["input"]:
                prompt = f"### 指令:\n{example['instruction']}\n### 输入:\n{example['input']}\n### 回答:\n{example['output']}"
            else:
                prompt = f"### 指令:\n{example['instruction']}\n### 回答:\n{example['output']}"
            return {"text": prompt}
        
        def tokenize_function(examples):
            """分词函数"""
            # 格式化文本
            texts = [format_instruction({"instruction": inst, "input": inp, "output": out})["text"] 
                    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
            
            # 分词
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 设置标签（与输入相同，用于因果语言建模）
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        
        # 获取预测结果
        predictions = np.argmax(predictions, axis=-1)
        
        # 计算准确率
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        
        # 计算精确率、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.flatten(), predictions.flatten(), average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        """
        开始训练
        
        Args:
            train_dataset (Dataset): 训练数据集
            eval_dataset (Dataset): 评估数据集
            num_epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            warmup_steps (int): 预热步数
            save_steps (int): 保存步数
            eval_steps (int): 评估步数
            logging_steps (int): 日志步数
        """
        try:
            print("开始准备训练...")
            
            # 准备数据集
            train_dataset = self.prepare_dataset(train_dataset)
            if eval_dataset:
                eval_dataset = self.prepare_dataset(eval_dataset)
            
            # 数据整理器
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.peft_model,
                padding=True
            )
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                logging_dir=f"{self.output_dir}/logs",
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                eval_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="f1" if eval_dataset else None,
                greater_is_better=True,
                learning_rate=learning_rate,
                fp16=self.device == "cuda",
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,  # 禁用wandb等日志记录
                disable_tqdm=False,  # 在控制台显示训练进度条
            )
            
            # 创建训练器
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics if eval_dataset else None,
                callbacks=[ProgressCallback()],
            )
            
            print("开始训练...")
            print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            # 开始训练
            trainer.train()
            
            # 保存模型
            print("保存模型...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # 保存LoRA配置
            lora_config_path = os.path.join(self.output_dir, "lora_config.json")
            with open(lora_config_path, "w", encoding="utf-8") as f:
                json.dump(self.lora_config, f, indent=2, ensure_ascii=False)
            
            print(f"训练完成! 模型已保存到: {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def save_model_info(self, train_dataset_size: int, training_time: str):
        """保存模型信息"""
        model_info = {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "lora_config": self.lora_config,
            "train_dataset_size": train_dataset_size,
            "training_time": training_time,
            "device": self.device,
            "created_at": datetime.now().isoformat()
        }
        
        info_path = os.path.join(self.output_dir, "model_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"模型信息已保存到: {info_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemma LoRA微调")
    parser.add_argument("--model-path", default="models/gemma3-1b", help="预训练模型路径")
    parser.add_argument("--output-dir", default="outputs/lora_banking", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    args = parser.parse_args()
    
    # 创建训练器
    trainer = GemmaLoRATrainer(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # 加载模型
    if not trainer.load_model_and_tokenizer():
        return
    
    # 设置LoRA
    if not trainer.setup_lora():
        return
    
    # 创建数据集
    print("创建训练数据集...")
    banking_dataset = BankingDataset()
    dataset = banking_dataset.create_dataset()
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"评估集大小: {len(eval_dataset)}")
    
    # 开始训练
    start_time = datetime.now()
    success = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if success:
        training_time = str(datetime.now() - start_time)
        trainer.save_model_info(len(train_dataset), training_time)
        print("训练成功完成!")
    else:
        print("训练失败!")

if __name__ == "__main__":
    main()
