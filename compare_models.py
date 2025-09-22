#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比测试脚本
比较微调前后模型在银行领域问题上的回答效果
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime

from model_manager import ModelManager
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

class ModelComparator:
    def __init__(self, base_model_path: str = "models/gemma3-1b"):
        """
        初始化模型对比器
        
        Args:
            base_model_path (str): 基础模型路径
        """
        self.base_model_path = base_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        
        print(f"使用设备: {self.device}")
    
    def load_base_model(self):
        """加载基础模型"""
        try:
            print("正在加载基础模型...")
            
            # 加载分词器
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            # 加载模型
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.base_model = self.base_model.to(self.device)
            
            print("基础模型加载完成!")
            return True
            
        except Exception as e:
            print(f"基础模型加载失败: {e}")
            return False
    
    def load_finetuned_model(self, peft_model_path: str):
        """加载微调后的模型"""
        try:
            print(f"正在加载微调模型: {peft_model_path}")
            
            manager = ModelManager(self.base_model_path)
            if manager.load_peft_model(peft_model_path):
                self.finetuned_model = manager
                print("微调模型加载完成!")
                return True
            else:
                print("微调模型加载失败!")
                return False
                
        except Exception as e:
            print(f"微调模型加载失败: {e}")
            return False
    
    def create_test_questions(self) -> List[Dict[str, str]]:
        """创建银行领域测试问题"""
        test_questions = [
            {
                "question": "如何开设银行账户？",
                "category": "账户管理",
                "expected_keywords": ["身份证", "收入证明", "银行网点", "开户申请"]
            },
            {
                "question": "银行卡丢失了怎么办？",
                "category": "账户安全",
                "expected_keywords": ["挂失", "客服电话", "身份证", "补办"]
            },
            {
                "question": "个人住房贷款需要什么条件？",
                "category": "贷款业务",
                "expected_keywords": ["收入证明", "征信记录", "首付", "购房合同"]
            },
            {
                "question": "如何选择理财产品？",
                "category": "投资理财",
                "expected_keywords": ["风险承受", "投资目标", "流动性", "分散投资"]
            },
            {
                "question": "信用卡逾期会有什么后果？",
                "category": "信用卡",
                "expected_keywords": ["征信记录", "罚息", "催收", "信用评级"]
            },
            {
                "question": "如何保护网银安全？",
                "category": "网络安全",
                "expected_keywords": ["强密码", "安全环境", "短信验证", "异常监控"]
            },
            {
                "question": "外汇业务如何办理？",
                "category": "外汇业务",
                "expected_keywords": ["身份证件", "用途证明", "外汇账户", "汇率"]
            },
            {
                "question": "银行保险产品有哪些类型？",
                "category": "保险业务",
                "expected_keywords": ["储蓄型", "投资型", "保障型", "年金保险"]
            }
        ]
        
        return test_questions
    
    def generate_base_response(self, question: str, max_new_tokens: int = 256) -> Tuple[str, float]:
        """生成基础模型回复"""
        try:
            start_time = time.time()
            
            # 格式化输入
            prompt = f"### 指令:\n{question}\n### 回答:\n"
            
            # 编码输入
            inputs = self.base_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.base_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.base_tokenizer.pad_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.base_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            return response.strip(), generation_time
            
        except Exception as e:
            print(f"基础模型生成失败: {e}")
            return f"生成失败: {e}", 0.0
    
    def generate_finetuned_response(self, question: str, max_new_tokens: int = 256) -> Tuple[str, float]:
        """生成微调模型回复"""
        try:
            start_time = time.time()
            
            response = self.finetuned_model.generate_response(
                question, 
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            generation_time = time.time() - start_time
            
            return response, generation_time
            
        except Exception as e:
            print(f"微调模型生成失败: {e}")
            return f"生成失败: {e}", 0.0
    
    def evaluate_response_quality(self, response: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """评估回复质量"""
        response_lower = response.lower()
        
        # 计算关键词匹配度
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
        keyword_match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
        
        # 计算回复长度
        response_length = len(response)
        
        # 计算句子数量
        sentence_count = response.count('。') + response.count('！') + response.count('？')
        
        # 评估回复完整性（简单启发式）
        completeness_score = min(1.0, response_length / 200)  # 假设200字符为完整回复
        
        return {
            "matched_keywords": matched_keywords,
            "keyword_match_rate": keyword_match_rate,
            "response_length": response_length,
            "sentence_count": sentence_count,
            "completeness_score": completeness_score,
            "overall_score": (keyword_match_rate * 0.4 + completeness_score * 0.6)
        }
    
    def run_comparison(self, peft_model_path: str, output_file: str = None) -> Dict[str, Any]:
        """
        运行模型对比测试
        
        Args:
            peft_model_path (str): 微调模型路径
            output_file (str): 输出文件路径
            
        Returns:
            Dict: 对比结果
        """
        print("开始模型对比测试...")
        
        # 加载模型
        if not self.load_base_model():
            print("基础模型加载失败!")
            return {}
        
        if not self.load_finetuned_model(peft_model_path):
            print("微调模型加载失败!")
            return {}
        
        # 获取测试问题
        test_questions = self.create_test_questions()
        
        results = {
            "test_time": datetime.now().isoformat(),
            "base_model_path": self.base_model_path,
            "finetuned_model_path": peft_model_path,
            "device": self.device,
            "test_questions": len(test_questions),
            "comparison_results": []
        }
        
        print(f"开始测试 {len(test_questions)} 个问题...")
        
        for i, test_case in enumerate(test_questions):
            print(f"\n测试问题 {i+1}/{len(test_questions)}: {test_case['question']}")
            
            # 生成基础模型回复
            print("生成基础模型回复...")
            base_response, base_time = self.generate_base_response(test_case['question'])
            
            # 生成微调模型回复
            print("生成微调模型回复...")
            finetuned_response, finetuned_time = self.generate_finetuned_response(test_case['question'])
            
            # 评估回复质量
            base_evaluation = self.evaluate_response_quality(base_response, test_case['expected_keywords'])
            finetuned_evaluation = self.evaluate_response_quality(finetuned_response, test_case['expected_keywords'])
            
            # 保存结果
            comparison_result = {
                "question": test_case['question'],
                "category": test_case['category'],
                "expected_keywords": test_case['expected_keywords'],
                "base_model": {
                    "response": base_response,
                    "generation_time": base_time,
                    "evaluation": base_evaluation
                },
                "finetuned_model": {
                    "response": finetuned_response,
                    "generation_time": finetuned_time,
                    "evaluation": finetuned_evaluation
                },
                "improvement": {
                    "keyword_match_improvement": finetuned_evaluation['keyword_match_rate'] - base_evaluation['keyword_match_rate'],
                    "completeness_improvement": finetuned_evaluation['completeness_score'] - base_evaluation['completeness_score'],
                    "overall_improvement": finetuned_evaluation['overall_score'] - base_evaluation['overall_score'],
                    "time_difference": finetuned_time - base_time
                }
            }
            
            results["comparison_results"].append(comparison_result)
            
            # 打印结果摘要
            print(f"基础模型评分: {base_evaluation['overall_score']:.3f}")
            print(f"微调模型评分: {finetuned_evaluation['overall_score']:.3f}")
            print(f"改进程度: {comparison_result['improvement']['overall_improvement']:.3f}")
        
        # 计算总体统计
        total_improvements = [r['improvement']['overall_improvement'] for r in results['comparison_results']]
        avg_improvement = sum(total_improvements) / len(total_improvements) if total_improvements else 0
        
        results["summary"] = {
            "average_improvement": avg_improvement,
            "positive_improvements": len([i for i in total_improvements if i > 0]),
            "negative_improvements": len([i for i in total_improvements if i < 0]),
            "neutral_improvements": len([i for i in total_improvements if i == 0])
        }
        
        # 保存结果
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n对比结果已保存到: {output_file}")
        
        # 打印总结
        print(f"\n=== 对比测试总结 ===")
        print(f"测试问题数量: {len(test_questions)}")
        print(f"平均改进程度: {avg_improvement:.3f}")
        print(f"正向改进: {results['summary']['positive_improvements']}")
        print(f"负向改进: {results['summary']['negative_improvements']}")
        print(f"无变化: {results['summary']['neutral_improvements']}")
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型对比测试")
    parser.add_argument("--peft-model-path", required=True, help="微调模型路径")
    parser.add_argument("--base-model-path", default="models/gemma3-1b", help="基础模型路径")
    parser.add_argument("--output-file", default="comparison_results.json", help="输出文件路径")
    args = parser.parse_args()
    
    # 创建对比器
    comparator = ModelComparator(args.base_model_path)
    
    # 运行对比测试
    results = comparator.run_comparison(
        peft_model_path=args.peft_model_path,
        output_file=args.output_file
    )
    
    if results:
        print("对比测试完成!")
    else:
        print("对比测试失败!")

if __name__ == "__main__":
    main()
