#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma3模型对话脚本
支持控制台交互式对话
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

class GemmaChatBot:
    def __init__(self, model_path="models/gemma3-1b", use_stream=True, prompt_template=None, peft_model_path=None, base_model_path="models/gemma3-1b"):
        """
        初始化聊天机器人
        
        Args:
            model_path (str): 模型路径
            use_stream (bool): 是否使用流式输出
            prompt_template (str | None): 将用户输入包装到的提示模板。可包含占位符"{input}"
            peft_model_path (str | None): LoRA/PEFT 适配器路径
            base_model_path (str): 当使用PEFT时的基础模型路径
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_stream = use_stream
        self.peft_model_path = peft_model_path
        self.base_model_path = base_model_path
        # 默认模板（若未提供），包含占位符 {input}
        self.prompt_template = prompt_template or (
            "你是一个专业且简洁的助手。请直接、准确地回答问题，必要时给出步骤。\n\n"
            "[用户问题]\n{input}\n\n[请给出回答]"
        )
        print(f"使用设备: {self.device}")
        print(f"流式输出: {'开启' if use_stream else '关闭'}")
        if self.prompt_template is not None:
            print("已启用自定义Prompt模板")

    def _format_input_with_prompt(self, user_input: str) -> str:
        """将用户输入包装到Prompt模板中"""
        template = self.prompt_template or "{input}"
        # 如果模板包含占位符，优先替换；否则在模板后追加用户输入
        if "{input}" in template:
            try:
                return template.format(input=user_input)
            except Exception:
                # 回退：直接拼接
                return f"{template}\n\n{user_input}"
        else:
            return f"{template}\n\n{user_input}"
        
    def load_model(self):
        """加载模型和分词器（支持基础模型或PEFT适配器）"""
        try:
            # 判断是否为PEFT目录（自动探测）
            is_adapter_dir = os.path.isdir(self.model_path) and os.path.exists(os.path.join(self.model_path, "adapter_config.json"))

            if self.peft_model_path or is_adapter_dir:
                peft_dir = self.peft_model_path or self.model_path
                print("检测到PEFT适配器，开始加载LoRA模型...")
                print(f"适配器路径: {peft_dir}")
                print(f"基础模型路径: {self.base_model_path}")

                # 优先从适配器目录加载分词器（若存在），否则回退到基础模型
                tokenizer_source = peft_dir if os.path.exists(os.path.join(peft_dir, "tokenizer.json")) else self.base_model_path
                print("加载分词器...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # 加载基础模型
                print("加载基础模型...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    base_model = base_model.to(self.device)

                # 应用PEFT适配器
                print("应用PEFT适配器...")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    peft_dir,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                )
                print("PEFT模型加载完成!")
                return True
            else:
                print("正在加载基础模型...")
                print(f"模型路径: {self.model_path}")

                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

                # 加载分词器
                print("加载分词器...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )

                # 加载模型
                print("加载模型...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)

                print("模型加载完成!")
                return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def generate_response_stream(self, user_input, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        流式生成回复

        Args:
            user_input (str): 用户输入
            max_new_tokens (int): 最大新生成token数
            temperature (float): 温度参数
            top_p (float): top_p参数

        Yields:
            str: 流式生成的文本片段
        """
        try:
            # 先将用户输入包裹到指定Prompt模板
            formatted_input = self._format_input_with_prompt(user_input)

            # 使用tokenizer的encode方法获取input_ids和attention_mask
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # 创建流式输出器
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,  # 跳过输入提示
                skip_special_tokens=True
            )

            # 流式生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    streamer=streamer
                )

            # 返回完整回复用于后续处理
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            print(f"生成回复时出错: {e}")
            return ""

    def generate_response(self, user_input, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        生成回复（非流式，用于兼容性）

        Args:
            user_input (str): 用户输入
            max_new_tokens (int): 最大新生成token数
            temperature (float): 温度参数
            top_p (float): top_p参数

        Returns:
            str: 模型回复
        """
        try:
            # 先将用户输入包裹到指定Prompt模板
            formatted_input = self._format_input_with_prompt(user_input)

            # 使用tokenizer的encode方法获取input_ids和attention_mask
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2
                )

            # 解码输出，只保留新生成的部分
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            return f"生成回复时出错: {e}"
    
    def chat(self):
        """开始对话"""
        if not self.load_model():
            return
        
        print("\n" + "="*50)
        print("Gemma3 对话机器人")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清屏")
        print("="*50 + "\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("你: ").strip()
                
                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见!")
                    break
                
                # 检查清屏命令
                if user_input.lower() in ['clear', '清屏']:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                # 检查空输入
                if not user_input:
                    continue
                
                # 生成回复
                if self.use_stream:
                    # 流式生成回复
                    print("机器人: ", end="", flush=True)
                    response = self.generate_response_stream(user_input)
                    print()  # 换行
                else:
                    # 非流式生成回复
                    print("机器人: ", end="", flush=True)
                    response = self.generate_response(user_input)
                    print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Gemma3 对话机器人")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument("--model-path", default="models/gemma3-1b", help="模型路径（可指向基础模型或PEFT适配器）")
    parser.add_argument("--peft-model-path", default=None, help="LoRA/PEFT 适配器路径（可选，若--model-path即为适配器目录可不填）")
    parser.add_argument("--base-model-path", default="models/gemma3-1b", help="当使用PEFT时的基础模型路径")
    parser.add_argument("--prompt", default=None, help="直接传入的Prompt模板，使用{input}占位符")
    parser.add_argument("--prompt-file", default=None, help="从文件读取Prompt模板（优先级高于 --prompt）")
    args = parser.parse_args()
    
    # 解析模板
    prompt_template = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except Exception as e:
            print(f"读取Prompt模板文件失败: {e}")
            prompt_template = args.prompt
    else:
        prompt_template = args.prompt

    # 创建聊天机器人实例
    chatbot = GemmaChatBot(
        model_path=args.model_path,
        use_stream=not args.no_stream,
        prompt_template=prompt_template,
        peft_model_path=args.peft_model_path,
        base_model_path=args.base_model_path
    )
    
    # 开始对话
    chatbot.chat()

if __name__ == "__main__":
    main()
