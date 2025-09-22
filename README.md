## 项目概述

本项目提供基于 Gemma3-1B 的本地微调与推理工具链，包含：

- 交互式对话：`chat_with_model.py`
- LoRA 微调：`lora_finetune.py`
- 一键流程：`quick_start.py`（数据构建 → 微调 → 对比）
- 效果对比：`compare_models.py`
- 银行业务示例数据集：`banking_dataset/` 与 `banking_dataset.py`

适用于 Windows 与 GPU/CPU 环境，默认自动选择设备（CUDA 优先）。

## 快速开始

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 准备基础模型

- 将 Gemma3-1B 模型文件放置到 `models/gemma3-1b/` 目录（包含 `config.json`、`tokenizer.json`、`model.safetensors` 等）。

3) 一键运行完整流程（推荐）

```bash
python quick_start.py
```

该脚本会：
- 构建银行领域数据集
- 进行 LoRA 微调（输出保存到 `outputs/lora_banking_时间戳/`）
- 基于内置问题集完成基础模型 vs 微调模型对比，结果保存为 `comparison_results_*.json`

## 目录结构

- `models/gemma3-1b/`：基础模型（已在 `.gitignore` 中忽略）
- `outputs/`：微调产物与检查点（已在 `.gitignore` 中忽略）
- `banking_dataset/`：Arrow 数据与元数据
- `banking_dataset.py`：构建/加载银行领域数据集脚本
- `lora_finetune.py`：LoRA 训练主脚本
- `chat_with_model.py`：交互聊天脚本（支持流式输出）
- `compare_models.py`：基础模型与微调模型效果对比
- `comparison_results_*.json`：对比结果摘要（已在 `.gitignore` 中忽略）

## 微调（LoRA）

直接使用训练脚本：

```bash
python lora_finetune.py \
  --model-path models/gemma3-1b \
  --output-dir outputs/lora_banking_YYYYMMDD_HHMMSS \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 2e-4
```

说明：
- 会自动加载 `banking_dataset.py` 生成的数据集，按 8:2 划分训练/评估。
- 训练完成后会保存 tokenizer、LoRA 适配器、`lora_config.json` 与 `model_info.json`。

## 交互对话

使用基础模型：

```bash
python chat_with_model.py
```

使用微调结果：

```bash
python chat_with_model.py --model-path outputs/lora_banking_YYYYMMDD_HHMMSS
```

常用参数：
- `--no-stream`：关闭流式输出
- `--prompt` 或 `--prompt-file`：自定义 Prompt 模板（包含占位符 `{input}`）

运行时命令：
- 输入 `quit` / `exit` 退出
- 输入 `clear` 清屏

## 模型效果对比

对比基础模型与微调模型：

```bash
python compare_models.py \
  --peft-model-path outputs/lora_banking_YYYYMMDD_HHMMSS \
  --base-model-path models/gemma3-1b \
  --output-file comparison_results_YYYYMMDD_HHMMSS.json
```

对比脚本将基于内置的银行问题集生成回答，并进行启发式打分，输出包含：
- 各问题的回答、时间、关键词匹配与完整度评分
- 汇总统计（平均改进、正/负/无变化数量）

## 数据集

构建/更新银行领域数据集：

```bash
python banking_dataset.py
```

数据将保存在 `banking_dataset/` 目录下（Arrow 格式，包含 `dataset_info.json`、`state.json`）。

## 环境与性能建议

- GPU 建议：>= 8GB 显存；CUDA 环境将自动启用 bfloat16 推理/训练（由脚本内部自动处理）。
- CPU 也可运行，但推理与训练速度会明显下降。
- 磁盘空间：模型与输出较大，请预留足够空间。

## 常见问题

- 无法找到模型：请检查 `models/gemma3-1b/` 是否存在且文件完整。
- 依赖缺失：按提示运行 `pip install -r requirements.txt`。
- GBK 编码问题（Windows 控制台）：`quick_start.py` 已设置 `encoding='utf-8', errors='replace'` 处理常见输出编码冲突。

## 许可证

仅用于学习与研究，使用前请遵守基础模型与依赖库的各自许可证。
