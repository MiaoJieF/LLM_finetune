#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行领域训练数据集
包含银行相关的问答对，用于LoRA微调
"""

import json
import os
from datasets import Dataset
from typing import List, Dict, Any

class BankingDataset:
    def __init__(self):
        """初始化银行数据集"""
        self.dataset = None
        self.raw_data = self._create_banking_data()
    
    def _create_banking_data(self) -> List[Dict[str, str]]:
        """创建银行领域训练数据"""
        banking_data = [
            # 账户管理相关
            {
                "instruction": "如何开设银行账户？",
                "input": "",
                "output": "开设银行账户需要以下步骤：1. 准备有效身份证件（身份证、护照等）2. 准备收入证明或工作证明 3. 前往银行网点或在线申请 4. 填写开户申请表 5. 进行身份验证 6. 设置账户密码和网银密码 7. 激活账户。不同银行可能有具体要求，建议提前咨询。"
            },
            {
                "instruction": "银行卡丢失了怎么办？",
                "input": "",
                "output": "银行卡丢失后应立即采取以下措施：1. 立即拨打银行客服电话挂失（24小时服务）2. 通过手机银行或网银进行临时挂失 3. 携带身份证到银行网点办理正式挂失 4. 申请补办新卡 5. 修改相关密码。挂失后原卡立即失效，可避免资金损失。"
            },
            {
                "instruction": "如何查询账户余额？",
                "input": "",
                "output": "查询账户余额有多种方式：1. 手机银行APP：登录后查看账户信息 2. 网银：登录个人网银查看 3. ATM机：插入银行卡查询 4. 银行网点：携带身份证和银行卡到柜台查询 5. 电话银行：拨打客服电话按语音提示操作。推荐使用手机银行，方便快捷且安全。"
            },
            
            # 贷款相关
            {
                "instruction": "个人住房贷款需要什么条件？",
                "input": "",
                "output": "个人住房贷款申请条件包括：1. 年龄：18-65周岁 2. 收入证明：月收入是月供的2倍以上 3. 征信记录：无不良信用记录 4. 首付比例：首套房30%，二套房40%以上 5. 工作稳定：有固定收入来源 6. 购房合同：有效的购房合同 7. 其他材料：身份证、户口本、收入证明等。具体条件因银行而异。"
            },
            {
                "instruction": "贷款利率是如何确定的？",
                "input": "",
                "output": "贷款利率确定因素包括：1. 央行基准利率：银行参考央行公布的基准利率 2. 个人信用状况：征信记录良好可享受优惠利率 3. 收入水平：收入越高，利率可能越低 4. 贷款期限：期限越长，利率通常越高 5. 银行政策：不同银行有不同的利率政策 6. 市场环境：经济环境影响利率水平。建议多比较几家银行的利率。"
            },
            {
                "instruction": "提前还款划算吗？",
                "input": "",
                "output": "提前还款是否划算需要考虑：1. 剩余本金：本金越多，提前还款越划算 2. 利率水平：当前利率较高时提前还款更划算 3. 投资收益率：如果投资收益率高于贷款利率，不建议提前还款 4. 违约金：部分银行收取提前还款违约金 5. 资金需求：考虑是否有其他资金需求 6. 心理因素：无债一身轻的心理满足。建议根据个人情况综合考虑。"
            },
            
            # 投资理财相关
            {
                "instruction": "银行理财产品有哪些类型？",
                "input": "",
                "output": "银行理财产品主要类型包括：1. 保本型：本金有保障，收益相对较低 2. 非保本型：收益较高但存在亏损风险 3. 固定收益型：收益相对稳定 4. 浮动收益型：收益随市场波动 5. 短期型：投资期限较短，流动性好 6. 长期型：投资期限较长，收益相对较高。选择时需根据风险承受能力和投资目标决定。"
            },
            {
                "instruction": "如何选择适合自己的理财产品？",
                "input": "",
                "output": "选择理财产品需要考虑：1. 风险承受能力：评估自己能承受的最大损失 2. 投资目标：明确投资目的和期望收益 3. 投资期限：根据资金使用时间选择产品 4. 流动性需求：考虑是否需要随时取用资金 5. 产品了解：充分了解产品特点和风险 6. 分散投资：不要把所有资金投入单一产品 7. 专业咨询：必要时咨询理财师。建议从低风险产品开始。"
            },
            
            # 信用卡相关
            {
                "instruction": "信用卡逾期会有什么后果？",
                "input": "",
                "output": "信用卡逾期后果包括：1. 征信记录：产生不良信用记录，影响未来贷款 2. 罚息费用：产生高额罚息和滞纳金 3. 催收电话：银行会进行电话催收 4. 账户冻结：严重逾期可能导致账户被冻结 5. 法律风险：长期逾期可能面临法律诉讼 6. 信用评级：影响个人信用评级。建议按时还款，如有困难及时与银行沟通。"
            },
            {
                "instruction": "如何提高信用卡额度？",
                "input": "",
                "output": "提高信用卡额度的方法：1. 按时还款：保持良好的还款记录 2. 增加消费：适当增加信用卡使用频率 3. 收入证明：提供更高的收入证明 4. 资产证明：提供房产、车辆等资产证明 5. 主动申请：定期主动申请提额 6. 多元化消费：在不同类型商户消费 7. 长期使用：长期保持良好的使用记录。银行会根据综合评估决定是否提额。"
            },
            
            # 网银安全相关
            {
                "instruction": "如何保护网银安全？",
                "input": "",
                "output": "保护网银安全的方法：1. 强密码：设置复杂且不易猜测的密码 2. 定期更换：定期更换网银密码 3. 安全环境：在安全的网络环境下使用网银 4. 及时退出：使用完毕后及时退出网银 5. 短信验证：开启短信验证功能 6. 设备绑定：绑定常用设备 7. 异常监控：关注账户异常变动 8. 防病毒：安装正版杀毒软件。发现异常立即联系银行。"
            },
            {
                "instruction": "网银转账限额是多少？",
                "input": "",
                "output": "网银转账限额因银行和账户类型而异：1. 普通账户：单笔限额通常1-5万元，日累计限额5-20万元 2. 高级账户：限额相对较高 3. 企业账户：限额通常更高 4. 认证方式：不同认证方式限额不同 5. 个人设置：可在银行规定范围内自行设置限额 6. 特殊业务：大额转账需要特殊审批。具体限额请咨询开户银行或查看网银设置。"
            },
            
            # 外汇相关
            {
                "instruction": "如何办理外汇业务？",
                "input": "",
                "output": "办理外汇业务需要：1. 有效身份证件：身份证、护照等 2. 外汇用途证明：如留学、旅游、商务等 3. 银行账户：在银行开立外汇账户 4. 额度申请：根据用途申请相应额度 5. 汇率了解：了解当前汇率情况 6. 手续费：了解相关手续费标准 7. 时间安排：外汇业务通常需要一定时间。建议提前准备材料并咨询银行具体要求。"
            },
            {
                "instruction": "外汇汇率是如何确定的？",
                "input": "",
                "output": "外汇汇率确定因素：1. 市场供求：外汇市场供求关系是主要因素 2. 经济基本面：国家经济状况影响汇率 3. 利率水平：利率差异影响资本流动 4. 政治因素：政治稳定性影响汇率 5. 央行政策：央行货币政策影响汇率 6. 国际环境：全球经济环境变化 7. 投机因素：市场投机行为影响短期波动。汇率是实时变动的，建议关注银行实时汇率。"
            },
            
            # 保险相关
            {
                "instruction": "银行保险产品有哪些？",
                "input": "",
                "output": "银行保险产品主要包括：1. 储蓄型保险：兼具储蓄和保障功能 2. 投资型保险：与投资市场挂钩 3. 保障型保险：纯保障功能 4. 年金保险：提供定期给付 5. 健康保险：医疗保障 6. 意外保险：意外伤害保障 7. 财产保险：财产损失保障。选择时需了解产品特点和风险，根据需求选择合适产品。"
            },
            {
                "instruction": "购买银行保险需要注意什么？",
                "input": "",
                "output": "购买银行保险注意事项：1. 产品了解：充分了解产品条款和保障范围 2. 费用透明：了解所有费用和扣费情况 3. 保障需求：根据实际需求选择产品 4. 缴费能力：确保有持续缴费能力 5. 犹豫期：利用犹豫期仔细考虑 6. 理赔条件：了解理赔条件和流程 7. 专业咨询：必要时咨询专业保险顾问 8. 比较选择：多比较不同产品。不要被高收益承诺误导。"
            }
        ]
        
        return banking_data
    
    def create_dataset(self, save_path: str = None) -> Dataset:
        """
        创建训练数据集
        
        Args:
            save_path (str): 保存路径，如果提供则保存到文件
            
        Returns:
            Dataset: 训练数据集
        """
        # 转换为datasets格式
        dataset_dict = {
            "instruction": [item["instruction"] for item in self.raw_data],
            "input": [item["input"] for item in self.raw_data],
            "output": [item["output"] for item in self.raw_data]
        }
        
        self.dataset = Dataset.from_dict(dataset_dict)
        
        # 保存数据集
        if save_path:
            self.dataset.save_to_disk(save_path)
            print(f"数据集已保存到: {save_path}")
        
        return self.dataset
    
    def load_dataset(self, load_path: str) -> Dataset:
        """
        从文件加载数据集
        
        Args:
            load_path (str): 数据集路径
            
        Returns:
            Dataset: 加载的数据集
        """
        if os.path.exists(load_path):
            self.dataset = Dataset.load_from_disk(load_path)
            print(f"数据集已从 {load_path} 加载")
        else:
            print(f"数据集路径不存在: {load_path}")
            self.dataset = self.create_dataset()
        
        return self.dataset
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        if self.dataset is None:
            self.create_dataset()
        
        return {
            "总样本数": len(self.dataset),
            "字段": list(self.dataset.features.keys()),
            "示例": self.dataset[0] if len(self.dataset) > 0 else None
        }
    
    def add_custom_data(self, instruction: str, output: str, input_text: str = ""):
        """添加自定义数据"""
        new_item = {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
        self.raw_data.append(new_item)
        print(f"已添加新数据: {instruction}")

def main():
    """测试数据集功能"""
    print("创建银行领域数据集...")
    
    # 创建数据集实例
    banking_dataset = BankingDataset()
    
    # 创建数据集
    dataset = banking_dataset.create_dataset("banking_dataset")
    
    # 显示数据集信息
    info = banking_dataset.get_dataset_info()
    print("\n数据集信息:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 显示几个示例
    print("\n数据集示例:")
    for i in range(min(3, len(dataset))):
        print(f"\n示例 {i+1}:")
        print(f"问题: {dataset[i]['instruction']}")
        print(f"回答: {dataset[i]['output'][:100]}...")

if __name__ == "__main__":
    main()
