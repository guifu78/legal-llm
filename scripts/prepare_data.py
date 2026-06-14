# -*- coding: utf-8 -*-
"""
法律数据准备脚本
用于准备训练数据集
支持多种数据格式：问答、对话、知识库
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent

class LegalDataProcessor:
    """法律数据处理器"""
    
    def __init__(self):
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def create_sample_legal_qa(self) -> List[Dict]:
        """创建法律问答示例数据"""
        
        legal_qa_data = [
            {
                "instruction": "请解释什么是合同的成立要件",
                "input": "",
                "output": "合同的成立要件包括：\n1. 当事人：合同必须有双方或多方当事人\n2. 意思表示：当事人必须有订立合同的意思表示\n3. 标的：合同必须有明确的标的\n4. 内容：合同内容必须确定或可能确定\n5. 形式：根据法律规定，某些合同需要采用书面形式"
            },
            {
                "instruction": "什么是诉讼时效？",
                "input": "",
                "output": "诉讼时效是指权利人在法定期间内不行使权利，即丧失请求人民法院依诉讼程序强制义务人履行义务的权利的制度。\n\n根据《中华人民共和国民法典》第一百八十八条，向人民法院请求保护民事权利的诉讼时效期间为三年。法律另有规定的，依照其规定。"
            },
            {
                "instruction": "什么是正当防卫？",
                "input": "",
                "output": "正当防卫是指为了使国家、公共利益、本人或者他人的人身、财产和其他权利免受正在进行的不法侵害，而采取的制止不法侵害的行为，对不法侵害人造成损害的，属于正当防卫，不负刑事责任。\n\n正当防卫的构成要件：\n1. 起因条件：存在不法侵害\n2. 时间条件：不法侵害正在进行\n3. 对象条件：针对不法侵害人本人\n4. 主观条件：具有防卫意识\n5. 限度条件：没有明显超过必要限度"
            },
            {
                "instruction": "如何签订一份有效的劳动合同？",
                "input": "",
                "output": "签订劳动合同需要注意以下要点：\n\n1. 合同必备条款：\n   - 用人单位基本信息\n   - 劳动者个人信息\n   - 劳动合同期限\n   - 工作内容和工作地点\n   - 工作时间和休息休假\n   - 劳动报酬\n   - 社会保险\n\n2. 注意事项：\n   - 合同应在用工之日起一个月内签订\n   - 试用期最长不超过六个月\n   - 合同文本双方各执一份"
            },
            {
                "instruction": "什么是知识产权中的商标侵权？",
                "input": "",
                "output": "商标侵权是指未经商标注册人的许可，在同一种商品或者类似商品上使用与其注册商标相同或者近似的商标的行为。\n\n商标侵权的主要表现：\n1. 未经许可在同一种商品上使用相同商标\n2. 未经许可在类似商品上使用相同或近似商标\n3. 销售侵犯注册商标专用权的商品\n4. 伪造、擅自制造他人注册商标标识"
            },
            {
                "instruction": "什么是行政诉讼？",
                "input": "",
                "output": "行政诉讼是指公民、法人或者其他组织认为行政机关的行政行为侵犯其合法权益，依法向人民法院提起诉讼，由人民法院进行审理并作出裁判的诉讼活动。\n\n行政诉讼的特点：\n1. 被告恒定为行政机关\n2. 审查对象是行政行为\n3. 不适用调解（行政赔偿、补偿除外）\n4. 举证责任由被告承担"
            }
        ]
        
        return legal_qa_data
    
    def save_data(self, data: List[Dict], filename: str):
        """保存数据到文件"""
        
        filepath = self.data_dir / f"{filename}.jsonl"
        
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"数据已保存到: {filepath}")
        print(f"  数据条数: {len(data)}")
    
    def process_all(self):
        """处理所有数据"""
        
        print("=" * 50)
        print("法律数据处理器")
        print("=" * 50)
        
        # 创建问答数据
        print("\n创建法律问答数据...")
        qa_data = self.create_sample_legal_qa()
        
        # 打乱数据
        random.seed(42)
        random.shuffle(qa_data)
        
        # 分割训练集和验证集
        split_idx = int(len(qa_data) * 0.8)
        train_data = qa_data[:split_idx]
        val_data = qa_data[split_idx:]
        
        # 保存数据
        print("\n保存数据...")
        self.save_data(train_data, "train")
        self.save_data(val_data, "val")
        
        print("\n" + "=" * 50)
        print("数据准备完成！")
        print("=" * 50)

def main():
    """主函数"""
    processor = LegalDataProcessor()
    processor.process_all()

if __name__ == "__main__":
    main()
