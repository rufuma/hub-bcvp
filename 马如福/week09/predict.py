# -*- coding: utf-8 -*-
import torch
import json
from transformers import BertTokenizerFast
from config import Config
from model import TorchModel

"""
模型预测测试代码
功能：输入文本，输出识别出的实体（PERSON/LOCATION/TIME/ORGANIZATION）
"""


class NERPredictor:
    def __init__(self, config, model_path):
        """
        初始化预测器
        :param config: 配置字典（Config）
        :param model_path: 训练好的模型权重路径（如model_output/epoch_20.pth）
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载BERT Fast分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])

        # 2. 加载schema（标签索引→标签名）
        self.id2label = self.load_id2label(config["schema_path"])

        # 3. 初始化模型并加载权重
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 预测模式（关闭dropout）

    def load_id2label(self, schema_path):
        """加载schema，生成 索引→标签名 的映射"""
        with open(schema_path, encoding="utf8") as f:
            label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        return id2label

    def predict(self, text):
        """
        单文本预测
        :param text: 输入文本（如"张三2025年在北京市海淀区工作"）
        :return: 实体结果字典（key：实体类型，value：实体列表）
        """
        # 1. 文本编码（和训练时保持一致）
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,  # 字符→token映射，用于还原实体
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"  # 返回tensor格式
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        offset_mapping = encoded["offset_mapping"].cpu().numpy()[0]  # (seq_len, 2)

        # 2. 模型预测
        with torch.no_grad():
            pred = self.model(input_ids, attention_mask)  # CRF解码结果

        # 3. 处理预测结果（CRF输出是list of list，需展平）
        pred_labels = pred[0]  # 取第一条数据的预测标签索引

        # 4. 还原为字符级标签，并解码实体
        char_labels = self.token2char_label(pred_labels, offset_mapping, len(text))
        entities = self.decode_entities(text, char_labels)

        return entities

    def token2char_label(self, token_labels, offset_mapping, char_len):
        """将token级标签还原为字符级标签"""
        char_labels = [8] * char_len  # 默认O类（索引8）
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 跳过特殊token（CLS/SEP/PAD）
                continue
            if start >= char_len:  # 截断后的token
                continue
            # token对应字符的起始位置赋值
            char_labels[start] = token_labels[token_idx]
            # 多字符token的后续字符继承label（保证实体完整性）
            for char_idx in range(start + 1, end):
                if char_idx < char_len:
                    char_labels[char_idx] = token_labels[token_idx]
        return char_labels

    def decode_entities(self, text, char_labels):
        """解码字符级标签为实体（和evaluate.py逻辑一致）"""
        entities = {
            "PERSON": [],
            "LOCATION": [],
            "TIME": [],
            "ORGANIZATION": []
        }
        # 标签索引转字符串（方便正则匹配）
        label_str = "".join([str(x) for x in char_labels[:len(text)]])

        # 匹配各类实体（B-X + I-X的组合）
        # LOCATION: B=0, I=4 → 正则匹配 04+
        for match in re.finditer(r"(04+)", label_str):
            s, e = match.span()
            entities["LOCATION"].append(text[s:e])
        # ORGANIZATION: B=1, I=5 → 15+
        for match in re.finditer(r"(15+)", label_str):
            s, e = match.span()
            entities["ORGANIZATION"].append(text[s:e])
        # PERSON: B=2, I=6 → 26+
        for match in re.finditer(r"(26+)", label_str):
            s, e = match.span()
            entities["PERSON"].append(text[s:e])
        # TIME: B=3, I=7 → 37+
        for match in re.finditer(r"(37+)", label_str):
            s, e = match.span()
            entities["TIME"].append(text[s:e])

        return entities

    def batch_predict(self, texts):
        """
        批量预测
        :param texts: 文本列表（如["张三在上海", "2025年公司开会"]）
        :return: 实体结果列表（每个元素对应单文本的实体字典）
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


# --------------------- 测试示例 ---------------------
if __name__ == "__main__":
    import re  # 补充导入正则（decode_entities需要）

    # 1. 配置参数（和训练时一致）
    config = Config
    # 2. 模型权重路径（替换为你训练好的模型路径，如epoch_20.pth）
    model_path = r"model_output/epoch_10.pth"

    # 3. 初始化预测器
    predictor = NERPredictor(config, model_path)

    # 4. 单文本测试
    test_text = "张三2025年3月在北京市海淀区百度大厦工作"
    single_result = predictor.predict(test_text)
    print("=" * 50)
    print(f"输入文本：{test_text}")
    print("单文本预测结果：")
    for entity_type, entity_list in single_result.items():
        print(f"{entity_type}：{entity_list}")

    # 5. 批量文本测试
    test_texts = [
        "李四2024年5月于深圳市腾讯大厦参加会议",
        "王五在2023年加入阿里巴巴集团"
    ]
    batch_results = predictor.batch_predict(test_texts)
    print("=" * 50)
    print("批量预测结果：")
    for idx, (text, result) in enumerate(zip(test_texts, batch_results)):
        print(f"文本{idx + 1}：{text}")
        for entity_type, entity_list in result.items():
            print(f"  {entity_type}：{entity_list}")
