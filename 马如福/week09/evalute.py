# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.valid_dataset = self.valid_data.dataset
        self.valid_sentences = self.valid_dataset.sentences
        self.valid_offset_mapping = self.valid_dataset.offset_mappings


    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        batch_size = self.config["batch_size"]
        #total_batch = len(self.valid_data)
        for index, batch_data in enumerate(self.valid_data):
            #sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, attention_mask, trun_labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id, attention_mask) #不输入labels，使用模型当前参数进行预测
            start_idx = index * batch_size
            end_idx = min((index + 1) * batch_size, len(self.valid_sentences))
            batch_sentences = self.valid_sentences[start_idx:end_idx]
            batch_offset = self.valid_offset_mapping[start_idx:end_idx]

            self.write_stats(trun_labels, pred_results, batch_sentences, batch_offset)
        self.show_stats()
        return

    def write_stats(self, true_labels, pred_results, sentences, offset_mappings):
        assert len(sentences) == len(offset_mappings)
        batch_size = len(true_labels)

        # 处理非CRF的情况（取argmax）
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)

        for idx in range(batch_size):
            if idx >= len(sentences):  # 处理最后一个batch不足的情况
                break
            # 1. 获取单条数据
            raw_sentence = sentences[idx]
            offset_mapping = offset_mappings[idx]
            true_label = true_labels[idx].cpu().detach().tolist()
            pred_label = pred_results[idx] if self.config["use_crf"] else pred_results[idx].cpu().detach().tolist()

            # 2. 还原为字符级label（核心：反向映射）
            char_true_labels = self.token2char_label(true_label, offset_mapping, len(raw_sentence))
            char_pred_labels = self.token2char_label(pred_label, offset_mapping, len(raw_sentence))

            # 3. 解码实体
            true_entities = self.decode(raw_sentence, char_true_labels)
            pred_entities = self.decode(raw_sentence, char_pred_labels)

            # 4. 更新统计
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def token2char_label(self, token_labels, offset_mapping, char_len):
        """将token级label还原为字符级label"""
        char_labels = [8] * char_len  # 默认O类（8）
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 跳过特殊token
                continue
            if start >= char_len:  # 截断后的token
                continue
            # token对应字符的起始位置赋值（子词覆盖父词）
            char_labels[start] = token_labels[token_idx]
        return char_labels

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


