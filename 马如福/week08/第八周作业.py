修改点为：修改之后的准确率在百分之九十左右
在model.py中，将类SentenceEncoder中的 forward修改如下
    def forward(self, sentence1, sentence2, sentence3):
        anchor = self.sentence_encoder(sentence1)
        positive = self.sentence_encoder(sentence2)
        negative = self.sentence_encoder(sentence3)
        return self.cosine_triplet_loss(anchor, positive, negative)

将load.py中，random_train_sample方法整改如下：
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 确保有至少两个不同的类别才能生成负样本
        if len(standard_question_index) < 2:
            raise ValueError("至少需要两个不同的类别才能生成triplet样本")

        # 随机选择锚点和正样本所属的类别p
        p = random.choice(standard_question_index)
        # 确保类别p下至少有2个问题（才能选出锚点和正样本）
        while len(self.knwb[p]) < 2:
            p = random.choice(standard_question_index)

        # 从类别p中选锚点(anchor)和正样本(positive)（两个不同的样本）
        anchor, positive = random.sample(self.knwb[p], 2)

        # 选择负样本所属的类别n（必须与p不同）
        n = random.choice(standard_question_index)
        while n == p:
            n = random.choice(standard_question_index)

        # 从类别n中选负样本(negative)
        negative = random.choice(self.knwb[n])

        return [anchor, positive, negative]
