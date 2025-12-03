      model  learning_rate    test_accuracy predict_time_100(s)
0  fasttext          0.010           0.8415           0.0002
1   textcnn          0.001           0.8082           0.0080
2    bilstm          0.001           0.6664           0.0117
3      bert          0.000           0.9133           2.3581

根据上面的实验得出如下结论
精度排序：BERT > TextCNN > BiLSTM > fastText，预训练模型（BERT）在语义理解上优势明显。
速度排序：fastText > TextCNN > BiLSTM > BERT，简单模型（fastText）训练与预测速度更快。
