import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BertSelfAttention(nn.Module):
    # 多头自注意力层
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 生成Q，K，V的线性层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        # shape为[batch_size, num_head, seq_len, d_k]
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算公式 Q.K^T/sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn_weights = F.softmax(attn_scores, dim=-1)
        # matmul 最后两维矩阵乘法，前导维度可广播
        output = torch.matmul(attn_weights, v)

        # 将多头结果进行拼接 + 线性投影
        # contiguous的目的是为了后面的view可执行
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_out(output)
        return output, attn_weights


class BertFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Linear(gelu(linear(x)))
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class BertEncoderLayer(nn.Module):
    # 单个BERT Encoder层
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = BertSelfAttention(d_model, num_heads)
        self.feed_forward = BertFeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 归一化
        atten_output, atten_weights = self.attn(x)
        x = self.norm1(x + self.dropout(atten_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, atten_weights


class BertEncoder(nn.Module):
    def __init__(self, d_model=768, num_heads=12, num_layers=12):
        super().__init__()
        # 堆叠num_layers个encoder层
        self.layers = nn.ModuleList([BertEncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        attn_weights_list = []
        for layer in self.layers:
            x, atten_weights = layer(x)
            attn_weights_list.append(atten_weights)
        return x, attn_weights_list


if __name__ == "__main__":
    bert_encoder = BertEncoder(d_model=768, num_heads=12, num_layers=12)
    x = torch.randn(2, 10, 768)

    output, attn_weights = bert_encoder(x)
    print("Encoder 输出形状：", output.shape)
    print("注意力权重层数：", len(attn_weights))
