
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from test_layers.Embed import DataEmbedding
from test_layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.num_kernels = configs.num_kernels
        # self.hidden_dim = configs.hidden_dim
        # self.dropout = configs.dropout

        # 定义卷积层
        self.conv = nn.Sequential(
            Inception_Block_V1(self.d_model, configs.d_ff,
                               num_kernels=self.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, self.d_model,
                               num_kernels=self.num_kernels)
        )
        # -------------------------
        # 这里增加一个或多个线性层
        # hidden_dim = self.hidden_dim  # 可以自己设定
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(configs.d_model, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(hidden_dim, configs.d_model),
        #     nn.ReLU(),
        # )
        # -------------------------

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # 确保输入数据的长度可以被period整除
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([x.shape[0], length - T, x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x

            # reshape 和卷积处理
            out = out.reshape(B, out.shape[1] // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # 经过 MLP Head
        # res = self.mlp_head(res)

        # 残差连接
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc):
        # 正常化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 数据嵌入
        enc_out = self.enc_embedding(x_enc, None)

        # TimesNet 模块
        for i in range(self.configs.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 投影回原始维度
        dec_out = self.projection(enc_out)

        # 反正常化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, enc_out.shape[1], 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, enc_out.shape[1], 1))

        return dec_out
