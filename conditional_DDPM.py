from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from parameter_predictor import PhysicsParamPredictor


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class FeatureModulationEmbed(nn.Module):
    """特征调制嵌入模块 - 专为数值型特征设计"""

    def __init__(self, input_dim, emb_dim):
        super(FeatureModulationEmbed, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # 特征独立嵌入层
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, emb_dim // 4),  # 每个特征映射到 emb_dim//4 维
                nn.GELU(),
                nn.Linear(emb_dim // 4, emb_dim // 4)
            ) for _ in range(input_dim)
        ])

        # 特征加权融合层
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

        # 融合后的特征投影到目标维度
        self.fusion_proj = nn.Sequential(
            nn.Linear((emb_dim // 4) * input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.norm = nn.LayerNorm(emb_dim)

        # 空条件嵌入（可学习）
        self.null_embed = nn.Parameter(torch.zeros(1, emb_dim))

    def forward(self, c, context_mask=None):
        """
        c: 条件向量 [batch, 4]
        context_mask: dropout掩码 [batch]
        """
        batch_size = c.size(0)

        # 特征独立嵌入
        embeddings = []
        for i in range(self.input_dim):
            feature = c[:, i].unsqueeze(1)  # [batch, 1]

            emb = self.feature_embeddings[i](feature) # * self.feature_weights[i]

            embeddings.append(emb)

        # 拼接所有特征嵌入
        concatenated = torch.cat(embeddings, dim=1)  # [batch, (emb_dim//4)*4]

        # 投影到目标维度
        # weighted_concatenated = concatenated * self.feature_weights.view(1, -1).repeat(batch_size, 1)

        combined = self.fusion_proj(concatenated)  # [batch, emb_dim]

        # 层归一化
        combined = self.norm(combined)

        # 应用条件dropout
        if context_mask is not None:
            # 创建混合嵌入
            mask = context_mask.view(-1, 1).float()
            null_emb = self.null_embed.expand(batch_size, -1)
            combined = (1 - mask) * combined + mask * null_emb

        return combined


class FeatureInteractionEmbed(nn.Module):
    """特征交互增强嵌入 - 自动学习特征关系"""

    def __init__(self, input_dim, emb_dim):
        super(FeatureInteractionEmbed, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # 特征交互层 - 捕捉特征间关系
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, 2 * emb_dim),
            nn.GELU()
        )

        # 特征调制层 - 增强关键特征
        self.feature_modulation = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # 特征权重学习
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        self.weight_proj = nn.Linear(input_dim, emb_dim)

        # 空条件嵌入
        self.null_embed = nn.Parameter(torch.zeros(1, emb_dim))

        # 输出归一化
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, c, context_mask=None):
        batch_size = c.size(0)

        # 1. 特征交互
        interaction = self.interaction_layer(c)  # [batch, 2*emb_dim]

        # 2. 特征权重学习
        weights = F.softmax(self.feature_weights, dim=0)
        weighted_c = c * weights.unsqueeze(0)  # [batch, input_dim]
        weight_emb = self.weight_proj(weighted_c)  # [batch, emb_dim]

        # 3. 特征调制
        modulated = self.feature_modulation(interaction)  # [batch, emb_dim]

        # 4. 结合特征交互和权重信息
        combined = modulated * weight_emb  # 乘法交互增强特征重要性
        combined = self.norm(combined)

        # 5. 条件dropout
        if context_mask is not None:
            mask = context_mask.view(-1, 1).float()
            null_emb = self.null_embed.expand(batch_size, -1)
            combined = (1 - mask) * combined + mask * null_emb

        return combined


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        #self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(12), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed3 = EmbedFC(1, 1 * n_feat)

        self.contextembed1 = FeatureInteractionEmbed(n_classes, 2 * n_feat)
        self.contextembed3 = FeatureInteractionEmbed(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspospip install d2l==0.17.6e2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 12, 12),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        #self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        '''
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
        '''

        self.out = nn.Conv2d(2 * n_feat, 1, 3, 1, 1)

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        #down3 = self.down3(down2)

        hiddenvec = self.to_vec(down2)

        # embed context, time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat, 1, 1)

        cemb1 = self.contextembed1(c, context_mask).view(-1, self.n_feat * 2, 1, 1)
        cemb3 = self.contextembed3(c, context_mask).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        #up2 = self.up1(cemb1 * up1 + temb1, down3)  # add and multiply embeddings

        up3 = self.up2(cemb1 * up1 + temb1, down2)
        up4 = self.up3(cemb3 * up3 + temb3, down1)

        out = self.out(torch.cat((up4, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        # self.cp_model = PhysicsParamPredictor().to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.para_loss = nn.SmoothL1Loss()

        self.l1 = 0.25

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(x.shape[0], 1) + self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        noise_pred = self.nn_model(x_t, c, _ts / self.n_T, context_mask)

        return self.loss_mse(noise, noise_pred)

    def sample(self, n_sample, c_i, size, device, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = c_i.reshape((1, 4))
        c_i = c_i.repeat(n_sample, 1)
        #c_i = c_i.reshape(n_sample, c_i.size())
        # don't drop context at test time
        context_mask = torch.zeros(n_sample, 1).to(device)

        c_i = c_i.to(device)

        x_i_store = []  # keep track of generated steps in case want to plot something
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i, c_i, t_is, context_mask)

            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
