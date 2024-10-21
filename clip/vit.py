from collections import OrderedDict

import torch
from torch import nn

# 768
# 来源：VIT模型将输入224*224尺寸化成16*16像素的patch，那么每个patch为16*16*3 = 768，其中3为图像通道，将每个patch投影为768维度表示，也就是本文中self.conv1通道为768的缘故。
# 196与49区别：196也是来源VIT将224变成16尺寸的patch，那么共有224*224 / (16*16) = 196，而本文的patch尺寸为32，变成224224 / (3232) = 49。

# 图像处理模型VIT的构造
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    # LayerNorm 类继承自 PyTorch 的 nn.LayerNorm，对 PyTorch 已有模块的定制或优化
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
# 这个自定义的 LayerNorm 类在前向传播时首先将输入张量 x 转换为 torch.float32（单精度浮点数），
# 然后调用父类的 forward 方法来计算层归一化。计算完成后，它将结果转回原始的数据类型（可能是 torch.float16 或其他类型），以确保输出与输入具有相同的数据类型。

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
# 通过激活函数神经网络就可以拟合各种曲线

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # attn_mask: 注意力掩码，用于控制哪些位置应该被注意。
        # self.attn: 使用nn.MultiheadAttention创建一个多头注意力层。
        # self.ln_1 和 self.ln_2: 使用前面定义的LayerNorm类创建层归一化层。
        # self.mlp: 一个小型的多层感知机（MLP），包括一个线性层、一个QuickGELU激活函数层，以及另一个线性层。
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # 注意力函数attention
    # 这个函数用于计算输入x的多头注意力。
    #
    # 如果attn_mask不为None，则将其数据类型和设备与输入x对齐。
    # 使用self.attn计算多头注意力，并返回输出。这里设置了need_weights=False，表示不需要返回注意力权重。

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        # 输入一个张量 x，并通过 self.resblocks（即所有 ResidualAttentionBlock 的序列）传递它。
        # 返回经过所有残差注意力块处理后的输出。
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        # 这边反正就是初始化整个VIT模型
        # input_resolution: 输入图像的分辨率（例如，224x224）。
        #
        # patch_size: 图像被切分的块的大小（例如，16x16）。
        #
        # width: 每个块的特征维度（也称为嵌入维度）。
        # layers: Transformer编码器的层数。
        #
        # heads: 在多头自注意力机制中的头数。
        # output_dim: 模型的输出维度，通常用于分类任务。
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        #-----------------------------------------------#
        #   224, 224, 3 -> 196, 768
        #-----------------------------------------------#
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.conv1：一个二维卷积层，用于将输入图像切分为小块，并提取每个小块的特征。输出特征的维度为width（嵌入维度）。卷积核的大小和步长都设置为
        # patch_size，因此输出的特征图的大小是(input_resolution // patch_size)x(input_resolution // patch_size)。

        scale = width ** -0.5
        #--------------------------------------------------------------------------------------------------------------------#
        #   class_embedding部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #   模型参数是在训练过程中通过反向传播和梯度下降进行更新的。
        #   最前面的那个token,它被添加到序列化的图像特征中，使得模型能够捕获全局信息。其形状为 [1, width]，
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个class_embedding，将class_embedding堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，class_embedding会与图片特征进行特征的交互。最终分类时，我们取出class_embedding的特征，利用全连接分类。
        #--------------------------------------------------------------------------------------------------------------------#
        #   196, 768 -> 197, 768
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #--------------------------------------------------------------------------------------------------------------------#
        #  初始化一个可学习的参数（嵌入向量）class_embedding。这里，nn.Parameter是一个特殊的类，用于将张量包装成模型参数，以便PyTorch能够自动对其进行优化。
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上class_embedding后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        #--------------------------------------------------------------------------------------------------------------------#
        #   197, 768 -> 197, 768
        #初始化position_embedding   (input_resolution // patch_size) ** 2就是图像中所有块的数量。
        #生成一个形状为((input_resolution // patch_size) ** 2 + 1, width)的随机张量，用于初始的位置嵌入。
        # nn.Parameter(...)将张量转换为模型参数，这样它可以在训练过程中被优化。就像训练的权重参数
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # 初始化了两个层归一化（Layer Normalization）层。层归一化是一种用于深度神经网络的技术，它有助于加速训练并改善模型的性能。
        # 这一行初始化了一个Transformer模型。width是输入和输出特征的维度，layers是Transformer中的层数，heads是每个多头注意力机制中的头数。
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        # 这里初始化了一个线性投影层，它将Transformer的输出从width维映射到output_dim维。这通常用于将Transformer的输出转换为所需的输出格式或维度。
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # 这行代码将输入张量x通过一个卷积层self.conv1。输出张量的形状是[*, width, grid, grid]，其中*代表任意的维度（如批处理大小），width是卷积层的输出通道数，grid是卷积后空间维度的大小。
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # 这里将张量x的形状重新调整为[*, width, grid ** 2]，其中grid ** 2是将两个空间维度grid合并成一个维度。
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 通过permute方法，将张量x的维度顺序重新排列，变为[*, grid ** 2, width]。
        # 通过调用permute方法，我们将维度从[批处理大小, 通道数, 序列长度]更改为[批处理大小, 序列长度, 通道数]。通常是为了满足Transformer模型或其他序列处理模型的输入要求
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # self.class_embedding.to(x.dtype)将class_embedding就是开头token格式转换为与输入的x相同的数据类型。
        x = x + self.positional_embedding.to(x.dtype)
        # 位置嵌入向量positional_embedding被转换为与x相同的数据类型，然后加到x上，合并成一个token。
        x = self.ln_pre(x)
        # 将加了位置嵌入的x通过层归一化层self.ln_pre。
        # 层归一化通常放在激活函数之前，有时也放在全连接层或卷积层之后。
        # 在您的代码中，self.ln_pre 可能被用于稳定模型的内部表示，并有助于梯度在反向传播过程中更好地流动。
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 从[N, L, D]（N是批处理大小，L是序列长度，D是特征维度）变为[L, N, D]。
        x = self.transformer(x)
        # 将重新排列后的张量x输入到Transformer模型中。
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 再次重新排列张量x的维度，从[L, N, D]变回[N, L, D]。
        x = self.ln_post(x[:, 0, :])
        # 这个切片操作提取了 x 第二个维度的第一个元素（假设 x 的第二个维度代表序列长度，这里取的是序列中的第一个元素或token）。这个操作通常在处理 Transformer 模型的输出时使用，
        # 因为 Transformer 的输出通常包含序列中每个位置的表示，而我们可能只对序列的某个特定位置（比如开始位置或结束位置）感兴趣。
        # 层归一化通常在每个样本的每个特征维度上独立地计算均值和标准差，并使用这些统计量来归一化输入。
        if self.proj is not None:
            x = x @ self.proj
        # 如果self.proj（投影层）不为空，则将x与self.proj进行矩阵乘法，以将x映射到新的特征空间，@ 运算符，这是 Python 中矩阵乘法的简写
        return x