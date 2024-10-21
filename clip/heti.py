import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
# 用Transformer引入了bert文件，用VisionTransformer引入了Vit，这个文件才是中心
from .bert import Transformer

# from .modelECA import Bottleneck, ECA_ResNet
# from .model import Bottleneck, ModifiedResNet
# from .ECA_aug import ECA_aug, SplitBlock
# from .ecares import eca_resnet50
# from .shufflenetv2 import ShuffleNetV2
# from .augshufflenetv2 import AugShuffleNetV2

# from .transnext_cuda import transnext_tiny
from .simple_tokenizer import SimpleTokenizer,tokenize
# from .longclip import tokenize
from .vit import VisionTransformer


# 多模态融合的操作，另外两个bert和VIT的模型架构好了，在此输入数据后实例化成对象，最后进行CLIP的相似度计算


class CLIP(nn.Module):
    def __init__(
            self,
            bert_type="openai",
            # 这边都是默认参数，假如那边给值没有对应的参数，才用，就是传递过来是vision_layers是分组，这边是int,因为调用的是ECA_Resnet
            embed_dim=512,
            # vision
            input_resolution=224,
            #     视觉Transformer的层数。
            vision_layers=12,
            vision_width=768,
            vision_patch_size=32,
            # text   context_length文本部分的上下文长度，即模型可以处理的文本标记的最大数量
            context_length=77,
            transformer_layers=12,
            #     文本Transformer的层数。
            transformer_width=768,
            transformer_heads=12,
            vocab_size=49408,
            #     词汇表的大小。指的是模型所使用的词汇或标记（tokens）的数量
            #     对于CLIP模型中的文本部分，vocab_size通常与所使用的文本编码器（例如BERT）的词汇表大小相对应。
            #     BERT模型（无论是“openai”版本还是“huggingface”版本）通常有一个固定的词汇表，
            #     这个词汇表是在预训练过程中确定的，并且通常包含了大量的单词片段（word pieces）或子词单元（subword units），以便能够表示各种不同的单词和短语
            **kwargs
    ):
        super().__init__()  # nn.Module调用父类的init

        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)) and vision_layers:
            vision_heads = vision_width * 32 // 64
            # self.visual = ECA_ResNet(
            #     block=Bottleneck,
            #     num_blocks=vision_layers,
            #     num_classes=512
            #     )
            #             self.visual = eca_resnet50(
            #                 k_size=[3, 3, 3, 3],
            #                 num_blocks=vision_layers,
            #                 num_classes=512
            #              )
            #             self.visual = ECA_aug(
            #                 net_size=0.5,
            #                 num_blocks=vision_layers,
            #                 num_classes=512
            #              )
            #             self.visual = ShuffleNetV2(
            #                 net_size=0.5,
            #                 num_blocks=vision_layers,
            #                 num_classes=512
            #              )
            self.visual = AugShuffleNetV2(
                net_size=1.5,
                num_blocks=vision_layers,
                num_classes=512
            )
            print("eca_aug")


        elif isinstance(vision_layers, int):
            # 文本部分初始化
            vision_heads = vision_width // 64
            # 是视觉部分的Transformer模型，它根据提供的参数（如输入分辨率、块大小、宽度、层数、注意力头数和输出维度）进行初始化VIT。
            self.visual = VisionTransformer(
                input_resolution=input_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
            print("ViT")
        else:
            # 处理 vision_layers 既不是整数也不是（非空）列表/元组的情况
            print("Invalid type for vision_layers")
        # 文本部分初始化
        # 文本类型
        self.bert_type = bert_type
        if bert_type == "openai":
            self.tokenizer = SimpleTokenizer()
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        elif bert_type == "huggingface":
            # 使用Hugging Face的预训练BERT模型和标记器。这里加载了指定路径下的中文BERT模型和标记器。
            # transformer_width 是从加载的BERT模型的配置中提取的隐藏层大小。
            self.tokenizer = BertTokenizer.from_pretrained(r'bert-base-chinese')
            self.transformer = BertModel.from_pretrained(r'bert-base-chinese')
            transformer_width = self.transformer.config.hidden_size
        # self.text_projection 是一个参数化的投影层，用于将Transformer的输出投影到与视觉部分相同的嵌入维度。
        # 使用正态分布初始化self.text_projection的参数。
        # self.ln_final 是一个层规范化层，用于规范化Transformer的输出。
        # self.logit_scale 是一个可学习的参数，用于缩放模型的输出logits，以控制模型的置信度。
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        self.ln_final = nn.LayerNorm(transformer_width)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    # @property 是一个装饰器，它使得dtype方法表现得像一个属性。当你访问obj.dtype时，Python会自动调用dtype(obj)方法，而不需要加括号。就是无参调用
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # 视觉Transformer模型第一个卷积层的权重的数据类型。这通常用于确保模型的其他部分使用与视觉模块相同的数据类型。

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        # 这个方法用于构建注意力掩码，通常用于Transformer模型中的自注意力机制。
        # 创建一个大小为context_length x context_length的空张量。
        mask = torch.empty(self.context_length, self.context_length)
        # 将这个张量的所有元素填充为负无穷大。在PyTorch中，使用负无穷大作为注意力掩码的一个常见方法，因为它会在softmax运算中使对应的注意力权重接近于0
        mask.fill_(float("-inf"))

        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        # image.type(self.dtype) 将输入图像的数据类型转换为模型视觉部分所使用的数据类型，强制转换。
        # self.visual(image.type(self.dtype)) 将处理后的图像传递给上面初始化好的VITransformer模型，并返回其输出。
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # 这个方法接受一个text参数，它是要被编码的文本输入。
        if self.bert_type == "openai":
            # 使用tokenize函数和self.tokenizer对文本进行分词，并将结果转移到模型视觉部分的第一个卷积层权重的设备上。
            text = tokenize(self.tokenizer, text).to(self.visual.conv1.weight.device)
            # 将分词后的文本转换为token块
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            # 每个文本块加上位置embedding
            x = x + self.positional_embedding.type(self.dtype)
            # 改变张量的维度顺序，以适应Transformer模型的输入要求。
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            # 转回来
            x = self.ln_final(x).type(self.dtype)
            # 应用层归一化，并通过一个线性投影得到最终的文本表示。
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        elif self.bert_type == "huggingface":
            # 分词,设备选择好,然后输入到transformer，用投影层得到输出
            # 使用Hugging Face的tokenizer对文本进行编码，返回PyTorch张量，并启用填充。
            # 这个参数指示分词器是否需要对输入进行填充（padding）。填充是为了确保所有输入序列的长度相同，这样模型就可以处理批量的输入数据。
            x = self.tokenizer(text, return_tensors="pt", padding=True)
            # 将输入ID、注意力掩码和token类型ID转移到模型视觉部分的设备上。
            input_ids = x.input_ids.to(self.visual.conv1.weight.device)
            attention_mask = x.attention_mask.to(self.visual.conv1.weight.device)
            token_type_ids = x.token_type_ids.to(self.visual.conv1.weight.device)
            # 使用Hugging Face的Transformer模型处理编码后的文本，并获取池化层的输出。
            x = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).pooler_output
            # 应用层归一化，并通过一个线性投影得到最终的文本表示。
            # self.ln_final(x): 将x通过一个层归一化（Layer Normalization）层，这有助于模型更好地训练。
            # .type(self.dtype): 确保x的数据类型与模型的其他部分一致。
            # x @ self.text_projection: 再次通过线性变换（投影）将归一化后的嵌入转换为另一种表示形式。
            x = self.ln_final(x).type(self.dtype)
            x = x @ self.text_projection

        return x

    def forward(self, image, text):
        # 使用encode_image方法（或层）对输入图像进行编码，
        # 得到图像的特征表示image_features使用encode_image方法（或层）对输入图像进行编码，得到图像的特征表示image_features
        image_features = self.encode_image(image)
        # 提取文本特征:
        text_features = self.encode_text(text)
        # 特征归一化:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 计算logits的缩放因子:

        logit_scale = self.logit_scale.exp()
        # 计算图像与文本之间的匹配分数:使用缩放因子logit_scale和点积操作@计算每个图像特征向量与所有文本特征向量之间的匹配分数。
        logits_per_image = logit_scale * image_features @ text_features.t()
        # 结果logits_per_image是一个矩阵，其中每一行代表一个图像与所有文本特征向量的匹配分数。
        # 通过转置logits_per_image矩阵，我们得到logits_per_text，其中每一行代表一个文本与所有图像特征向量的匹配分数。
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
