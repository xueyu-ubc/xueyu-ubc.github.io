---
title: Vision-Language Models
date: 2024-09-12 15:00:00
tags: VLM
categories: Multimodal LLMs
mathjax: true
---


# Architecture of Vision-Language Models
视觉语言模型（VLM）集成了视觉（图像）和文本（语言）信息处理功能。旨在理解和生成涉及图像和文本的内容，从而能够执行图像标题、视觉问题解答和文本到图像的生成等任务。VLM 的架构以视觉和语言模式的有效融合 (fusion) 为核心，这一过程需要复杂的机制来调整和整合来自文本和图像的信息。下面，我们从模态融合、模态对齐以及训练策略三个角度介绍 VLM 的常见框架。
## 模态融合 
1. 早期融合。在这种方法中，视觉输入和文本输入在早期阶段就已经融合在一个空间中。

2. 中间融合。在对每种模态进行一定的独立处理之后再进行融合处理。

3. 后期/决策层融合。在后期融合中，两种模态通过进行深层的独立处理，并在接近输出时进行融合。这种方法能让两种模式保持更长时间的分离，从而在融合前进行更专业的处理。

## 模态对齐
1. 跨模态注意力。模型通常使用注意力机制（如Transformer），来对齐一种模态（例如图像中的物体）与另一种模态（例如句子中的词语）之间的元素。这有助于模型理解图像的**特定部分**如何与**特定的文本元素**相关联，从而增强模型的跨模态理解能力。

2. 联合嵌入空间。创建一个联合或共享的表示空间，将视觉和文本特征投射到该空间中。这个空间的设计旨在使得来自不同模态的语义相似概念在空间中相互接近。通过这种方式，无论是视觉信息还是文本信息，都能在同一个空间中找到相应的语义关联。

## 训练策略
1. 对比学习。对比学习通常用于模态对齐，训练模型将语义相似的文本和图像表示拉近，同时将语义不相似的表示推远。通过这种方式，模型能更好地理解并区分不同模态间的语义关系。

2. 多任务学习。通过在多个任务上训练模型（如图像字幕生成、视觉问答等），提升模型理解和整合多模态信息的能力。多任务学习可以让模型在处理不同模态的复杂任务时变得更加灵活和高效。

# VLMs 常用的连接模块
为了有效地整合视觉和语言这两种不同的模态，VLMs 使用了专门的机制，例如Adapters和线性层。本部分将详细介绍各类VLMs常用的构建组件以及如何在模型中将视觉与语言输入联系起来。

## Adapters/MLPs/Fully Connected Layers in VLMs
Adapters 是一种小型的神经网络模块，可以插入到已有模型中。在VLMs的上下文中，Adapters 有助于整合视觉和文本数据，它通过转换一种模态的表示，使之与另一模态兼容，从而实现模态对齐。Adapters 通常由几层全连接层（即多层感知机，MLP）构成。Adapters 接收一种编码器（例如视觉编码器）的输出，并将其转换成另一种编码器或解码器（如语言模型）可以处理的格式。

线性层或全连接层是神经网络的基本组成部分。在VLMs中，线性层在处理视觉编码器的输出时至关重要。图像经过视觉编码器（如卷积神经网络CNN或基于Transformer的视觉模型）处理后，会产生特征表示。为了让这些特征更适用于文本任务，线性层将视觉特征转换成兼容文本模态的格式，以便后续的融合处理。

在VLM中，视觉数据经过 Adapters 和线性层处理后，通常会与文本数据进行融合。融合通常在进入语言模型之前或在语言模型内部进行，使VLM能够结合视觉和文本信息生成回应或分析。在一些 VLM 中，整个模型（包括视觉编码器、线性层和语言模型）可以进行端到端训练。这种方法能帮助模型更好地学习视觉和文本信息的整合与解释，从而提升整体理解和生成的效果。在实际应用中，选择只用线性层、只用适配器，还是两者结合，主要取决于模型的设计目标和计算资源的限制。

# CLIP

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)

## Approach

核心思想：利用自然语言的监督信号来训练一个较好的视觉模型。

利用自然语言的监督信号去训练一个视觉模型的好处在于：

1. 以往的视觉模型训练需要对图片进行类别标注，消耗大量的人力资源。结合可以直接获取的文本信息，不需要对图片进行额外标注，数据规模更大，模型的输入输出不再是单一的标签，自由度更高。

2. 相较于单一模态特征（比如单一视觉特征），使用多模态的特征，很容易进行 zero shot 的迁移学习。

目标函数的选择：如果使用预测目标函数，即根据图片去预测对应的文本，由于一张图片对应的文本描述具有多样性，从而会导致模型训练效率较低。对比之下，如果只考虑图片和文本是否匹配，这种对比目标函数可以将约束放松，提高模型的训练效率。


![](./VLM/clip1.jpg)

1. contrastive pre-trainng: 模型的输入是 $N$ 个配对的图文，图像通过一个 image encoder，文本通过一个 text encoder，对应得到 $N$ 个文本特征和 $N$ 个图像特征。然后通过计算余弦相似度进行对比学习。矩阵对角线都属于正样本，剩余 $(N^2 - N)$ 个都是负样本。

2. creating dataset classifier from text: 考虑到用于预训练的图文中的文本通常是一个句子，因此，在推理的时候会将 label 转换为一个句子，即使用一个 prompt template，然后 fed into text encoder。

3. zero-shot prediction: 对于一张新的图片，通过 image encoder 得到图片特征。所有感兴趣的标签通过 prompt engineering 后会变成句子， fed into 预训练好的 text encoder，会得到相应的文本特征。将图像特征和若干个文本特征计算余弦相似度，然后通过 softmax 得到概率分布，最大概率对应的句子（标签）即为相应的物体。

对于 image encoder， 可以选择 ResNets 或 vision transforemr。对于 text encoder，可以选择 transformer。

伪代码：首先，文本使用 text encoder 进行特征提取，图像使用 image encoder 进行特征提取。然后投射层将不同模态的特征转换为相同纬度的向量，再进行 $l_2$ norm 的标准化处理得到用于对比的两个特征。通过计算余弦相似度得到 logits，和 ground truth 计算交叉熵目标函数得到 loss。对于 clip 而言，正样本都在对角线上，所以通过 labels = np.arange(n) 创建 ground truth。这里分别针对 image 和 text 计算了两个对称的 loss，再计算平均。

![](./VLM/clip2.jpg)

## prompt engineering and ensembling

Prompt: 提示，文本的引导作用。

为什么需要 prompt engineering 和 prompt ensembling?

1. 词具有多义性。一个单词具有多个不同的含义。比如 'remote' 该词可以作为 ‘遥控器’，也具有‘遥远的’含义。如果不结合上下文信息，text encoder 很难抽取正确的特征。

2. 在预训练时，图像匹配的文本通常是一个句子，但是在推理时，文本输入通常是 label 对应的一个单词，此时会出现 distribution gap 的问题。

基于上述问题，作者提出一种 prompt template，即 "A photo of a {label}"。这种方式可以将标签转换为一个句子，避免了 distribution gap 的问题，同时 label 在句中的位置通常表示是个名词，一定程度上解决了多义性问题。

此外，也可以将一些先验信息加入 prompt template 中，比如食物、动物的数据。

prompt ensembing: 使用多个 prompt template，然后将结果综合起来。

## 模型评价
优势：适用于图像文本匹配。可以提前对数据库内的图像、文本进行特征抽取。对于新来的文本或者图像，只需要做一个简单的点乘，具有灵活性和高效性。

缺点：

1. 如果推理的数据集相较于预训练的数据集 out of distribution，Clip的泛化能力也会很差。

2. 从给定的类别中进行判别选择，而不是生成新的输出。

3. ...


# How to Train Really Large Models on Many GPUs?

[Blog](https://lilianweng.github.io/posts/2021-09-25-train-large/)

# ViLT
[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334)

已有 vision-and-language pre-training (VLP) 工作的不足：

1. 抽取图像特征的效率较低，需要的时间远远高于模态融合部分。

2. 使用已预训练好的模型抽取特征，可能泛化能力较弱，不是 end-to-end 形式。

![](./VLM/vilt3.jpg?300x300)

传统的 VLP 框架和 ViLT：

1. 使用特征检测的模型，对于给定的图片，通过 CNN backbone 进行特征提取，然后基于这些特征使用 roi 等抽取属于物体的特征，得到若干个离散物体的特征向量。文本通过 linear embedding 得到文本特征。对于得到的图像序列和文本序列，再进行模态融合。使用目标检测抽取的特征的方式效率较低。

2. 基于grid 特征检测的模型，对于给定的图片，通过 CNN backbone 得到特征图，然后将特征图拉伸得到相应的序列。

3. ViLT，借鉴 ViT 的 patch embedding layer, ViLT 将图像划分为若干 patches，然后通过一个 linear 投影层得到 patch embeddings。文本也是通过一个 linear 投影层得到 word embeddings。最后两个序列都 fed into 一个 transformer。

从时间上看，ViLT 计算高效，参数量更少。基于目标检测的模型性能最好，基于grid 特征检测的模型性能最差，ViLT处于两者中间。在使用较少参数量的前提下，效果也不相上下。

## Taxonomy of Vision-and-Language Models
We propose a taxonomy of vision-and-language models based on two points: (1) whether the two modalities have an even level of expressiveness in terms of dedicated parameters and/or computation; and (2) whether the two modalities interact in a deep network.

![](./VLM/vilt1.jpg)

图中，VE 表示如何抽取 visual embedding，TE 表示如何抽取 text embedding，MI 表示如何进行模态融合 (modality interaction)。

(a) 轻量的文本 encoder，昂贵的视觉 encoder，轻量的 modality interaction。

(b) visual encoder 和 text encoder 的特征提取能力一样，计算量上基本等价，modality interaction 部分使用简单的 dot-product。代表性模型为 CLIP。

(3) text encoder 非常轻量，visual encoder 使用目标检测模型，计算昂贵，modality interaction 也很昂贵。代表性算法为：ViLBERT, UNITER。

(4) 轻量的 text encoder 和 visual encoder，复杂的 modality interaction。代表性算法为 ViLT。

## Modality Interaction Schema
模态融合主要包括两类：

1. single-stream approaches: 将抽取的文本特征序列和图像特征序列进行拼接作为输入。

2. dual-stream approaches: 两个模型分别对两个序列进行处理，提取单一模态信息，然后再进行融合。

本文使用 single-stream approach，避免引入更多的参数。

## Vision-and-Language Transformer

![](./VLM/vilt2.jpg)

对于多模态的 sing -stream approaches，需要将两个模态的序列进行拼接，因此需要一个 modal-type embedding 告诉模型该 token 属于哪个模态。此外，每个模态前面都需要加上 [CLS] 特殊标记符。然后， patch embedding + position embedding + modal-type embedding 作为 transformer encoder 的输入。

本文主要使用了两类 loss，分别是文本距离和语言完形填空部分，具体而言：

1. image text matching loss: 判断对于给定的配对图文，哪个是真的图文对，哪个是假的图文对。

2. word patch alignment loss：计算文本特征输出和图像特征输出之间的距离。

3. masked language modeling loss: 单词重建的 loss。


# ALBEF, Align before Fuse
[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/pdf/2107.07651)

https://blog.salesforceairesearch.com/align-before-fuse/

https://nancyyanyu.github.io/posts/paper-albef/


模型设计方面：
- 在多模态学习中，视觉特征远远大于文本特征，因此需要使用较为强大的视觉模型(big ViT)。

- 此外，模态之间的融合也至关重要，因此模态融合模型也需要足够大（big modality interaction）。

loss 方面：
- Clip 使用的是 image text contrastive (ITC) loss，效果不错，可以采用。

- ViLT使用的是 Image Text Matching的loss（ITM），word patch alignment (WPA) loss，文本的单词和 image 的 patch 之间进行对应关系。但是 WPA loss 计算非常慢，因此不考虑。

- Bert中常用的计算 loss 方式是 Mask Language Modeling (MLM)，mask 掉某个词然后再去预测这个词。比较常用。

总之，直观上来说，结合 ITC、MLM 和 ITM 的 loss 应该效果不错。

结合上面的考虑，ALBEF 使用复杂的 image encoder (12 blocks) 和 multimodal encoder (6 blocks) ，相对轻量的 text encoder (6 blocks)。并且，考虑使用 image-text contrastive loss (ITC) 对 image embedding 和 text embedding 进行对齐，最后还使用了 ITM 和 MLM loss。

## Introduction
已有的工作使用同一个 transformer-based multimodal encoder 去同时 model visual tokens 和 word tokens。并且对于 visual tokens 还是基于目标检测的模型 (region-based image features)。由于使用的 visual encoder 提取器是基于**预训练**的目标检测器，而不是 end-to-end 训练得到的，这种方式得到的 visual tokens 和 word tokens 并不匹配，从而使得模态融合 encoder 训练困难。(注意，ALBEF 和 ViLT 都想丢弃使用目标检测器的 encoder，但是二者出发点不同， ViLT 是从提升计算效率角度出发。)

因此，本文贡献： 
1. 提出一种对比学习的 loss，对 image 和 text 在 fusing 之前进行对齐。
2. 针对 noisy web data，提出 momentum distillation，通过生成伪标签达到自训练。noisy web data: 从网络上获取的图像-文本是具有噪音的。比如从搜索引擎获取的图文，文本中包含搜索引擎需要的关键词，但是文本并没有对图像进行很好的描述。

## Model Architecture

![](./VLM/albef1.jpg)

image: 对于给定图像，将其划分为若干个 patches，然后 fed into 一个 12 层的 vision transformer encoder。

原始 bert 有 12 层，考虑到 image encoder 要比文本模型大，用于融合的 multimodal encoder 也是越复杂越好。这里将 bert model 进行拆分，维持原始计算参数量。
text: bert model 的前六层用作 text encoder。

融合部分: bert 的后六层用作 multimodal encoder。

除了 ViT 和 BERT 模型外，还有一个 momentum model。该模型也包含了 ViT 和 BERT 模型，只不过是将左侧模型的参数进行移动平均得到，用来产生更多的 负样本对 以及 momentum distilation。

目标函数：ITC loss

# VLMO 
使用一个共享的 self-attention 层，然后使用不同的 feed forward 层去学习不同的模态特征。

# BLIP 
[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/pdf/2201.12086)

## Model Architecture
![](./VLM/blip1.jpg)

整体上看，对于图像部分，有一个 $N$ 层的 ViT。对于文本部分，分别使用三个 text encoder 去计算三个不同的目标函数。在 blip 中，同种颜色代表同样的共享参数，

对于第一个 text encoder，具有 $N$ 层，主要是将文本特征和图像特征进行对比学习，计算 ITC loss 来进行分类任务。第二个 image-grounded text encoder。提取得到的图像特征通过 cross attention 进入模型，文本特征通过 self-attention 得到，然后进行融合得到多模态的特征，计算 ITM loss 来判断 image-text pairs 是否匹配。相较于第一个 text encoder，第二个 encoder 只需要学习额外的 cross attention 层。为了能够执行 生成式 任务，blip 添加了一个 decoder。由于 decoder 不能看到完整的句子，因此将 causal self-attention 替换掉前面 encoder 中的 bi self-attention，不过 cross-attention 和 feed forward 层依旧和前面的共享参数。这里 decoder 使用的 language modeling loss (LM loss)，根据前面的文本去预测后面的文本，而不是进行文本的完形填空 (i.e., MLM loss)。

不同 text encoder 使用不同的 token，分别是 [CLS]，[Encode]，[Decode]。

# BLIP2
![](./VLM/blip2.jpg)

## CapFilt
对于从网络上获取的图文数据质量比较糟糕，图片对应的文本描述往往不够准确。针对这种情况，blip finetune 了一个 filter 来筛选图文对，一个 captioner 来生成合成的文本。$\{I_w, T_w\}$ 是从 web 上获取的 noisy image-text pairs，$\{I_h, T_h\}$ 是人工标注的 image-text pairs，通常认为是高质量的。对于预训练好的 blip 模型，首先基于 $\{I_h, T_h\}$数据对两个预训练好的 text encoder 进行 finetune 得到 filter model。然后对 noisy data $\{I_w, T_w\}$ 进行筛选。同时，基于 $\{I_h, T_h\}$数据对预训练好的 decoder 进行 finetune，用来生成合成的 caption。由于生成的 caption 质量并不确定，因此将 $\{I_w, T_s\}$ 再通过 filter 进行筛选。最终得到数据 $D = \{I_w, T_w\} + \{I_w, T_s\} + \{I_h, T_h\}$。

## Q-Former
在BLIP-2框架中，Q-Former是一个可训练的模块，旨在连接冻结的图像编码器和LLM，实现视觉和语言模态的融合。Q-Former利用可学习的查询嵌入来帮助图像Transformer提取特征。查询嵌入通过自注意力层和交叉注意力层与冻结的图像特征交互。

Q-Former包含两个Transformer子模块：

1. **图像Transformer**：该子模块用于与冻结的图像编码器交互，主要负责提取视觉特征。通过与图像编码器的交互，图像Transformer能获取并聚焦于与文本模态相关的视觉信息。

2. **文本Transformer**：文本Transformer既可以作为文本编码器，也可以作为文本解码器来处理和生成文本信息。在BLIP-2架构中，文本Transformer帮助实现视觉和文本特征的对齐。

可学习的查询嵌入作用：Q-Former使用一定数量的 learnable query embeddings，用于在图像和文本模态间实现特征提取和交互。每个查询嵌入在自注意力和交叉注意力层中与其他特征交互，以提取最相关的信息。

**交互方式**

1. 自注意力层：查询嵌入之间通过自注意力层进行相互作用，使它们能够彼此整合信息。

2. 交叉注意力层：查询嵌入与冻结的图像特征通过交叉注意力层交互。交叉注意力层以隔一个Transformer块的方式插入，帮助模型在视觉模态中专注于与文本相关的特征。

3. 文本交互：这些查询嵌入还可以通过相同的自注意力层与文本信息交互，从而实现多模态信息的整合。

Q-Former基于BERTbase的预训练权重进行初始化，但其交叉注意力层则随机初始化。Q-Former包含1.88亿个参数，采用32个查询嵌入，每个查询的维度为768。通过这种设计，Q-Former的输出查询表示比原始图像特征小得多，使架构能够专注于提取与文本最相关的视觉信息，从而提高模型的效率和对齐能力。


# CONCH

[A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4)

论文介绍了一种视觉-语言基础通用模型-CONCH，利用不同来源的组织病理学图像、生物医学文本和超过 117 万个图像标题对等数据，通过任务识别进行预训练。CONCH 以最先进的视觉语言基础预训练框架 CoCa 为基础，使用一个图像编码器、一个文本编码器和一个多模态融合解码器，并结合使用对齐目标函数和标题目标函数进行训练。其中，对齐目标损失的目的是在模型的表征空间中对齐图像和文本模态，而标题目标则是学习预测与图像相对应的标题。论文总共使用 14 种不同的基准数据集，研究了 CONCH 在一系列任务中的能力，包括图像分类、图像到文本和文本到图像检索、图像分割和图像标题生成。

## 数据处理
为便于整理，论文将数据源分为两类：（1）EDU，包含从教育笔记中提取的数据；（2）PMC OA，从 PubMed Central Open Access Datase 下载的数据。

数据整理的挑战有两个：
1. 筛选组织病理学图像：下载的原始数据包含了组织病理学和非组织病理学的图像。

2. 处理图像面板：大量数据以图像面板的形式呈现，面板中的图像由多个子图像组成，图像标题文本中有时同时或分别包含了多个子图像的描述。

为了应对这些挑战，数据清理分为三个步骤：
1. 检测组织病理学图像：使用YOLOv5对象检测模型生成边界框 bounding boxes 以提取检测到的图像。这一步之前，作者首先通过生成合成数据来训练该对象检测模型。

2. 分割图像标题：作者在整理 EDU 数据集时收集了一个包含图像标题说明和拆分后标题说明的数据集，对GPT模型进行微调，原始图像标题为输入，拆分后的图像标题为输出，最终使得模型具有实现自动拆分图像标题的能力。

3. 子图像和标题进行对齐：首先在干净的EDU数据集上训练一个CLIP模型，将检测到的子图像与拆分后的标题说明进行对齐。使用训练后的模型，给定一组图像面板中的 $m$ 幅检测到的图像和 $n$ 个分割的文本，得到模型中的图像嵌入表征 ${u0, u1, ..., um}$ 和文本嵌入表征 ${v0, v1, ..., vn}$ 。然后两两计算余弦相似度，将相似度最高的作为一对图文数据。

通过以上三步以及进一步数据清理，形成了一个包含$117$万对人类组织病理学图像-说明的数据集。

## Visual-language pretraining

在训练过程中，作者同时考虑了两种loss，一种是图文对比损失（image-to-text and text-to-image contrastive loss)，另一种是针对标题的loss。

![](./VLM/conch1.jpg)

模型框架：模型包括一个 image encoder $f(\cdot; \theta)$，一个 text encoder $g(\cdot; \phi)$ 和一个图文融合的 decoder $h(\cdot; \psi)$。

Image encoder 包含了一个 backbone (参数为 $\theta_{\text{backbone}}$) 和 两个 attention pooler 模块，参数分别为  $\theta_{\text{contrast}}$ 和  $\theta_{\text{caption}}$。 Backbone 使用的是标准的 ViT，具有12层的 transformer层，12 个 attention heads，embedding 的维度为 $768$，hidden dimension 是 $3072$。Image 划分为 $16 \times 16$ 个 image tokens （$256$个），并在每个token上面添加可学习的绝对位置编码。ViT 将RGB图像转换为 feature maps. 基于从 ViT最后一层输出的image token 的特征表示 （其实也是输入decoder cross-attention 中的 quey)，每一个 attention pooler 从不同数量的 image tokens 上去学习相应的信息。具体来说，第一个 attention pooler $f_{\text{contrast}(\cdot; \theta_{\text{contrast}})}$ 使用一个 query 去学习一个 image token，用于捕捉image的全局特征。第二个 attention pooler $f_{\text{caption}(\cdot; \theta_{\text{caption}})}$ 使用 $n = 256$ 个 queries 去生成 $256$ 个 image tokens，用于获取 image的细颗粒度的局部信息，进而生成 caption。

Text encoder 和 multimodal decoder 分别包括 12个 transformer layers，embedding dimension 为 768，hidden dimension 为 3072。Text encoder 通过嵌入表将离散的单词token映射为连续的嵌入向量，并添加了可学习的绝对位置embeddings。Text encoder 为每个tokenized的 caption 添加了一个\<CLS\> token，用于在Transformer注意力过程中提取文本说明的全局表征。

Multimodal decoder 在每个多头自注意力层之后插入了交叉注意力层，以整合来自图像token的信息。最后结合语言模型输出预测下一个token在支持的词汇表中的分布。

假设有 $M$ 个 image-caption 图文对 $(x_i, w_i)_{i=1}^{M}$，其中caption $ w_i = (<BOS>, w_{i,1}, ..., w_{i,T}, <EOS>)$ 包含 $T$ 个 word tokens。对于一个图文对 $(x_i, w_i)$，假设通过 $f_{\text{contrast}(\cdot; \theta_{\text{contrast}})}$ 得到的输出是 $u_i$，通过 text encoder $g(\cdot;\phi)$ 在 \<CLS\> token 处经过 l2 normalization 后得到的输出是 $v_i$，那么loss是：

![](./VLM/itc.jpg)

其中前两项为 image-to-text and text-to-image contrastive loss, respectively, to maximize the cosine-similarity scores between paired image and text embeddings relative to remaining negative pairings in the mini-batch. The last term seeks to maximize the log-likelihood of each observed token under the multimodal autoregressive language model ( jointly parameterized by the image encoder, text encoder and multimodal decoder), conditioned on previous tokens in the caption, as well as the corresponding image. 

具体训练设置，主要包括以下几点：

- 训练轮数：每个视觉-语言预训练实验都运行了 40 个epoch。

- 硬件配置：实验分布式运行在 8个NVIDIA A100 80-GB GPU 上，每个GPU上的本地批量大小为48。

- 梯度累积：为了达到更大的有效全局批量大小，使用了 梯度累积，实现了 1,536 的全局批量大小（48 × 8 GPU × 4次梯度累积）。

- 图像大小：输入图像大小为 448 × 448像素，其中：对较大的图像，首先将其较短边调整为448像素，并对其进行中心裁剪。对较小的图像，按需进行 零填充 以达到所需的尺寸。


# Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos

## Abstract
组织病理学诊断需要对整张切片图像（WSI）进行全局分析，这就要求病理学家从不同的 WSI patch 中复合信息。然而，高分辨率的 WSI 对组织病理学多模式模型提出了挑战。训练组织病理学多模态模型需要用于微调的数据集，而目前的数据集包含单个图像patch的信息，没有每个 patch 间的空间概念，也没有更广泛的 WSI 视图。为了弥补这一不足，本文推出了 QUILT- INSTRUCT，这是一个包含 107,131 个组织病理学特定指令问/答对的大型数据集，这些指令以构成 WSI 的诊断相关图像patch为基础。数据集是利用 YouTube 上的组织病理学教育视频收集的，该视频通过自动提取叙述者的光标位置来提供叙述的局部定位。QUILT-INSTRUCT 支持上下文推理，从整个 WSI 中提取诊断和支持事实。利用 QUILT-INSTRUCT，我们训练出了 QUILT-LLAVA，它的推理能力超越了给定的单个图像patch，能够跨patch进行诊断推理。为了评估 QUILT-LLAVA，我们提出了一个全面的评估数据集，该数据集由 985 幅图像和 1283 个人工生成的问题解答组成。我们还使用公开的组织病理学数据集对 QUILT-LLAVA 进行了全面评估，结果显示 QUILT-LLAVA 在相对 GPT-4 分数上明显优于 SOTA $10%$ 以上，在开放集和封闭集 VQA 上分别优于 SOTA $4%$ 和 $9%$。

## QUILT- INSTRUCT 数据集构建
从 4149 个 YouTube 教育视频中构建了 QUILT-INSTRUCT，总时长超过 1000 小时。这些视频是最近组织病理学数据集 QUILT 的一部分。

在教育视频中，专家在讲述高分辨率 WSI 时往往会停顿一下，然后再用光标指示重点突出区域。我们通过三个步骤将非结构化视频转换为可用的视觉教学数据：首先，我们在视频中定位叙述者的光标。然后，对光标的位置进行时空聚类，以便在图像中将组织病理学概念视觉化。最后，利用提取的标题，使用 LLM 生成指令调整数据集 - QUILT-INSTRUCT。这一过程涉及提示（prompting）技术，从为每个图像patch 生成不同 Q/A 对的独立提示，到结合 WSI 中各patch信息的基于推理的提示，从而生成推理诊断的 Q/A 对。

![](./VLM/quilt1.jpg)

论文介绍了两种不同类型的问答生成方法，用于处理基于病理学的 Whole Slide Images (WSI，整个切片图像) 的文本生成任务。

- Independent Prompts（独立提示）。这一方法基于单个切片级别（patch-level）的文本输入进行问答生成。切片级别文本是与病理图像的某一小块相关的描述。这种提示生成的问答对是独立于整个图像或视频的上下文，仅依赖于该切片的内容，类似于文献[17]中的对话式和详细描述的生成方式。因为这些提示不依赖于其他信息源，因此被称为“独立提示”（Independent prompts）。

- Reasoning-based Prompts（推理提示）。这种方法利用整个视频中的上下文线索，特别是该视频围绕单个WSI的诊断展开，通过逐步揭示概念和线索。输入不仅包含切片级别的文本，还包括整个WSI的全局信息。因此，模型不仅仅基于当前的切片，还可以考虑到全局的诊断信息。通过这种方法，模型（如GPT-4）可以超越当前上下文进行推理，但仍然依靠从视频或图像中提取的事实信息，这样可以减少生成内容的虚构或不准确（即减少幻觉现象）。
  
简单来说，独立提示只基于局部信息生成问答，而推理提示则结合了全局的诊断线索，帮助模型更合理地推理并减少错误。

## Training QUILT-LLAVA & evaluating with QUILT-VQA
论文使用 QUILT-INSTRUCT 来训练 QUILT-LLAVA。在 QUILT-INSTRUCT 之外单独设计 QUILT-VQA，以评估 QUILT-LLAVA。最后，从 QUILT-VQA 中生成指令遵循测试集，以评估 QUILT-LLAVA 的指令遵循能力。

### Training QUILT-LLAVA

LLAVA 由一个vision 模块、多层感知机（MLP）和大语言模型（LLM）组成。这个设计允许语言模型处理视觉信息。
1. 首先，MLP 最初作为一个投影器被训练，直到收敛。在这个阶段，LLM 和视觉模块都被冻结，不会更新权重。
2. 随后，MLP 和 LLM 都会结合指令跟随数据进行微调，使模型与人类病理学家的诊断过程保持一致。

LLAVA 使用预训练的 CLIP 图像编码器，但在这个特定领域中，使用了在公共病理学数据集（如 QUILT-NET [9] 和 PLIP [8]）上训练的预训练 CLIP 模型。作者还通过不同的图像编码器、训练策略和视觉提示进行消融实验，以测试其效果。

![](./VLM/quilt2.jpg)


具体来说，
- 对齐视觉和语言模型。作者首先在病理学领域内对视觉和语言模型进行对齐。为此，从 QUILT 数据集中提取了 723K 图像-文本对，并将描述文本转换为问答格式。问答对的生成方法是随机选择一个预定义的问题（见附录图18），将其添加到图像描述前，形成问答对。问题设计用于描述图像中可见的视觉信息。在这一阶段，视觉和语言模型被冻结，仅训练 MLP 层，其任务是将来自图像编码器的嵌入映射到语言模型中，以便语言模型根据问题预测图像的描述。这一阶段的训练将病理学图像的嵌入与相应的文本嵌入对齐，确保了视觉信息能够被语言模型处理。

- 组织病理学数据集指令微调。论文使用 QUILT-INSTRUCT 对模型进行微调。在此阶段，冻结视觉编码器权重，继续训练 MLP 层和语言模块。

### Evaluation Data Generation: QUILT-VQA
在组织病理学领域，研究人员依靠 PathVQA [7] 和 PMC-VQA [31] 等评估数据集来评估其模型的性能。然而，这些数据集表现出明显的缺点，包括由于转述相同的问题而造成的严重重复。更糟糕的是，同一个问题经常会有相互矛盾的答案（见附录第 3.4 节）。相比之下，教育视频内容提供了一种宝贵的资源：解说员在解说过程中经常提出问题，然后自己给出答案，从而引入了互动元素。例如，解说员会说："你知道我们面对的是哪种器官吗？"然后接着详细说明："是的，这是一个结肠"。视频中的这种问答形式提供了丰富的有机问答数据集，可以提取并重新用于评估。

PathVQA: 包含从教科书和数字图书馆中的 4998 个病理图像标题对中提取的 32799 个问题-答案对。问题分为开放式问题和封闭式问题，前者包括 "什么"、"哪里"、"何时"、"谁的"、"如何"、"多少 "等问题，后者包括 "是"/"否 "等问题。我们使用了评估集中的 6761 个样本。

PMC-VQA: 包含一个由 34823 对图像组成的 VQA 测试集，这些图像涵盖各种模式或病症。该数据集是从 PMC-OA 文章中的图像标题对中整理出来的，采用多选格式。论文从该数据集中检索到 PMC-VQA 子集，其中包括 2318 对组织病理学 VQA 对。

QUILT-VQA: 首先，视频的文字转录内容会被处理，识别出问号 ("?") 所在的位置。
如果问号出现在某个稳定文本块（视频中关联图像的描述文本）45秒的时间范围内，作者将扩展该文本块，确保其包含带有问号的完整句子。这种方法确保了问题和视频中的视觉内容相匹配。在数据预处理和问号映射完成后，作者使用GPT-4直接从文本中提取问答对。
具体地，GPT-4的输入是经过处理的稳定文本块以及其中带有问号的句子，表明这些句子包含提问内容。在GPT-4初步提取完问答对后，作者进行了手动验证，以确保每个问答对不仅在医学上具有相关性，还与该文本块的内容紧密对应。

在提取完成后，作者将问题分为两类：依赖图像的问答对（Image-dependent Q/A pairs）：共1055对，这类问题引用了讲述者对特定图像的描述。基于一般医学知识的问答对（General-knowledge Q/A pairs）：共228对，这类问题与更广泛的医学知识相关，而不仅仅依赖于某一特定图像。

![](./VLM/quilt3.jpg)

### Evaluation data generation: Instruction Following Test Set
QUILT-VQA 的重点是评估 QUILT-LLAVA 的医学知识，除此之外，我们还旨在评估该模型在多模态对话中遵循指令的能力。为此，我们构建了一组 326 个问题，其中包括 256 个会话问题和 70 个详细描述问题，所有问题均来自 QUILT-VQA 中从未曾看过的视频中提取的图像-文本对话。为了生成这个评估集，我们采用了与生成 QUILT-INSTRUCT 时相同的基于会话和详细描述的提示。

## Experiments
本节将介绍 QUILT-LLAVA 在组织病理学 VQA 基准测试中与现有 SOTA 多模态模型的性能对比情况。首先，我们使用 GPT-4 对生成结果与真实答案进行了比对。其次，执行开放式和封闭式 VQA 任务。最后，使用视觉提示和不同的训练模型进行消融实验。

- 使用GPT-4 评估生成结果。评估的主要维度包括回答的帮助性、相关性、准确性和详细程度。评估方法：使用 GPT-4 来对比不同模型的输出：候选模型（QUILT-LLAVA） 和 GPT-4。GPT-4 会根据这些维度（帮助性、相关性、准确性和详细程度）对两个模型的回答进行评分，评分范围为1到10分，分数越高表示整体表现越好。除了分数外，GPT-4 还提供详细的解释，以帮助理解每个模型在生成回答时的表现，便于更好地分析模型的优势和不足。

- 可视化问题解答结果。这些VQA数据集包括开放式和封闭式问答题对。对于封闭式问题，准确率被用来衡量模型给出的正确答案的比例。与此相反，对于开放式问题，我们侧重于召回率，以评估模型的回答中包含真实tokens的频率。

## Conclusion and Limitations
GPT-4 仍然容易生成不准确的信息，导致 QUILT-LLAVA 产生错误陈述或“幻觉”现象（即模型生成的信息与真实内容不符）。此外，尽管我们明确指示 GPT-4 不要这样做，但 GPT-4 有时还是会只从标题中获取信息，而不是从图像中提取信息。

<!-- 
# xray-pulse
## 维度变换
- 图像输入：图像大小为[3, 224, 224].
- Image encoder: Image 划分为 $16 \times 16$ 个 image tokens（$256$个），并在每个token上面添加可学习的绝对位置编码，大小为[1, 257, 768].
- projector:  输出 [1, 197, 1408]
- q-former：输入[1, 197, 1408]， 输出： [1, 32, 768]； num_query_token=32
- projector: 
- pulse:  1, 32, 4096 -->


# Reference

- [Distilled AI](https://aman.ai/primers/ai/)
- [沐神, 论文精读](https://github.com/mli/paper-reading)
- [BLIP系列文章小结](http://www.myhz0606.com/article/blip_hub)











