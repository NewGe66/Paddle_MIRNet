# Learning Enriched Features for Real Image Restoration and Enhancement

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
  - [3.1 准备环境]()
  - [3.2 准备数据]()
  - [3.3 准备模型]()
- [4. 开始使用]()
  - [4.1 模型训练]()
  - [4.2 模型评估]()
  - [4.3 模型预测]()
- [5. 模型推理部署]()
  - [5.1 基于Inference的推理]()
  - [5.2 基于Serving的服务化部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()


**注意：**

(1) 目录可以使用[gh-md-toc](https://github.com/ekalinin/github-markdown-toc)生成；

(2) 示例repo和文档可以参考：[AlexNet_paddle](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/README.md)。

## 1. 简介

在图像采集时，经常会引入不同程度的退化(**degradation**)，这可能是由于相机的物理因素限制，也可能是由于不合适的照明条件。因此常常会产生有噪点(noisy)和低对比度(low-contrast)的图像。近年来，深度学习模型在图像恢复和增强(image restoration and enhancement)方面取得重大进展，因为它可以从大规模数据集中学习到较强的前沿信息。

以从低分辨率(退化)图像中恢复出高质量图像内容为目标，图像恢复(**Image Restoration**)已经在众多领域得到广泛应用。现有的基于CNN的方法通常在全分辨率(**full-resolution**)或渐进式低分辨率(**progressively low-resolution**)上进行：在full-resolution下虽然得到了良好的空间精确度(**spatially precise**)，但是不能获得鲁棒性较好的上下文信息(**context**)；而在progressively low-resolution下，虽然在语义上可靠(**semantically reliable**)/得到好的上下文信息，但是在空间上并不太准确。在本文中，提出了一种新颖的结构，可以通过神经网络保持空间上精确的高分辨率表示；并从低分辨率表示中获取良好的上下文信息(strong contextual information)。

网络的核心是包含几个关键元素的多尺度残差块(**multi-scale residual block**)：

(a)用于提取多尺度特征的并行多分辨率卷积流(**parallel multi-resolution convolution streams**)；

(b)跨多分辨率流的信息交换；

(c)用于捕捉上下文信息的空间和通道注意机制(**spatial and channel attention mechanisms**)；

(d)基于注意机制的多尺度特征聚合(**aggregation**)。

简言之，**MIRNet**学习丰富的特征，结合了多个尺度的上下文信息的同时保持了高分辨率的细节，在图像去噪、超分辨率、图像增强任务上取得了极好的效果。



<center> MIRNet模型图

**测试效果：**

|             原图              |       添加噪声后        | MIRNET去噪后             |
| :---------------------------: | :---------------------: | ------------------------ |
| ![](0001-0011groundtruth.png) | ![](0001-0011input.png) | ![](0001-0011MIRNet.png) |

注意：在给出参考repo的链接之后，建议添加对参考repo的开发者的致谢。

**论文:** [Learning Enriched Features for Real Image Restoration and Enhancement](http://xxx.itp.ac.cn/pdf/2003.06792v2.pdf)

**参考repo:** [https://github.com/swz30/MIRNet](https://github.com/swz30/MIRNet)

在此非常感谢`$参考repo的 github id$`等人贡献的[repo name](url)，提高了本repo复现论文的效率。

**aistudio体验教程:** [地址](url)


## 2. 数据集和复现精度

训练集：[ SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) 采用了5款智能机（Google Pixel、iPhone 7、Samsung Galaxy S6 Edge、Motorola Nexus 6、LG G4）在四种相机参数下拍摄了10个场景，200个场景实例，每个场景连续拍摄了150张图像，总共30000张噪声图像。其中160个场景实例作为训练集，40个场景实例作为测试集（the benchmark）。

测试集：

​	（1）图像去噪：

​		a.DND由4台消费相机捕获的50幅图像组成。由于图像具有很高的分辨率，数据集提供商从每张图像中提取了大小为512 × 512的20种作物，总共产生1000个patch。所有这些patches都用于测试(因为DND不包含训练集或验证集)。由于地面真实无噪声图像没有公开发布，所以图像质量的PSNR和SSIM评分只能通过在线服务器获取。

​		b.SIDD特别地由智能手机摄像头收集。因为传感器小，而且分辨率高，智能手机图像的噪音比数码单反要高得多。

SIDD包含320对图像用于训练，1280对图像用于验证。

​	（2）图像超分：

​		a.RealSR

​	（3）图像增强：

​		a.LOL

​		b.MIT-Adobe FiveK		

|  模型  | 数据集                                                     | PSNR/SSIM (验收精度) | LPIPS/FID (复现精度) |                           下载链接                           |      |      |
| :----: | ---------------------------------------------------------- | :------------------: | :------------------: | :----------------------------------------------------------: | ---- | ---- |
| MIRNet | [ SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) |     39.72/0.959      |                      | [预训练模型](https://paddle-model-ecology.bj.bcebos.com/model/alexnet_reprod/alexnet_pretrained.pdparams) \|  [Inference模型](https://paddle-model-ecology.bj.bcebos.com/model/alexnet_reprod/alexnet_infer.tar) \| [日志](https://paddle-model-ecology.bj.bcebos.com/model/alexnet_reprod/alexnet_train.log) |      |      |

给出本repo中用到的数据集的链接，然后按格式描述数据集大小与数据集格式。

格式如下：

- 数据集大小：关于数据集大小的描述，如类别，数量，图像大小等等
- 数据集下载链接：链接地址
- 数据格式：关于数据集格式的说明

基于上述数据集，给出论文中精度、参考代码的精度、本repo复现的精度、数据集名称、模型下载链接（模型权重和对应的日志文件推荐放在**百度云网盘**中，方便下载）、模型大小，以表格的形式给出。如果超参数有差别，可以在表格中新增一列备注一下。

如果涉及到`轻量化骨干网络验证`，需要新增一列骨干网络的信息。

## 3. 准备数据与环境


### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求，格式如下：

- 硬件：xxx
- 框架：
  - PaddlePaddle >= 2.2.0

然后介绍下怎样安装PaddlePaddle以及对应的requirements。

建议将代码中用到的非python原生的库，都写在requirements.txt中，在安装完PaddlePaddle之后，直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.2 准备数据

简单介绍下全量数据和少量数据分别怎么使用，给出使用命令。


### 3.3 准备模型


可以在此提示用户怎么下载预训练模型、inference模型（如果有）


## 4. 开始使用


### 4.1 模型训练

简单说明一下训练的命令，建议附一些简短的训练日志。

可以简要介绍下可配置的超参数以及配置方法。

### 4.2 模型评估

简单说明一下评估的命令以及结果，建议附一些简短的评估日志。

### 4.3 模型预测


在这里简单说明一下预测的命令，需要提供原始图像、文本等内容，在文档中体现输出结果。


## 5. 模型推理部署

如果repo中包含该功能，可以按照Inference推理、Serving服务化部署再细分各个章节，给出具体的使用方法和说明文档。


## 6. 自动化测试脚本

介绍下tipc的基本使用以及使用链接


## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
