<div align="center">

<img src="docs/assets/signal-banner.svg" alt="SIGNAL — Synergistic Multi-Granularity Semantic Alignment Learning for Multimodal Recommendation" width="100%">

### 面向多模态推荐的协同多粒度语义对齐学习

<sub>Expert Systems with Applications · Volume 333 · 2027 · Article 134101</sub>

Meizi Li<sup>†</sup>, Hang Gao<sup>†</sup>, Bo Zhang, Yang Yu, Guojian Zou<sup>*</sup>, Ling Ding<sup>*</sup>

<sup>†</sup> 共同第一作者 &nbsp; <sup>*</sup> 通讯作者

**[阅读论文](https://doi.org/10.1016/j.eswa.2026.134101)** &nbsp; / &nbsp; **[PDF 全文](docs/paper/SIGNAL.pdf)** &nbsp; / &nbsp; **[模型代码](src/models/signal.py)** &nbsp; / &nbsp; **[English](README.md)**

</div>

---

<p align="center">
  <a href="#项目简介">项目简介</a> &nbsp; · &nbsp; <a href="#模型框架">模型框架</a> &nbsp; · &nbsp; <a href="#主要结果">主要结果</a> &nbsp; · &nbsp; <a href="#环境与运行入口">运行入口</a> &nbsp; · &nbsp; <a href="#引用">引用</a>
</p>

## 项目简介

本仓库提供 **SIGNAL** 的模型实现，训练与评估基础设施基于 [MMRec](https://github.com/enoche/MMRec)。SIGNAL 将图编码、对比对齐与轻量级注意力扩散结合起来，同时学习整体和局部粒度的多模态语义。

独立的图编码与对比对齐容易破坏语义邻域，也难以消除模态间细粒度的语义冲突。SIGNAL 通过编码与对齐的协同设计保留语义结构，并用扩散模块进一步细化局部语义。

## 模型框架

<p align="center">
  <img src="docs/assets/framework.png" alt="SIGNAL：预对齐图编码、层次相似性对比与轻量级注意力扩散" width="100%">
</p>

*论文图 2：PGE 与 HSC 协同完成整体语义对齐，LAD 进一步协调局部语义冲突。*

<table>
<tr>
<td width="33%" valign="top">
<sub><b>01 / 编码</b></sub>
<h3>PGE</h3>
<b>预对齐图编码</b>
<p>通过图间注意力融合不同模态的图结构，在编码阶段建立初步语义对齐。</p>
</td>
<td width="33%" valign="top">
<sub><b>02 / 对齐</b></sub>
<h3>HSC</h3>
<b>层次相似性对比</b>
<p>结合基础与差异化对比学习，在对齐表征的同时保留图编码形成的语义邻域。</p>
</td>
<td width="33%" valign="top">
<sub><b>03 / 细化</b></sub>
<h3>LAD</h3>
<b>轻量级注意力扩散</b>
<p>借助可靠性感知引导，以少量恢复步骤细化跨模态语义冲突。</p>
</td>
</tr>
</table>

其中，共现图是根据**训练集交互**构建的行为视图，用于补充视觉和文本图的语义信息。

## 主要结果

<img src="docs/assets/results-highlights.svg" alt="论文报告的 Recall@20 提升：Baby +5.54%，Sports +8.37%，Clothing +5.10%，TikTok +6.10%" width="100%">

以下为论文 **Table 4** 中固定随机种子实验的结果。R@K 表示 Recall@K，N@K 表示 NDCG@K；提升幅度针对每个数据集上 R@20 最好的非 SIGNAL 基线。

| 数据集 | R@10 | R@20 | N@10 | N@20 | R@20 提升 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baby | 0.0679 | **0.1047** | 0.0368 | **0.0463** | +5.54% |
| Sports | 0.0783 | **0.1180** | 0.0424 | **0.0526** | +8.37% |
| Clothing | 0.0674 | **0.0989** | 0.0367 | **0.0446** | +5.10% |
| TikTok | 0.0772 | **0.1165** | 0.0408 | **0.0506** | +6.10% |

论文 Table 5 另提供四个随机种子的稳定性分析，完整消融、协同机制分析与鲁棒性实验见[论文](docs/paper/SIGNAL.pdf)。

## 环境与运行入口

```bash
git clone https://github.com/Sail-high/SIGNAL.git
cd SIGNAL
conda env create -f environment.yml
conda activate signal
```

环境文件沿用原仓库的 PyTorch 1.11 依赖基线，使用 Python 3.9。当前实现包含 CUDA 专用操作；论文实验使用 48 GB 显存的 NVIDIA RTX A40。

数据格式与共现图说明见[数据准备](data/README.md)。训练入口通过参数指定模型和数据集：

```bash
cd src
python main.py --model SIGNAL --dataset baby
```

必须从 `src/` 目录启动，因为配置和数据路径相对于当前工作目录解析。论文超参数与配置规则见[实验设置](docs/REPRODUCIBILITY.md)。

## 代码导航

| 路径 | 内容 |
| :--- | :--- |
| [src/models/signal.py](src/models/signal.py) | SIGNAL 与 LAD 实现 |
| [src/main.py](src/main.py) | 训练入口，默认模型为 SIGNAL |
| [src/configs](src/configs) | 现有框架与基线配置 |
| [data](data) | 数据准备说明 |
| [preprocessing](preprocessing) | 继承自 MMRec 的预处理工具 |
| [docs/paper/SIGNAL.pdf](docs/paper/SIGNAL.pdf) | 论文全文 |
| [docs/upstream](docs/upstream) | 原 MMRec 说明及图示归档 |

## 引用

```bibtex
@article{li2027signal,
  title   = {{SIGNAL}: Synergistic multi-granularity semantic alignment learning for multimodal recommendation},
  author  = {Li, Meizi and Gao, Hang and Zhang, Bo and Yu, Yang and Zou, Guojian and Ding, Ling},
  journal = {Expert Systems with Applications},
  volume  = {333},
  pages   = {134101},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.134101}
}
```

引用年份按正式卷期使用 **2027**；论文在线发表日期为 **2026 年 8 月 19 日**。

## 致谢与许可

感谢 [MMRec](https://github.com/enoche/MMRec) 提供训练、数据加载、评估与基线实现。本仓库保留原始作者署名及 [GPL v3 许可证](LICENSE)，框架引用见[致谢说明](ACKNOWLEDGEMENTS.md)。论文及图示保留各自的出版声明。

## 联系方式

如有 SIGNAL 相关问题，请联系 **高航（Hang Gao）**：

[1000566340@smail.shnu.edu.cn](mailto:1000566340@smail.shnu.edu.cn)

也欢迎提交 [GitHub Issue](https://github.com/Sail-high/SIGNAL/issues)。
