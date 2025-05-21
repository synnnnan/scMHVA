# scMHVA:Dynamic Synthesis of Multi-Modal Representations for CITE-seq Data Integration and Analysis

# Contents
- [Overview](#overview)
- [Architecture](#Architecture)
- [Installation](#Installation)
- [Data availability](#Data-availability)
- [Usage](#Usage)
- [Key Functions](#Key-Functions)
- [Results](#Results)
- [Contact](#Contact)

# Overview
Single-cell multi-omics technologies are pivotal for unraveling the complexities of biological systems, with CITE-seq (Cellular Indexing of Transcriptomes and Epitopes by Sequencing) standing out as particularly beneficial. The dual-modality capability makes CITE-seq particularly advantageous for dissecting cellular heterogeneity and understanding the dynamic interplay between transcriptomic and proteomic landscapes. However, despite the vast potential of CITE-seq for analyzing cellular heterogeneity, current computational models often fail to capture the complex interactions between RNA and antibody-derived tag (ADT) modalities, resulting in suboptimal cell clustering and limiting the utility in revealing cellular functions and states. To address these issues, we propose scMHVA (Single-Cell multi-omics Multi-head Attention-based Variational Autoencoder), designed to effectively integrate the different modalities of CITE-seq data for better clustering, batch effect correction, visualization, and downstream analysis. scMHVA effectively integrates CITE-seq data by learning the common embedding from both RNA and ADT modalities, utilizing a weighted mean fusion module to merge their latent representations. After that, scMHVA applies a multi-head self-attention mechanism to the joint representation to enhance inter-modality correlations, capturing the complex relationships within CITE-seq datasets. Extensive experiments demonstrated that scMHVA outperformed existing single-modal and multi-modal clustering methods across multiple CITE-seq datasets, with linear scalability in runtime and effective batch effect elimination, establishing it as a powerful tool for analyzing large-scale CITE-seq data. Additionally, we demonstrated that scMHVA successfully annotated different cell types in a published mouse thymocyte dataset and revealed dynamics of immune cell development.

# Architecture
![frame](https://github.com/synnnnan/scMHVA/blob/main/frame.png)

# Installation
### Requirements
```
[python 3.8+]
[pytorch 1.13.1]
[scikit-learn 1.3.2]
[scanpy 1.9.6]
[scipy 1.10.1]
[pandas 1.5.3]
[anndata 0.8.0]
```
For specific setting, please see <a href="https://github.com/synnnnan/scMHVA/blob/main/requirements.txt">requirements</a>.
### Installation
```
$ conda create -n scMHVA_env python=3.8.18
$ conda activate scMHVA_env
$ pip install -r requirements.txt
```
# Data availability
The real multi-omics CITE-seq datasets are freely available at [data](https://zenodo.org/records/15450184).

# Usage
Running example and parameter setting can be found at <a href="https://github.com/synnnnan/scMHVA//blob/main/tutorial.ipynb">tutorial.ipynb</a>.

# Key Functions
 
The key functions of the source code and their detailed description.

| Function      | Description                                                |
| ------------  | ---------------------------------------------------------  |
| preprocess.py | Function of the data preprocessing module of scMHVA        |
| fusion.py     | Function of the fusion module of scMHVA                    |
| model.py      | Fine-Grained Learning for Cellular Embedding Representation|
| train.py      | the model training of scMHVA                               |
| util.py       | the utility functions of the network                       |
| tutorial.ipynb| a example of scMHVA                                        |

# Results

Multiple comparison experiments were conducted on twelve CITE-seq datasets from different sequencing
platforms using a variety of clustering metrics. The experimental results indicated that scMHVA
outperforms other state-of-the-art methods on these datasets, and also demonstrated scalability to large
scale CITE-seq datasets. 

# Contact
If you have any suggestions or questions, please email me at ynshi24@mails.jlu.edu.cn.

# License
This project is covered under the **MIT License**.
