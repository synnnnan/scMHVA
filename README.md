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
Single‐cell multi‐omics technologies are pivotal for deciphering the complexities of biological systems, with Cellular Indexing of Transcriptomes and Epitopes by Sequencing (CITE‐seq) emerging as a particularly valuable approach. The dual‐modality capability makes CITE‐seq particularly advantageous for dissecting cellular heterogeneity and understanding the dynamic interplay between transcriptomic and proteomic landscapes. However, existing computational models for integrating these two modalities often struggle to capture the complex, non‐linear interactions between RNA and antibody‐derived tags (ADTs), and are computationally intensive. To address these issues, scMHVA, a novel and lightweight framework designed to integrate the diverse modalities of CITE‐seq data, is proposed. scMHVA utilizes an adaptive dynamic synthesis module to generate consolidated yet heterogeneous embeddings from RNA and ADT modalities. Subsequently, scMHVA enhances inter‐modality correlations within the joint representation by applying a multi‐head self‐attention mechanism, effectively capturing the intricate mapping relationships between mRNA expression levels and protein abundance. Extensive experiments demonstrate that scMHVA consistently outperformed existing single‐modal and multi‐modal clustering methods across CITE‐seq datasets of varying scales, exhibiting linear runtime scalability and effectively eliminating batch effects, thereby establishing it as a robust tool for large‐scale CITE‐seq data analysis. Additionally, it is demonstrated that scMHVA successfully annotates different cell types in a published mouse thymocyte dataset and reveals dynamics of immune cell development.

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
