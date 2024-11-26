# scMHVA
# Learning Synthesized Representations of Single-Cell CITE-seq Data Enables Characterization and Interpretation of Cellular Heterogeneity and Regulatory Mechanisms
Single-cell multi-omics technologies are pivotal for unraveling the complexities of biological systems, with CITE-seq (Cellular Indexing of Transcriptomes and Epitopes by Sequencing) standing out as particularly beneficial. With simultaneous quantification of RNA and cell surface proteins at the single-cell level, CITE-seq provides unique, multidimensional insight into cellular states and functions. However, despite the vast potential of CITE-seq for analyzing cellular heterogeneity, current computational models often fail to capture the complex interactions between RNA and antibody-derived tag (ADT) modalities, resulting in suboptimal cell clustering and limiting the utility in revealing cellular functions and states. To address these issues, we propose scMHVA (Single-Cell multi-omics Multi-head Attention-based Variational Autoencoder), designed to effectively integrate the different modalities of CITE-seq data for better clustering, batch effect removal, visualization, and downstream analysis. scMHVA effectively integrates CITE-seq data by learning the common embedding from both RNA and ADT modalities, utilizing a weighted mean fusion module to merge their latent representations. After that, scMHVA applies a multi-head self-attention mechanism to the joint representation to enhance inter-modality correlations, capturing the complex relationships within CITE-seq datasets. Extensive experiments demonstrated that scMHVA outperformed existing single-modal and multi-modal clustering methods across multiple CITE-seq datasets, with linear scalability in runtime and effective batch effect elimination, establishing it as a powerful tool for analyzing large-scale CITE-seq data. Additionally, we demonstrated that scMHVA successfully annotated different cell types in a human peripheral blood mononuclear cell dataset and revealed regulatory mechanisms in the intercellular immune system.

![frame](https://github.com/synnnnan/scMHVA/blob/main/frame.png)

## Installation
### pip
        pip install -r requirements
### usage
Running example and parameter setting can be found at <a href="https://github.com/synnnnan/scMHVA//blob/main/tutorial.ipynb">tutorial.ipynb</a>.
## License
This project is covered under the **MIT License**.
