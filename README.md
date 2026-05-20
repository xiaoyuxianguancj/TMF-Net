# Few-shot transferable deep learning enables growth-year identification and cross-scale multi-omics associations in the precious medicinal orchid *Dendrobium nobile*
This repository provides the research resources for “TMF-Net: A Dynamic–Static Collaborative Framework for Growth-Year Identification of Medicinal Plants”, including the model implementation, dataset description, and documentation for the mobile application.

---

## Abstract

Abstract
The growth year critically determines the medicinal and economic value of medicinal plants, yet its identification remains dependent on destructive assays. Here, we construct a multi-phenological dataset of *Dendrobium nobile* and develop TMF-Net, a dynamic–static synergistic framework that encodes cross-phenological growth-rate differences as temporal contrastive features and constructs species-decoupled developmental representations through fine-grained structural enhancement and adaptive fusion. TMF-Net achieves an F1 score of 94.73%, outperforming representative convolutional architectures. Notably, robust generalization is achieved across genus (*Bletilla striata*) and distinct crop species (banana and tomato), despite substantial morphological divergence and limited sample sizes, indicating transferable modeling capability. Mechanistically, model attention converges on stem nodes undergoing progressive lignification. Integrated transcriptomic and lignin metabolomic analyses further reveal coordinated activation of the lignin biosynthesis pathway, confirming that the learned representations reflect genuine developmental programs rather than incidental phenotypic variation. By linking deep visual features with molecular potential regulatory networks, this work establishes a cross-scale mechanistic framework that advances deep learning from decision visualization to mechanistic verifiability and provides a generalizable paradigm for intelligent assessment of the temporal variation in plant growth status. 

# 📂 Dataset Description

## Self-Constructed Datasets of Two Valuable Medicinal Plant Species
The two self-constructed datasets of medicinal plants supporting the conclusions of this article are available in the Zenodo repository at https://doi.org/10.5281/zenodo.20177266. The data will be made publicly available upon acceptance of the manuscript.
## Cross-Crop Temporal Task

- [Banana RipeNess Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/shahRiaR26s/baNaNa-RipeNess-classificatioN-dataset)
- [Tomato Growth Stage Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/swapNilNaique/tomato-data)
- to evaluate the model’s general temporal modeling capability
---

# 🚀 Download Pretrained Model

Due to the large size of the dataset and model files, they are not hosted directly on GitHub.

- The architecture of TMF-Net is available in the model directory.

---

# 📱 Mobile Application

- A mobile application was developed based on TMF-Net. Please refer to the App directory for details.  

