# TMF-Net: A Dynamic–Static Collaborative Framework for Mechanism-Resolved Age Identification in Medicinal Plants

This repository provides the research resources for “TMF-Net: A Dynamic–Static Collaborative Framework for Growth-Year Identification of Medicinal Plants”, including the model implementation, dataset description, and documentation for the mobile application.

---

## Abstract

Abstract
The growing year is a key determinant of the medicinal and economic value of medicinal materials, yet current identification methods rely on destructive detection and lack efficient, non-destructive, and mechanistically verifiable approaches, constituting a bottleneck for large-scale quality control. Although deep learning has significantly improved recognition accuracy, its modeling still mainly depends on static phenotypic features, and its interpretability mostly remains at the visualization of decision regions, without establishing empirical links to real developmental mechanisms. In this paper, based on a self-constructed multi-phenological dataset of the precious medicinal plant Dendrobium nobile, a dynamic and static synergistic framework, TMF-Net, was proposed. Cross-phenological growth rate differences were converted into computable temporal difference features, and a species-decoupled growth dynamic representation was constructed through fine-grained structural enhancement and adaptive feature fusion. An F1 score of 94.73% was achieved, significantly outperforming mainstream convolutional architectures. The stable generalizability of the proposed method was further validated on a self-constructed cross-genus dataset of the precious medicinal plant Bletilla striata and on cross-crop (banana and tomato) temporal tasks, and a general temporal processing capability that can effectively adapt to different crops and different temporal scenarios was demonstrated. A mobile application developed based on the model was applied to realize plant age identification for medicinal plants, further verifying the deployability and practical feasibility of the method. At the mechanistic level, the stem node regions of D. nobile focused on by the model were highly consistent with the temporal characteristics of progressive lignin deposition at the cellular microscopic level. Integrated transcriptomic and metabolomic analyses further confirmed the coordinated regulation of the lignin biosynthesis pathway, indicating that the discriminative features of the model corresponded to genuine tissue developmental programs rather than accidental phenotypic differences. Thus, a cross-scale mechanistic loop from deep visual representations to molecular regulatory networks was constructed, advancing deep learning models from decision visualization to mechanistic verifiability and providing a generalizable research paradigm for the intelligent analysis of plant growth status.

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

