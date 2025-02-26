# A redundancy reduction contrastive learning based cancer subtype clustering method using multi-omics data (CLCluster)

Alternative splicing (AS) allows one gene to produce several protein variants, offering valuable predictive insights into cancer and facilitating targeted therapies. Although multi-omics data are used to identify cancer subtypes, AS is rarely utilized for this purpose. Here, we propose a redundancy-reduction contrastive learning-based method (CLCluster) based on copy number variation, methylation, gene expression, miRNA expression, and AS for cancer subtype clustering of 33 cancer types. Ablation experiments emphasize the benefits of using AS data to subtype cancer. We identified 2,921 cancer subtype-related AS events associated with patient survival and conducted multiple analyses including open reading frame annotation, RNA binding protein (RBP)-associated AS regulation, and splicing-related anticancer peptides (ACPs) prediction for therapeutic biomarkers. The CLCluster model is more effective in identifying prognostic-relevant cancer subtypes than other models. The effective annotation of cancer subtype related AS events facilitates the identification of therapeutically targetable biomarkers in patients.
![](./CLCluster.svg)

## Dependencies

CLCluster is a clustering algorithm based on redundancy-reduction contrastive learning. It is used to classify cancer subtypes using multi-omics data of patients as input and providing the subtype of cancer as output. The method performs feature extraction by redundancy-reduction contrastive learning model. For the extracted features, after introducing survival information for further dimensionality reduction, clustering is performed using Mean-Shift to obtain cancer subtypes. To prevent model collapse and not require negative examples or asymmetric structures, we employ a unique loss function for redundancy-reduction contrastive learning.

The model is developed with Python 3.10, Pytorch 2.2.1, and CUDA12.1, for other environments, please refer to `environment.yml`

1. Download and install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

2. Create an python environment and install required packages.

```bash
conda create -n mcluster && conda activate mcluster
bash environment.sh
```
> The environment can also be created by `create env -f enviroment.yml`


## Start
### Cancer Subtype Classification
1. Example data (CESC) is in *test* and can be download to test. If you want to run your own data, please refer to the data preprocessing method in the paper for data preprocessing.

2. Modify the configuration. Run the test file without modifying the configuration. If you run your own data, please modify the cancer name and other information in *yaml*.

```bash
    python main.py
```

This will run CLCluster with the test dataset in `test/`. 

#### Results include:
1. Raw data downscaling results `CESC_features.csv`. The low-dimensional embedding is generated after inputting the original dataset into the contrastive learning module.

![image](https://github.com/user-attachments/assets/90cc64d3-4e22-4c7a-8304-584b18c09c1e)


2. Cancer subtype clustering results `CESC_cluster.csv`. CLCluster divides CESC into four subtypes.

![54fb903f33b2501283fa2bcab62da9b](https://github.com/user-attachments/assets/49097b09-905a-4095-a4b2-aac9b81e1a60)


3. subtype KM images `CESC_KM.png`, subtype TSEN visualization images `CESC_TSEN.png`.
   The KM analysis result of CESC was significant (p=0.0387), which indicates that there are significant survival differences among the four subtypes identified by CLCluster.
   The results of TSNE show that the patients corresponding to the four subtypes identified by CLCluster can be clearly distinguished in terms of spatial characteristics.
![图片2](https://github.com/user-attachments/assets/afece776-6db2-4965-9a7b-aec14d514c02)


The results will be in `out/`.

We provide the preprocessed data and model clustering results of all other cancers in the [data/CLCluster](https://www.synapse.org/Synapse:syn64598517/files/).

### Drug Sensitivity
We provide the drug sensitivity analysis results in [data/Drug Sensitivity](https://www.synapse.org/Synapse:syn64598517/files/). If you want to make your own drug annotation based on the cancer subtype classification results, please refer to the following steps：

1. Using [oncoPredict](https://cran.r-project.org/web/packages/oncoPredict/index.html) to predict the drug sensitivity relationship between samples and 198 drugs based on gene expression data.

2. Using clustering results as grouping features, perform variance analysis, screen the variance analysis results, and screen out drugs with significant differences in drug sensitivity between subtypes when P<0.05

3. Obtaining drug target data from Cancerdrugsdb and complete target annotation of early-onset drugs.



### ACP Prediction
We provide the input and output data for ACP prediction in [data/ACP](https://www.synapse.org/Synapse:syn64598517/files/), and you can directly run the input data using [ACPredStackL](https://github.com/liangxiaoq/ACPredStackL) and [AACFlow](https://github.com/z11code/AACFlow). 

If you want to predict ACP in your own AS markers, you can refer to the following steps:

1. Obtaining cancer hallmark genes from COSMIC and screening for in-frame AS events within the CDS region of these genes.

2. Identifying the corresponding DNA sequences generated by these splicing events using TCGASpliceSeq and using ORFfinder to identify peptides likely to be produced by these DNA sequences, with a specific focus on segments ranging from 10 to 60 amino acids, as this range corresponds to ACPs.

3. Using these peptide sequences as inputs and use [ACPredStackL](https://github.com/liangxiaoq/ACPredStackL) and [AACFlow](https://github.com/z11code/AACFlow) models to predict whether these peptides are ACPs.

