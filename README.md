# A redundancy reduction contrastive learning based cancer subtype clustering method using multi-omics data (CLCluster)

Here, we propose a redundancy-reduction contrastive learning-based method (CLCluster) based on copy number variation, methylation, gene expression, miRNA expression, and alternative splicing for cancer subtype clustering of 33 cancer types.
CLCluster (shown in Figure 1) is a redundancy-reduction contrastive learning clustering method based on multi-omics datasets for cancer subtyping. The method performs feature extraction by redundancy-reduction contrast learning model. For the extracted features, after introducing survival information for further dimensionality reduction, clustering is performed using Mean-Shift to obtain cancer subtypes. To prevent model collapse and not require negative examples or asymmetric structures, we employ a unique loss function for Redundancy-reduction contrastive learning.
![](./CLCluster.svg)


## Dependencies

The model is developed with Python 3.10, Pytorch 2.2.1, and CUDA12.1, for other environments, please refer to `environment.yml`

1. Download and install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

2. Create an python environment and install required packages.

```bash
conda create -n mcluster && conda activate mcluster
bash environment.sh
```
> The environment can also be created by `create env -f enviroment.yml`

The version of the databases
| Tools | source | Identifier |
| ---- | ---- | ---- |
| TCGA | v35.0 | https://gdc.cancer.gov/ |
| TCGASpliceSeq | 2.1 | https://bioinformatics.mdanderson.org/TCGASpliceSeq/index.jsp |
| GENCODE | V45 | https://www.gencodegenes.org/human/release_45.html |
| RBPDB | v1.3.1 | http://rbpdb.ccbr.utoronto.ca/proteins.php?species_filter=9606 |
| UniProtKB | 2023_03 | https://www.uniprot.org/ |
| COSMIC | v101 | https://cancer.sanger.ac.uk/cosmic/download/cosmic/v101/cancergenecensus |
| ExonSkipDB |  | https://ccsm.uth.edu/ExonSkipDB/ |



## Start
1. Example data (CESC) is in *test* and can be download to test. If you want to run your own data, please refer to the data preprocessing method in the paper for data preprocessing.

2. Modify the configuration. Run the test file without modifying the configuration. If you run your own data, please modify the cancer name and other information in *yaml*.

```bash
    python main.py
```

This will run CLCluster with the test dataset in `test/`. 

Results include raw data downscaling results `CESC_features.csv`, cancer subtype clustering results `CESC_cluster.csv`, subtype KM images `CESC_KM.png`, subtype TSEN visualization images `CESC_TSEN.png`. The results will be in `out/`.We provide the preprocessed data and model clustering results of all other cancers in the [data](https://www.synapse.org/Synapse:syn64598517/files/).

