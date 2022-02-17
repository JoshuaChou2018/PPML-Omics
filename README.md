# Overview

This repository contains the implementation of applications with PPML-Omics from:

**Juexiao Zhou, Siyuan Chen, et al. "PPML-Omics: a Privacy-Preserving federated Machine Learning system protects patients’ privacy from omic data"**

![fig1](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/fig1.S2jS3C.png)

**Fig. 1 PPML-Omics: a Privacy-Preserving federated Machine Learning system protects patients’ privacy from omic data** a, Schematic overview of the relationships and interactions between distributed data owners, algorithm owners, attackers and tech- niques in the field of secure and private AI. b, Schematic overview of our PPML-Omics system design, in which the clients train the local model with local private data and the noised gradients with DP mechanism goes through the shuffling mechanism. Then all noised and shuffled updates are sent to the central server and used for updating the central model. c, Illustration of 3 representative tasks, datasets and attacks of omic data in this paper for demonstrating the utility and privacy-preserving capability of our PPML-Omics system, including the 1) cancer classification with bulk RNA-seq, 2) clustering with scRNA-seq and 3) integration of morphology and gene expression with spatial transcriptomics.

# Prerequisites

>  Important packages:
>
>  
>
>  \- python=3.9.7=hb7a2778_3_cpython
>
>  \- pytorch=1.9.1=cuda112py39h4e14dd4_3
>
>  \- torchvision=0.9.0=py39cuda112hc5182df_0_cuda
>
>  \- cudatoolkit=11.5.0=h36ae40a_9
>
>  \- cudnn=8.2.1.32=h86fa8c9_0
>
>  \- numpy=1.20.3=py39hf144106_0
>
>  \- pandas=1.3.2=py39h8c16a72_0
>
>  \- scikit-learn=0.24.2=py39ha9443f7_0
>
>  \- scipy=1.5.3=py39hf3f25e7_0
>
>  \- tqdm=4.62.2=pyhd3eb1b0_1

# Usage

## Files Description

`environment.yml`: creating an environment from the environment.yml file

`/Attack`: source codes for attack experiments

`/Application`: examples of applying PPML-Omics, users can eaisy modify based on it to meet their own requirements.

## Environment Bulding

```
# notice you may need to change the prefix in the environment.yml according to your home path
# several minutes for building the environment
conda env create -f environment.yml
conda activate ppmlomics
```

## Examples

### Application 1: Cancer type classification with PPML-Omics

![image-20220217104849671](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/image-20220217104849671.eWS8R5.png)

#### Root

Please run all codes in this section with ` Application/CancerTypeClassification` as root.

#### Dataset Preparation

The complete dataset can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1ZtVEYm4aB8nPMbrB9gDJ8ZFBede0DqHq?usp=sharing).

```
python 01.processData.py
```

#### Optional Parameters

```
python 02.simulationApp.py --help
```

```
usage: 02.simulationApp.py [-h] [--device DEVICE] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--epsilon EPSILON] [--delta DELTA] [--mode MODE] [--client CLIENT]
                                 [--l2_clip L2_CLIP] [--nprocess NPROCESS] [--expname EXPNAME] [--train_data TRAIN_DATA] [--test_data TEST_DATA] [--shuffle_model SHUFFLE_MODEL]

dfsa

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       default: cuda:0
  --epochs EPOCHS       default: 50
  --batch_size BATCH_SIZE
                        default: 1
  --lr LR               default: 0.01
  --epsilon EPSILON     default: 100
  --delta DELTA         default: 0.5
  --mode MODE           default: SGD, {SGD, SIGNSGD, DP, DPSIGNSGD}
  --client CLIENT       default: 3
  --l2_clip L2_CLIP     default: 5
  --nprocess NPROCESS   default: 20
  --expname EXPNAME     experiment name
  --train_data TRAIN_DATA
                        will load data/{}.npy
  --test_data TEST_DATA
                        will load data/{}.npy
  --shuffle_model SHUFFLE_MODEL
                        0: off, 1: on
```

#### Systems

PPML-Omics provides 4 basic modes for calculate the gradients: SGD, SIGNSGD, DP, DPSIGNSGD, and 1 shuffling mode, with --shuffle_model sets to 1, which can be freely combined into different systems.

##### Example of centrally trained system (#client=1)

```
python 02.simulationApp.py --mode SGD --client 1 --epochs 100 --batch_size 32 --lr 0.001 --expname SGD --train_data train_log10 --test_data test_log10
```

##### Example of FL system with 5 clients

```
python 02.simulationApp.py --mode SGD --client 5 --epochs 100 --batch_size 32 --lr 0.001 --expname SGD --train_data train_log10 --test_data test_log10
```

##### Example of FL+DP system with different privacy budget $\epsilon$

```
python 02.simulationApp.py --mode DP --client 5 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 1 --expname DP_e1 --train_data train_log10 --test_data test_log10
python 02.simulationApp.py --mode DP --client 5 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 5 --expname DP_e5 --train_data train_log10 --test_data test_log10
python 02.simulationApp.py --mode DP --client 5 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 10 --expname DP_e10 --train_data train_log10 --test_data test_log10
python 02.simulationApp.py --mode DP --client 5 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 15 --expname DP_e15 --train_data train_log10 --test_data test_log10
```

##### Example of FL+DP+shuffling system

```
python 02.simulationApp.py --mode DP --client 5 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 5 --shuffle_model 1 --expname DPSM_e5 --train_data train_log10 --test_data test_log10
```

##### Example of varing number of clients

```
python 02.simulationApp.py --mode DP --client 5 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 1 --expname DP_e1 --train_data train_log10 --test_data test_log10
python 02.simulationApp.py --mode DP --client 50 --epochs 100 --batch_size 32 --lr 0.001 --epsilon 1 --expname DP_e1 --train_data train_log10 --test_data test_log10
```

#### Example of MIA

```
python 03.attackApp.py --model path/to/model
```

#### Visualization of MIA

```
MIAVisualization.ipynb
```



### Application 2: Clustering with scRNA-seq data

![image-20220210164936041](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/image-20220210164936041.0maKK3.png)

#### Root

Please run all codes in this section with ` Application/SingleCellClustering` as root.

#### Dataset Preparation

The count data for 4 scRNA-seq dataset can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1ZtVEYm4aB8nPMbrB9gDJ8ZFBede0DqHq?usp=sharing).

The original .rds file can be found in scDHA [website](https://bioinformatics.cse.unr.edu/software/scDHA/resource/Reproducibility/Data/) 

#### Optional Parameters

```
python 01.simulationApp.py --help
```

```
usage: 01.simulationApp.py [-h] [--device DEVICE] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--epsilon EPSILON] [--delta DELTA] [--mode MODE] [--client CLIENT] [--l2_clip L2_CLIP] [--nprocess NPROCESS] [--dataset DATASET] [--expname EXPNAME] [--train_data TRAIN_DATA] [--test_data TEST_DATA] [--shuffle_model SHUFFLE_MODEL]

dfsa

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       default: cuda:0
  --epochs EPOCHS       default: 50
  --batch_size BATCH_SIZE
                        default: 1
  --lr LR               default: 0.01
  --epsilon EPSILON     default: 100
  --delta DELTA         default: 0.5
  --mode MODE           default: SGD, {SGD, SIGNSGD, DP, DPSIGNSGD}
  --client CLIENT       default: 3
  --l2_clip L2_CLIP     default: 5
  --nprocess NPROCESS   default: 20
  --dataset DATASET     dataset name
  --expname EXPNAME     experiment name
  --train_data TRAIN_DATA
                        will load data/{}.npy
  --test_data TEST_DATA
                        will load data/{}.npy
  --shuffle_model SHUFFLE_MODEL
                        0: off, 1: on
```

#### Example of centrally trained system (#client=1)

```
for dataset in yan pollen hrvatin camp-liver
do
	python 01.simulationApp.py --mode=SGD --client=1 --epochs=10 --lr=0.001 --dataset=$dataset --expname="centrally"
done 
```

##### Example of FL system with 5 clients

```
for dataset in yan pollen hrvatin camp-liver
do
	python 01.simulationApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --dataset=$dataset --shuffle_model=0 --expname="FL"
done 
```

##### Example of FL+DP system with different privacy budget $\epsilon$

```
for dataset in yan pollen hrvatin camp-liver
do
	python 01.simulationApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --epsilon 1 --dataset=$dataset --shuffle_model=0 --expname="DP_e_1"
	python 01.simulationApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --epsilon 5 --dataset=$dataset --shuffle_model=0 --expname="DP_e_5"
	python 01.simulationApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --epsilon 10 --dataset=$dataset --shuffle_model=0 --expname="DP_e_10"
	python 01.simulationApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --epsilon 15 --dataset=$dataset --shuffle_model=0 --expname="DP_e_15"

done 

```

##### Example of FL+DP+shuffling system

```
for dataset in yan pollen hrvatin camp-liver
do
	python 01.simulationApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --epsilon 5 --dataset=$dataset --shuffle_model=1 --expname="DP_Shuffle_e_5"
done
```

#### Example on Patient data

```
python 03.simulationPatientApp.py --mode=SGD --client=5 --epochs=10 --lr=0.001 --epsilon 5 --dataset=P0123 --shuffle_model=1 --expname="P0123"
```

#### Test and Visualization

```
# For datasets: yan pollen hrvatin camp-liver
python 02.Test.py --dataset="yan" --model="model/FLDP+Shuffle_yan_modelbest.tar" --expname="DP_Shuffle_e_5"
# For patients
python 04.TestPatient.py --dataset="P0123" --model="model/P0123_modelbest.tar" --expname="DP_Shuffle_e_5"
```

### Application 3: Integration of tumour morphology and gene expression with spatial transcriptomics

![image-20220217104027879](https://cdn.jsdelivr.net/gh/JoshuaChou2018/oss@main/uPic/image-20220217104027879.mJF1xw.png)

#### Root

Please run all codes in this section with ` Application/SpatialTranscriptomics` as root.

#### Dataset Preparation

The complete dataset can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1ZtVEYm4aB8nPMbrB9gDJ8ZFBede0DqHq?usp=sharing).

#### Optional Parameters

```
python 01.simulationApp.py --help
```

```
usage: 01.simulationApp.py [-h] [--device DEVICE] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--epsilon EPSILON] [--delta DELTA] [--mode MODE] [--client CLIENT] [--l2_clip L2_CLIP] [--nprocess NPROCESS] [--expname EXPNAME] [--shuffle_model SHUFFLE_MODEL]

dfsa

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       default: cuda:0
  --epochs EPOCHS       default: 50
  --batch_size BATCH_SIZE
                        default: 1
  --lr LR               default: 0.01
  --epsilon EPSILON     default: 100
  --delta DELTA         default: 0.5
  --mode MODE           default: SGD, {SGD, SIGNSGD, DP, DPSIGNSGD}
  --client CLIENT       default: 3
  --l2_clip L2_CLIP     default: 5
  --nprocess NPROCESS   default: 20
  --expname EXPNAME     experiment name
  --shuffle_model SHUFFLE_MODEL
                        0: off, 1: on
```

#### Example of centrally trained system (#client=1)

```
python 01.simulationApp.py --device cuda:0 --mode SGD --client 1 --epochs 30 --batch_size 64 --lr 1e-5 --expname centrally
```

#### Example of FL system with 5 clients

```
python 01.simulationApp.py --device cuda:0 --mode SGD --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --expname FL
```

##### Example of FL+DP system with different privacy budget $\epsilon$

```
python 01.simulationApp.py --device cuda:1 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 0.01 --expname DP_e001
python 01.simulationApp.py --device cuda:1 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 0.1 --expname DP_e01
python 01.simulationApp.py --device cuda:1 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 0.5 --expname DP_e05
python 01.simulationApp.py --device cuda:0 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 1 --expname DP_e1
python 01.simulationApp.py --device cuda:0 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 5 --expname DP_e5
python 01.simulationApp.py --device cuda:1 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 10 --expname DP_e10
python 01.simulationApp.py --device cuda:1 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 15 --expname DP_e15
```

#### Example of FL+DP+shuffling system

```
python 01.simulationApp.py --device cuda:1 --mode DP --client 5 --epochs 30 --batch_size 32 --nprocess 15 --lr 1e-6 --epsilon 0.01 --shuffle_model 1 --expname DPSM_e001
```

#### Example of iDLG on centrally trained system

```
python 02.attackApp.py --mode SGD --expname iDLG_attack_SGD
```

#### Example of iDLG on PPML-Omics

```
python 02.attackApp.py --mode DP --expname iDLG_attack_DP --epsilon 0.01
```



# Citation

If you use our work in your research, please cite our paper:

**Juexiao Zhou, Siyuan Chen, et al. "PPML-Omics: a Privacy-Preserving federated Machine Learning system protects patients’ privacy from omic data"**