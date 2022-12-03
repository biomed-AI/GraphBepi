# Intro  
GraphBepi is a novel framework for structure-based protein-protein interaction site prediction using deep graph convolutional network, which is able to capture information from high-order spatially neighboring amino acids. The GraphPPIS source code is designed for high-throughput predictions, and does not have the limitation of one query protein per run. We recommend you to use the [web server](http://bio-web1.nscc-gz.cn/apps) of GraphPPIS if your input is small.  
![GraphBepi_framework]()  

# System requirement  
GraphBepi is developed under Linux environment with:  
- python  3.9.12  
- numpy  1.21.5
- pandas  1.4.2
- fair-esm  2.0.0
- torch  1.12.1
- pytorch-lightning  1.6.4
- (optional) esmfold
# Software requirement  
To run the full & accurate version of GraphBepi, you need to make sure the following software is in the [mkdssp](./mkdssp) directory:  
[DSSP](https://github.com/cmbi/dssp) (*dssp ver 2.0.4* is Already in this repository) 

# Build dataset  
1. `git clone https://github.com/biomed-AI/GraphBepi.git && cd GraphBepi`
2. `python dataset.py --gpu 0`

It will take about 20 minutes to download the pretrained ESM-2 model and an hour to build our dataset with CUDA.
# Run GraphBepi for training
After build our dataset ***BCE_633***, train the model with default hyper params:
```
python train.py --dataset BCE_633
```
# Run GraphBepi for prediction  
For sequences in fasta file:  
```
python test.py -i fasta_file -f --gpu 0 -o ./output
```
For a protein structure in PDB file:  
```
python test.py -i pdb_file -p --gpu 0 -o ./output
```

# How to reproduce our work  
We provide the datasets, pre-computed features, the two pre-trained models, and the training and evaluation codes for those interested in reproducing our paper.  
The datasets used in this study (Train_335, Test_60, Test_315 and UBtest_31) are stored in ./Dataset in fasta format.  
The distance maps(L * L) and normalized feature matrixes PSSM(L * 20), HMM(L * 20) and DSSP(L * 14) are stored in ./Feature in numpy format.  
The pre-trained GraphPPIS full model and the simplified version using BLOSUM62 can be found under ./Model  
The training and evaluation codes can be found in [here](https://github.com/yuanqm55/GraphPPIS).  

# Web server, citation and contact  
The GraphPPIS web server is freely available: [old interface](https://biomed.nscc-gz.cn/apps/GraphPPIS) or [new interface](http://bio-web1.nscc-gz.cn/apps)  

Citation:  
```bibtex
@article{10.1093/bioinformatics/btab643,
    author = {Yuan, Qianmu and Chen, Jianwen and Zhao, Huiying and Zhou, Yaoqi and Yang, Yuedong},
    title = "{Structure-aware protein–protein interaction site prediction using deep graph convolutional network}",
    journal = {Bioinformatics},
    volume = {38},
    number = {1},
    pages = {125-132},
    year = {2021},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab643},
    url = {https://doi.org/10.1093/bioinformatics/btab643},
}
```

Contact:  
Zhuoyi Wei (weizhy8@mail2.sysu.edu.cn)
Yuansong Zeng (zengys@mail.sysu.edu.cn)    
Yuedong Yang (yangyd25@mail.sysu.edu.cn)

