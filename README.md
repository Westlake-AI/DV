Structure-preserving visualization for single-cell RNA-Seq profiles using deep manifold transformation with batch-correction
=============
![image](https://github.com/Westlake-AI/DV/blob/master/Framework.png)

Overview
=============
Dimensionality reduction and visualization play an important role in biological data analysis, such as data interpretation of single-cell RNA sequences (scRNA-seq). It is desired to have a visualization method that can not only be applicable to various application scenarios, including cell clustering and trajectory inference, but also satisfy a variety of technical requirements, especially the ability to preserve inherent structure of data and handle with batch effects. However, no existing methods can accommodate these requirements in a unified framework. In this paper, we propose a general visualization method, deep visualization (DV), that possesses the ability to preserve inherent structure of data and handle batch effects and is applicable to a variety of datasets from different application domains and dataset scales. The method embeds a given dataset into a 2- or 3-dimensional visualization space, with either a Euclidean or hyperbolic metric depending on a specified task type with type static (at a time point) or dynamic (at a sequence of time points) scRNA-seq data, respectively. Specifically, DV learns a structure graph to describe the relationships between data samples, transforms the data into visualization space while preserving the geometric structure of the data and correcting batch effects in an end-to-end manner. The experimental results on nine datasets in complex tissue from human patients or animal development demonstrate the competitiveness of DV in discovering complex cellular relations, uncovering temporal trajectories, and addressing complex batch factors. We also provide a preliminary attempt to pre-train a DV model for visualization of new incoming data.

 python main.py


Requirements
=============
You'll need to install the following packages in order to run the codes.
anndata==0.7.6  
louvain==0.7.0  
pandas==1.2.0  
seaborn==0.11.1  
torch==1.7.1  
pytorch-lightning==1.2.8  
scanpy==1.7.2  
scipy==1.5.2  
numpy==1.19.2  
matplotlib==3.3.2  
plotly==4.14.3  
notebook==6.4.0  
ipykernel==5.3.4  

Citation
=============
Xu, Y., Zang, Z., Xia, J. et al. Structure-preserving visualization for single-cell RNA-Seq profiles using deep manifold transformation with batch-correction. Commun Biol 6, 369 (2023).

If you have any problem about this package, please create an Issue or send us an Email at:

* sky.yongjie.xu@hotmail.com
* xuyongjie@westlake.edu.cn
* stan.zq.li@westlake.edu.cn
