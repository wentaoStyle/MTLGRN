# MTLGR


## Abstract
Constructing accurate gene regulatory networks (GRNs), which reflect the dynamic governing process between genes, is critical to understanding the diverse cellular process and unveiling the complexities in biological systems. 
With the development of computer sciences, computational-based approaches have been applied to the GRNs inference task. 
However, current methodologies face challenges in effectively utilizing existing topological information and prior knowledge of gene regulatory relationships, hindering the comprehensive understanding and accurate reconstruction of GRNs. 
In response, we propose a novel Graph Neural Network (GNN)-based Multi-T}ask Learning framework for GRN reconstruction, namely MTLGRN. 
Specifically, we first encode the gene promoter sequences and the gene biological features and concatenate the corresponding feature representations. 
Then, we construct a multi-task learning framework including Gene regulatory network reconstruction, Gene knockout predict, and Gene expression matrix reconstruction. 
With joint training, MTLGRN can optimize the gene latent representations by integrating gene knockout information, promoter characteristics, and other biological attributes. 
Extensive experimental results demonstrate superior performance compared with state-of-the-art baselines on the GRN reconstruction task, efficiently leveraging biological knowledge and comprehensively understanding the gene regulatory relationships. 
MTLGRN also pioneering attempts to simulate gene knockouts on bulk data by incorporating gene knockout information. 

## Workflow

![image](https://github.com/wentaoStyle/MTLGRN/blob/main/fig1_framework.pdf)

## Dependencies
```bash
conda env create -f environment.yml
```

## Data
To download the dataset required to run the code from the link below:
https://www.scidb.cn/en/s/VNvY3e](https://www.scidb.cn/en/s/uIZfqu

## Running
```bash
python main.py
```


# Citation
If you use this code for your research, please cite our paper [Refining Computational Inference of Gene Regulatory Networks: Integrating Knockout Data within a Multi-Task Framework]([https://biorxiv.org/content/10.1101/2023.09.25.559244v1.abstract](https://academic.oup.com/bib/article/25/5/bbae361/7724463?login=false)
```bash
@article{cui2024refining,
  title={Refining computational inference of gene regulatory networks: integrating knockout data within a multi-task framework},
  author={Cui, Wentao and Long, Qingqing and Xiao, Meng and Wang, Xuezhi and Feng, Guihai and Li, Xin and Wang, Pengfei and Zhou, Yuanchun},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={5},
  year={2024},
  publisher={Oxford Academic}
}
}
```

## License
This project is licensed under the MIT License.







