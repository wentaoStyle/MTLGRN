# MTLGR
Constructing accurate gene regulatory networks (GRNs), which reflect the dynamic governing process between genes, is critical to understanding the diverse cellular process and unveiling the complexities in biological systems. 
With the development of computer sciences, computational-based approaches have been applied to the GRNs inference task. 
However, current methodologies face challenges in effectively utilizing existing topological information and prior knowledge of gene regulatory relationships, hindering the comprehensive understanding and accurate reconstruction of GRNs. 
In response, we propose a novel Graph Neural Network (GNN)-based \underline{\textbf{M}}ulti-\underline{\textbf{T}}ask \underline{\textbf{L}}earning framework for \underline{\textbf{GRN}} reconstruction, namely MTLGRN. 
Specifically, we first encode the gene promoter sequences and the gene biological features and concatenate the corresponding feature representations. 
Then, we construct a multi-task learning framework including \textit{Gene regulatory network reconstruction}, \textit{Gene knockout predict}, and \textit{Gene expression matrix reconstruction}. 
With joint training, MTLGRN can optimize the gene latent representations by integrating gene knockout information, promoter characteristics, and other biological attributes. 
Extensive experimental results demonstrate superior performance compared with state-of-the-art baselines on the GRN reconstruction task, efficiently leveraging biological knowledge and comprehensively understanding the gene regulatory relationships. 
MTLGRN also pioneering attempts to simulate gene knockouts on bulk data by incorporating gene knockout information. 

# Dependencies
'''
conda env create -f environment.yml
'''

#Running
'''
python main.py
'''
