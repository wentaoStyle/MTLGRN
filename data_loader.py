import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch_geometric.data import Data


def kmer_encoding(sequences, k):
    """
    Encode DNA sequences using k-mer encoding.

    Parameters:
    sequences (list of str): List of DNA sequences.
    k (int): Length of k-mers.

    Returns:
    np.ndarray: k-mer encoded matrix.
    """
    kmer_dict = {}
    kmer_index = 0
    encoded_matrix = []

    for seq in sequences:
        kmer_list = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        encoded_vector = np.zeros(len(kmer_dict))

        for kmer in kmer_list:
            if kmer not in kmer_dict:
                kmer_dict[kmer] = kmer_index
                kmer_index += 1
            encoded_vector[kmer_dict[kmer]] = 1

        encoded_matrix.append(encoded_vector)

    return np.array(encoded_matrix)


def one_hot_encoding(categories, num_classes):
    """
    Convert categorical features to one-hot encoding.

    Parameters:
    categories (list of int): List of categorical feature values.
    num_classes (int): Total number of unique categories.

    Returns:
    np.ndarray: One-hot encoded matrix.
    """
    one_hot_matrix = np.zeros((len(categories), num_classes))
    for idx, category in enumerate(categories):
        one_hot_matrix[idx, category] = 1
    return one_hot_matrix


def preprocess_features(promoter_sequences, biological_features, k, num_classes):
    """
    Preprocess gene promoter sequences and biological features.

    Parameters:
    promoter_sequences (list of str): List of gene promoter sequences.
    biological_features (list of int): List of categorical biological features.
    k (int): Length of k-mers for encoding.
    num_classes (int): Number of unique biological feature categories.

    Returns:
    np.ndarray: Processed feature matrix after encoding and dimensionality reduction.
    """
    # K-mer encoding for promoter sequences
    promoter_features = kmer_encoding(promoter_sequences, k)

    # One-hot encoding for biological features
    biological_features_encoded = one_hot_encoding(biological_features, num_classes)

    # Concatenate encoded features
    combined_features = np.hstack((promoter_features, biological_features_encoded))

    # Dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    reduced_features = pca.fit_transform(combined_features)

    return reduced_features


def load_data(data_path):
    df = pd.read_csv(data_path)
    print(df)

    gene_list = []
    dise_id = df['Gene']
    Gene_id = df['Target_gene']

    for i in dise_id:
        gene_list.append(i)
    for i in Gene_id:
        gene_list.append(i)

    gene_list = set(gene_list)
    gene_list = list(gene_list)
    gene_mapping = {index_id: int(i) for i, index_id in enumerate(gene_list)}

    src_nodes = [gene_mapping[index] for index in df['Gene']]
    dst_nodes = [gene_mapping[index] for index in df['Target_gene']]
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    rev_edge_index = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)

    gene_feature = pd.read_csv("seq_feature.csv")
    result_gene_feature = pd.read_csv('gene_feature.csv')

    result_gene_feature = result_gene_feature.drop(['gene_name'], axis=1)
    result_gene_feature = result_gene_feature.values

    gene_feature = np.concatenate((gene_feature, result_gene_feature), axis=1)

    # Load promoter sequences and biological features
    promoter_sequences = [...]  # List of gene promoter sequences
    biological_features = [...]  # List of categorical biological features

    k = 6  # Example k-mer length
    num_classes = len(set(biological_features))  # Number of unique categories in biological features

    # Preprocess features
    processed_features = preprocess_features(promoter_sequences, biological_features, k, num_classes)

    data = Data()
    data.num_nodes = len(gene_mapping)
    data.x = torch.tensor(processed_features, dtype=torch.float32)
    data.edge_index = torch.cat([edge_index, rev_edge_index], dim=1)

    return data, gene_mapping


def get_fre_3(data_c, data_t):
    change = data_t - data_c

    mask_up = change >= np.log(2)
    mask_none = (change > -np.log(2)) & (change < np.log(2))
    mask_down = change <= -np.log(2)
    target = pd.DataFrame(np.zeros_like(data_c), dtype=int)

    target[mask_down] = 2
    target[mask_none] = 0
    target[mask_up] = 1

    class_hist, _ = np.histogram(target, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    label_freq = class_hist / class_hist.sum()
    target_fre = target
    target_fre.replace(0, 1 - label_freq[0], inplace=True)
    target_fre.replace(1, 1 - label_freq[1], inplace=True)
    target_fre.replace(2, 1 - label_freq[2], inplace=True)

    return target_fre


def get_fre_5(data_c, data_t):
    change = data_t - data_c

    mask_up5x = change >= np.log(5)
    mask_up2x = (change >= np.log(2)) & (change < np.log(5))
    mask_none = (change > -np.log(2)) & (change < np.log(2))
    mask_down2x = (change <= -np.log(2)) & (change > -np.log(5))
    mask_down5x = change <= -np.log(5)
    target = pd.DataFrame(np.zeros_like(data_c), dtype=int)

    target[mask_down5x] = 4
    target[mask_down2x] = 3
    target[mask_none] = 0
    target[mask_up2x] = 2
    target[mask_up5x] = 1

    class_hist, _ = np.histogram(target, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    label_freq = class_hist / class_hist.sum()
    target_fre = target

    target_fre.replace(0, 1 / label_freq[0], inplace=True)
    target_fre.replace(1, 0.01 * label_freq[1], inplace=True)
    target_fre.replace(2, 0.02 * label_freq[2], inplace=True)
    target_fre.replace(3, 0.01 * label_freq[3], inplace=True)
    target_fre.replace(4, 0.01 * label_freq[4], inplace=True)

    return target_fre
