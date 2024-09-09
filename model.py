import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.functional as pyg_func


class NET(nn.Module):
    def __init__(self, in_features, embedding_size, n_feature, out, dropout, slope, aggregator):
        super(NET, self).__init__()

        # Define the GAT layers
        self.gat1 = pyg_nn.GATConv(in_channels=in_features, out_channels=embedding_size, heads=8, concat=True)
        self.gat2 = pyg_nn.GATConv(in_channels=embedding_size * 8, out_channels=embedding_size, heads=1, concat=False)

        # Define the MLP for feature reconstruction
        self.mlp_reconstruct = nn.Sequential(
            nn.Linear(embedding_size, n_feature),
            nn.ReLU(),
            nn.Linear(n_feature, out)
        )

        # Define the MLP for knockout prediction
        self.mlp_knockout = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, out)
        )

        # Define dropout and activation functions
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope)
        self.elu = nn.ELU()

    def encode(self, x, edge_index):
        """
        Encodes node features using GAT layers.

        Parameters:
        x (Tensor): Node feature matrix.
        edge_index (Tensor): Edge index tensor.

        Returns:
        Tensor: Encoded node embeddings.
        """
        x = self.gat1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.leaky_relu(x)
        return x

    def decode_grn(self, z, edge_index):
        """
        Decodes node embeddings to predict gene regulatory interactions.

        Parameters:
        z (Tensor): Node embeddings.
        edge_index (Tensor): Edge index tensor.

        Returns:
        Tensor: Predictions for the gene regulatory network.
        """
        edge_index = edge_index
        edge_attr = torch.ones(edge_index.size(1), device=z.device)
        return torch.sigmoid(pyg_func.matmul(z[edge_index[0]], z[edge_index[1]]))

    def decode_knockout(self, x, knockout_vector):
        """
        Predicts the gene expression changes due to gene knockout.

        Parameters:
        x (Tensor): Node feature matrix before knockout.
        knockout_vector (Tensor): Vector representing the target gene knockout.

        Returns:
        Tensor: Predicted gene expression changes.
        """
        concat_vector = torch.cat((x, knockout_vector), dim=-1)
        return self.mlp_knockout(concat_vector)

    def reconstruct_expression(self, z, sample_features):
        """
        Reconstructs gene expression matrix based on node embeddings and sample features.

        Parameters:
        z (Tensor): Node embeddings.
        sample_features (Tensor): Feature matrix of samples.

        Returns:
        Tensor: Reconstructed gene expression matrix.
        """
        return self.mlp_reconstruct(z)

