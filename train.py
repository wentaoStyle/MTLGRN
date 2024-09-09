import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch_geometric.transforms as T
from model import NET
from data_loader import load_data

def Train(directory, epochs, aggregator, embedding_size, dropout, slope, lr, wd, random_seed,batch_size, device):
    # Device setup
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device(device)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
    ])

    # Data loading
    dataset, gene_mapping = load_data(directory)
    print(dataset)
    print("Number of genes:", len(gene_mapping))

    # Data transformation
    train_data, val_data, test_data = transform(dataset)
    print("Train Data:", train_data)
    print("Validation Data:", val_data)
    print("Test Data:", test_data)

    # Loss functions
    criterion_grn = nn.BCEWithLogitsLoss().to(device)

    def criterion_knockout(predictions, targets):
        return nn.MSELoss()(predictions, targets)

    criterion_reconstruction = nn.MSELoss().to(device)

    # Data preparation
    gene_expression_data = pd.read_csv('tf_ko.csv')
    feature= pd.read_csv('feature.csv').values.T
    gene_expression_data = gene_expression_data.drop(['index'], axis=1)

    num_genes = gene_expression_data.shape[0]
    num_features = num_genes * 2
    num_samples = gene_expression_data.shape[1]
    num_samples_control = num_samples // 2

    gene_expression_data = np.log(gene_expression_data + 1)
    control_data = gene_expression_data[:, num_samples_control:].T
    treatment_data = gene_expression_data[:, :num_samples_control].T
    control_data = np.concatenate((control_data, feature), axis=1)

    control_data_train, control_data_test, treatment_data_train, treatment_data_test = train_test_split(control_data, treatment_data)

    control_tensor_train = torch.tensor(control_data_train, dtype=torch.float32)
    treatment_tensor_train = torch.tensor(treatment_data_train, dtype=torch.float32)
    control_tensor_test = torch.tensor(control_data_test, dtype=torch.float32)
    treatment_tensor_test = torch.tensor(treatment_data_test, dtype=torch.float32)

    control_data_transposed = control_tensor_train[:, :num_genes].to(device)

    train_dataset = data.TensorDataset(control_tensor_train, treatment_tensor_train)
    test_dataset = data.TensorDataset(control_tensor_test, treatment_tensor_test)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Model setup
    model = NET(dataset.num_features, embedding_size, n_feature=num_features, out=num_genes, dropout=dropout,
                slope=slope, aggregator=aggregator).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)

    def train_step(batch_x, batch_y):
        model.train()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        node_embeddings = model.encode(train_data.x, train_data.edge_index)
        edge_labels = train_data.edge_label

        grn_predictions = model.decode_grn(node_embeddings, train_data.edge_index).view(-1)
        batch_features = batch_x[:, :num_features]

        knockout_predictions = model.decode_knockout(batch_features, node_embeddings)
        reconstruction_predictions = model.reconstruct_expression(node_embeddings, batch_x)

        loss_grn = criterion_grn(grn_predictions, edge_labels)
        loss_knockout = criterion_knockout(knockout_predictions, batch_y)
        loss_reconstruction = criterion_reconstruction(reconstruction_predictions, control_data_transposed.T)

        total_loss = loss_grn + loss_knockout + loss_reconstruction

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss

    @torch.no_grad()
    def evaluate(data):
        model.eval()
        node_embeddings = model.encode(data.x, data.edge_index)
        grn_predictions = model.decode_grn(node_embeddings, data.edge_index).view(-1).sigmoid()
        y_true = data.edge_label.cpu().numpy()
        y_pred = grn_predictions.cpu().numpy()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, y_pred >= 0.5), precision_score(y_true,
                                                                                                     y_pred >= 0.5), recall_score(
            y_true, y_pred >= 0.5), f1_score(y_true, y_pred >= 0.5)

    auc_results, acc_results, pre_results, recall_results, f1_results = [], [], [], [], []

    for epoch in range(1, epochs + 1):
        best_val_acc = final_test_acc = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            loss = train_step(batch_x, batch_y)
            val_auc, val_acc, val_pre, val_recall, val_f1 = evaluate(val_data)
            test_auc, test_acc, test_pre, test_recall, test_f1 = evaluate(test_data)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

            auc_results.append(test_auc)
            acc_results.append(test_acc)
            pre_results.append(test_pre)
            recall_results.append(test_recall)
            f1_results.append(test_f1)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')

    return auc_results, acc_results, pre_results, recall_results, f1_results
