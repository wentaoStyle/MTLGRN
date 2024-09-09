import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import metrics
import warnings
from train import Train

def main():
    warnings.filterwarnings("ignore")

    # Device setup
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Training parameters
    training_params = {
        'directory': 'mydata',
        'epochs': 100,
        'aggregator': 'GATConv',
        'embedding_size': 32,
        'dropout': 0.2,
        'slope': 0.2,  # LeakyReLU slope
        'lr': 0.0004,
        'wd': 1e-3,
        'random_seed': 1,
        'batch_size': 512,
        'device': device
    }

    # Train the model and get evaluation metrics
    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = Train(**training_params)

    # Print evaluation metrics
    def print_metrics(name, scores):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f'{name} mean: {mean_score:.4f}, variance: {std_score:.4f}')

    print_metrics('AUC', auc_scores)
    print_metrics('Accuracy', accuracy_scores)
    print_metrics('Precision', precision_scores)
    print_metrics('Recall', recall_scores)
    print_metrics('F1-score', f1_scores)

if __name__ == '__main__':
    main()
