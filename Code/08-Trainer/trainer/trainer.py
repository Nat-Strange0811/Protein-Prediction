#PyTorch imports
# Path and typing imports
from pathlib import Path

import torch
import torch.nn as nn
from configs import Config
from dataset import ProteinDataset

# FusionModel import
from fusion_model import FusionModel

# sklearn imports
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, random_split

'''
Class - Trainer:
This class defines a trainer for the FusionModel. It handles the training, validation, and testing of the model using a specified dataset.
The trainer implements early stopping based on validation AUC and saves the best model checkpoint during training.

Attributes:
- model: An instance of the FusionModel to be trained.
- dataset: The dataset to be used for training, validation, and testing.
- batch_size: The batch size for training and evaluation.
- patience: The number of epochs to wait for improvement in validation AUC before stopping training early.
- checkpoint_dir: The directory where the best model checkpoint will be saved.

Methods:
- __init__(self, model: FusionModel, dataset, batch_size: int = 32, patience: int = 20, 
checkpoint_dir: str = '/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Logs'): Initializes the Trainer with the specified parameters
and splits the dataset into training, validation, and test sets.
- _run_epoch(self, loader, train: bool) -> Tuple[float, float, float]: Runs a single epoch of training or evaluation on the specified
data loader and returns the average loss, AUC, and accuracy.
- train(self, epochs: int): Trains the model for the specified number of epochs, implementing early stopping based on validation AUC and 
saving the best model checkpoint.
- evaluate(self): Evaluates the best model on the test set and returns the test loss, AUC, and accuracy.
'''
class Trainer:
    #Init method to define class attributes and split the dataset into training, validation, and test sets.
    def __init__(
        self,
        model: FusionModel,
        dataset: ProteinDataset,
        config: Config,
    ):
        # Define the device to be used for training (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the model to the specified device
        self.model = model.to(self.device)
        self.patience = config.training.patience
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ID = f"{self.model.ID}-{dataset.ID}-Trainer_{config.training.seed}_{config.training.learning_rate}_{config.training.weight_decay}_{config.training.dropout_rate}"
        
        # Split dataset
        n = len(dataset)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(config.training.seed))
        
        # Create DataLoaders for training, validation, and testing
        self.train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=config.training.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=config.training.batch_size)
        
        # Define the loss criterion and optimizer for training
        self.criterion = nn.BCELoss()
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        
        self.val_threshold = 0.5  # Initialize the validation threshold to 0.5
        self.threshold = 0.5  # Initialize the threshold to 0.5
    
    def _optimal_threshold(self, labels, preds):
        #Youden's J statistic to find the optimal threshold for binary classification based on the ROC curve
        fpr, tpr, thresholds = roc_curve(labels, preds)
        optimal_idx = (tpr - fpr).argmax()
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold
        
    '''
    Method - _run_epoch:
    
    Runs a single epoch of training or evaluation on the specified data loader and returns the average loss, AUC, and accuracy.
    
    Inputs:
    - loader: The DataLoader to be used for training or evaluation.
    - train: A boolean indicating whether to run the epoch in training mode (True) or evaluation mode (False).

    Outputs:
    - avg_loss: The average loss over the epoch.
    - auc: The AUC score over the epoch.
    - accuracy: The accuracy over the epoch.
    
    Raises:
    - None
    '''
    def _run_epoch(self, loader, train: bool, threshold: float | None = None) -> tuple[float, float, float, float]:
        # Set the model to training or evaluation mode based on the 'train' parameter
        self.model.train() if train else self.model.eval()
        # Initialise total loss and lists to store predictions and labels for AUC and accuracy calculation
        total_loss = 0
        all_preds, all_labels = [], []
        
        # Use torch.set_grad_enabled to enable or disable gradient computation based on the 'train' parameter
        with torch.set_grad_enabled(train):
            # Iterate through the data loader to get batches of embeddings and labels
            for embeddings, labels, _ in loader:
                # Move embeddings and labels to the specified device (GPU or CPU)
                embeddings = [e.to(self.device) for e in embeddings]
                labels = labels.to(self.device)
                
                # Forward pass through the model to get predictions and compute the loss
                preds = self.model(embeddings).squeeze(-1)
                loss = self.criterion(preds, labels)
                
                # If in training mode, perform backpropagation and update the model parameters
                if train:
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()
                
                # Accumulate the total loss and store predictions and labels for AUC and accuracy calculation
                total_loss += loss.item()
                # Store predictions and labels for AUC and accuracy calculation
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Calculate average loss, AUC, and accuracy for the epoch
        avg_loss = total_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        if threshold is None:
            threshold = self._optimal_threshold(all_labels, all_preds)
        
        accuracy = sum(
            (p >= threshold) == l 
            for p, l in zip(all_preds, all_labels)
        ) / len(all_labels) * 100
        
        return avg_loss, auc, accuracy, threshold
    
    '''
    Method - train: 
    
    Trains the model for the specified number of epochs, implementing early stopping based on validation AUC and saving the best model
    checkpoint.
    
    Inputs:
    - epochs: The number of epochs to train the model.
    
    Outputs:
    - None
    
    Raises:
    - None
    '''
    def train(self, epochs: int):
        # Initialize variables to track the best validation AUC and the patience counter for early stopping
        best_auc = 0
        best_acc = 0
        best_train_auc = 0
        patience_counter = 0
        
        # Loop through the specified number of epochs to train the model
        for epoch in range(epochs):
            # Run a training epoch and a validation epoch, and get the average loss, AUC, and accuracy for both
            train_loss, train_auc, train_acc, _ = self._run_epoch(self.train_loader, train=True, threshold=self.val_threshold)
            val_loss, val_auc, val_acc, val_threshold = self._run_epoch(self.val_loader, train=False, threshold=None)
            self.val_threshold = val_threshold  # Store the optimal threshold for validation
            
            # Print the training and validation metrics for the current epoch
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} Acc: {val_acc:.1f}%"
            )
            
            # Check if the validation AUC has improved; if so, save the model checkpoint and reset the patience counter. If not, increment
            if val_auc > best_auc:
                best_auc = val_auc
                best_acc = val_acc
                self.threshold = val_threshold
                best_train_auc = train_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / f'best_model_{self.ID}.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
        self.results = {
            "best_auc": best_auc,
            "best_acc": best_acc,
            "best_epoch": epoch + 1 - patience_counter,
            "train_auc": best_train_auc
        }
    
    #Runs evaluation on the test set using the best model checkpoint and returns the test loss, AUC, and accuracy.
    def evaluate(self):
        self.model.load_state_dict(torch.load(
            self.checkpoint_dir / f'best_model_{self.ID}.pt', weights_only=True
        ))
        test_loss, test_auc, test_acc, _ = self._run_epoch(self.test_loader, train=False, threshold=self.threshold)
        print(f"Test Loss: {test_loss:.4f} AUC: {test_auc:.4f} Acc: {test_acc:.1f}%")
        
        return test_loss, test_auc, test_acc