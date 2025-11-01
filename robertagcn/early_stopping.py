import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, 
                 save_path='checkpoints/best_model.pt', mode='max'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path for the checkpoint to be saved to.
            mode: 'min' for loss, 'max' for F1/accuracy
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path
        self.mode = mode
        
        if mode == 'min':
            self.best_value = np.inf
            self.compare = lambda new, best: new < best - delta
        else:  # mode == 'max'
            self.best_value = -np.inf
            self.compare = lambda new, best: new > best + delta
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, metric_value, model, epoch, config):
        """
        Args:
            metric_value: validation F1 (if mode='max') or loss (if mode='min')
        """
        if self.best_score is None:
            self.best_score = metric_value
            self.best_value = metric_value
            self.save_checkpoint(metric_value, model, epoch, config)
        elif self.compare(metric_value, self.best_value):
            self.best_score = metric_value
            self.best_value = metric_value
            self.save_checkpoint(metric_value, model, epoch, config)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, metric_value, model, epoch, config):
        if self.verbose:
            metric_name = 'F1' if self.mode == 'max' else 'Loss'
            print(f'Validation {metric_name} improved ({self.best_value:.6f} --> {metric_value:.6f}). Saving...')
        
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch,
            'metric_value': metric_value,
            'config': config
        }, self.save_path)
        
        self.best_value = metric_value