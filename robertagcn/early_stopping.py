import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoints/best_model.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_loss, model, epoch, config):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, config)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, config)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, config):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        checkpoint = {
            'model_state': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': config,
            'early_stopping_best': self.best_score
        }
        
        torch.save(checkpoint, self.save_path)
        self.val_loss_min = val_loss