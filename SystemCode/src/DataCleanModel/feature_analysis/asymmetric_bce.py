import numpy as np

class AsymmetricBCEWithLogitsLoss:
    def __init__(self, negative_weight=5.0, positive_weight=1.0):
        """
        Asymmetric BCE Loss in pure NumPy.
        
        Args:
            negative_weight (float): Penalty multiplier for the negative class (FP reduction).
            positive_weight (float): Penalty multiplier for the positive class.
        """
        self.neg_weight = negative_weight
        self.pos_weight = positive_weight

    def forward(self, logits, targets):
        """
        Calculates the weighted BCE loss.
        
        Args:
            logits (np.ndarray): Raw, unnormalized predictions.
            targets (np.ndarray): Ground truth labels (0 or 1).
            
        Returns:
            float: The scalar loss value.
        """
        # Save inputs for the backward pass
        self.saved_logits = logits
        self.saved_targets = targets
        
        # 1. Numerically stable BCE with logits
        # Math: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        bce_loss = np.maximum(logits, 0) - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
        
        # 2. Calculate asymmetric weights
        weights = targets * self.pos_weight + (1.0 - targets) * self.neg_weight
        
        # 3. Apply weights and calculate mean
        weighted_loss = bce_loss * weights
        return np.mean(weighted_loss)
