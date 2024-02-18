import numpy as np
import xgboost as xgb


class MFLoss:
    """
    Custom Marine Fouling Loss Function (CMFLF) for prioritizing accurate predictions of higher fouling levels.
    """
    def __init__(self, weights, lambda_reg):
        """
        Initialize the loss function with specific weights and regularization parameter.
        
        Parameters:
        - weights (list of float): Weights assigned to each fouling level, increasing with severity.
        - lambda_reg (float): Regularization parameter controlling balance across classes.
        """
        self.weights = weights
        self.lambda_reg = lambda_reg
    
    def __call__(self, y_true, y_pred):
        """
        Compute the CMFLF loss given true labels and predictions.
        
        Parameters:
        - y_true (numpy.array): One-hot encoded true labels.
        - y_pred (numpy.array): Predicted probabilities for each class.
        
        Returns:
        - float: Computed loss value.
        """
        import numpy as np
        
        # Ensure y_pred is clipped to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Weighted log loss
        weighted_log_loss = -np.sum(self.weights * y_true * np.log(y_pred), axis=1)
        
        # Regularization term
        regularization = self.lambda_reg * np.sum(np.abs(self.weights - y_pred), axis=1)
        
        # Total loss
        total_loss = np.mean(weighted_log_loss + regularization)
        
        return total_loss


class MFLossObjective(MFLoss):
   """
    Adapter for MFLoss to be used as an objective in XGBoost training.
    """
    def __init__(self, weights, lambda_reg):
        super().__init__(weights, lambda_reg)
        
    def xgb_obj(self, y_true, y_pred):
        """
        Custom objective function for XGBoost that returns gradient and hessian.
        """
        # Placeholder for gradient and hessian calculation
        grad = self._calculate_gradient(y_true, y_pred)
        hess = self._calculate_hessian(y_true, y_pred)
        return grad, hess
    
    def _calculate_gradient(self, y_true, y_pred):
        # Simplified gradient calculation
        grad = y_pred - y_true
        return grad

    def _calculate_hessian(self, y_true, y_pred):
        # Simplified hessian calculation
        hess = np.ones_like(y_pred)
        return hess

