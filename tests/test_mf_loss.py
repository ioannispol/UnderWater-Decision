import numpy as np

from underwater_decision import MFLoss


def test_mf_loss():
    # Initialize the MFLoss with example weights and lambda_reg
    weights = np.array([1, 2, 3, 4])
    lambda_reg = 0.5
    mf_loss = MFLoss(weights, lambda_reg)
    
    # Example true labels and predictions
    y_true = np.array([[0, 0, 1, 0]])  # One-hot encoded true label for class 3
    y_pred = np.array([[0.1, 0.2, 0.6, 0.1]])  # Predicted probabilities
    
    # Calculate the loss
    loss = mf_loss(y_true, y_pred)
    
    # Expected loss calculation
    # Note: This is a simplified example. You should calculate this based on the actual formula and expected outcome.
    expected_loss = -np.sum(weights * y_true * np.log(y_pred)) + lambda_reg * np.sum(np.abs(weights - y_pred))
    expected_loss = np.mean(expected_loss)
    
    # Assert that the calculated loss is as expected
    np.testing.assert_almost_equal(loss, expected_loss, decimal=5, err_msg="MFLoss calculation is incorrect")
