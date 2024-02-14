# SAR Ship Classification with Different Loss Functions

This code explores the impact of various loss functions on the accuracy of SAR ship classification. It trains a convolutional neural network (CNN) with four different loss functions and compares their performance.

**Key Features:**

- Compares CrossEntropy, L1, MSE, BCEWithLogitsLoss, Kullback-Leibler Divergence and Focal Loss
- Uses PyTorch for deep learning
- Trains on the Fusar dataset, and OpenSARShip dataset
- Evaluates accuracy on a test set
- Supports GPU acceleration

## Loss Functions in SAR Ship Classification: 

This section briefly explains and presents formulas for the loss functions used in the paper:

**1. CrossEntropyLoss:**

* **Formula:** `CrossEntropy(y_true, y_pred) = -Σ[y_true_i * log(y_pred_i)]`
* **Explanation:** Measures the difference between the predicted probability distribution (y_pred) and the true label distribution (y_true) for each class.

**2. L1 Loss:**

* **Formula:** `L1Loss(y_true, y_pred) = 1/N * Σ|y_true_i - y_pred_i|`
* **Explanation:** Measures the average absolute difference between predicted values (y_pred) and true values (y_true). Robust to outliers.

**3. Mean Squared Error (MSE):**

* **Formula:** `MSELoss(y_true, y_pred) = 1/N * Σ(y_true_i - y_pred_i)^2`
* **Explanation:** Measures the average squared difference between predicted values (y_pred) and true values (y_true). Sensitive to outliers.

**4. BCEWithLogitsLoss:**

* **Formula:** `BCEWithLogitsLoss(y_true, y_pred) = -[y_true * log(σ(y_pred)) + (1 - y_true) * log(1 - σ(y_pred))]`
* **Explanation:** Measures the binary cross-entropy between the model's logits (unscaled output) and the true labels (0 or 1).

**5. Focal Loss:**

* **Formula:** `FocalLoss(y_true, y_pred) = -Σ[α * (1 - p_t)^γ * y_t * log(p_t)]`
* **Explanation:** Similar to BCEWithLogitsLoss, but introduces hyperparameters α and γ to downweight the loss for easily classified samples, focusing training on harder ones.

**6. Kullback-Leibler Divergence Loss:**

* **Formula:** `KLDiv(p||q) = Σ(p_i * log (p_i/q_i))`
* **Explanation:** This measures the difference between two probability distributions and can be used as a loss function. However, it is less common than Cross-Entropy due to its computational complexity and sensitivity to outliers.

## Code Details 

**Requirements:**

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- numpy

**Usage:**

1. Install dependencies: `pip install -r requirements.txt`
2. Download and extract the Fusar dataset into a directory named "data"
3. Keep only three classes of data, Fishing, Tanker and Cargo. Whereas keep Fusar data in tiff format, while OpenSARShip in png format.
4. Run the code in Jypyter Notebook: `sar_loss.ipynb`

**Output:**

- Prints training progress and loss values for each epoch
- Prints final accuracy for each loss function

**Customization:**

- Adjust hyperparameters like learning rate, batch size, and epochs in the code
- Add more loss functions to the `loss_functions` dictionary
- Experiment with different dataset directories and model architectures

**Contributing:**

- Fork the repository
- Create a branch for your changes
- Make your changes and tests
- Submit a pull request

**License:**

- [N/A]
