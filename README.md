# SAR Ship Classification with Different Loss Functions

This code explores the impact of various loss functions on the accuracy of SAR ship classification. It trains a convolutional neural network (CNN) with four different loss functions and compares their performance.

**Key Features:**

- Compares L1, MSE, BCEWithLogitsLoss, and Focal Loss
- Uses PyTorch for deep learning
- Trains on the Fusar dataset, and OpenSARShip dataset
- Evaluates accuracy on a test set
- Supports GPU acceleration

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
4. Run the code: `python train.py`

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
