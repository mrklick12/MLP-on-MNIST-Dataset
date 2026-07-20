import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# Creates a dataset of MNIST images and labels
# It takes in the images and converts them to tensors
# It then scales the pixel values to be between 0 and 1

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

# The batch size of the DataLoaders
batch_size = 64

# DataLoaders are a class that takes in the data and makes it iterable
# In this case, it will return (in batches of 64) the inputs and their labels
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# This will be run on a GPU, otherwise CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# This is the nerual network class that will be used to train the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # initalize its parent class
        self.flatten = nn.Flatten() # This will flatten the 28x28 images into a 784 element vector

        # 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss() # initializses the loss function to be used which is Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # initializes the optimizer to be used which is Adam with a learning rate of 0.001

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # gets the size of the dataset
    model.train() # sets the model to training mode
    for batch, (X, y) in enumerate(dataloader): # iterates through the dataloader
        X, y = X.to(device), y.to(device) # moves the data to the device (GPU or CPU)

        # Compute prediction error

        # FORWARD PASS
        pred = model(X) # gets the predictions from the model

        # CALCULATES LOSS
        loss = loss_fn(pred, y)

        # BACKWARD PASS
        loss.backward() # calculates the gradients
        optimizer.step() # updates the weights
        optimizer.zero_grad() # resets the gradients to zero

        if batch % 100 == 0: # prints out the loss every 100 batches
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):   
    size = len(dataloader.dataset) # gets the size of the dataset
    num_batches = len(dataloader) # gets the number of batches
    model.eval() # sets the model to evaluation mode
    test_loss, correct = 0, 0 # initializes the test loss and number of correct predictions to zero
    with torch.no_grad(): # disables gradient calculation
        for X, y in dataloader: # iterates through the dataloader
            X, y = X.to(device), y.to(device) # moves the data to the device (GPU or CPU)
            pred = model(X) # gets the predictions from the model
            test_loss += loss_fn(pred, y).item() # calculates the loss and adds it to the test loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # calculates the number of correct predictions and adds it to the correct variable
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 



epochs = 5 # number of epochs to train the model for
for t in range(epochs): # iterates through the number of epochs
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer) # trains the model
    test(test_dataloader, model, loss_fn) # tests the model
print("Done!")