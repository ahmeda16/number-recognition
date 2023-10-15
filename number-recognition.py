## Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

## Optional, for visualization purposes.
import matplotlib.pyplot as plt
# ---------------------------------------- PART 1 ---------------------------------------- Setting Tensor and Black and White Image
#Setting a "general 2D tensor"
tensor = torch.tensor([[1,2],[3,4]])

#Setting a 100x100 2D tensor of dones
tensor = torch.ones((100,100))
#Slicing Rows[30:50] and Cols[30:50] as zeros
tensor[30:50, 30:50] = 0
#Images are only black and white (1,0)

#General Function of showing a given plot with adjusted settings
def show_image(image):
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.show()

#Create a basic image and visualize it
#show_image(tensor)

# ---------------------------------------- PART 2 ---------------------------------------- Beginning to setup a NN model

#HYPERPARAMETERS
LEARNING_RATE = 0.001
HIDDEN_LAYER_UNITS = 500

#Fetching Data and Transform
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=T.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=T.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


#Visualize an image in our dataset
#show_image(train_dataset[12][0].squeeze())
#print(train_dataset[12][0].shape)

#Models in PyTorch are defined as a class
class SimpleNN(nn.Module):

    #Define our layers of the NN model. 
    #1 Input Layer, 1 Hidden Layer, 1 Output Layer (seperate, single-layer nerual nets chained together)
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(784, HIDDEN_LAYER_UNITS)                #input layer
        self.fc2 = nn.Linear(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS) #hidden layer
        self.fc3 = nn.Linear(HIDDEN_LAYER_UNITS, 10)                 #output layer


    #Defines what happens to the data as it goes through the network
    #1. flatten the 28x28 array into a 784x1 array
    #2. Apply each layer in order, with a relu activation function after each.
    #3. Convert whatever numbers the network outputs into a probability, assigning a higher probability for higher output values (softmax)
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        #Not calling activation function for output layer

        x = F.softmax(x, dim=1)

        return x
    
# ---------------------------------------- PART 3 ---------------------------------------- Training a NN model

#Defines a single neural network, if we want multiple networks, we instantiate multiple times
model = SimpleNN()

#Defines a Optimizer and Loss Function, encourages the model to assign a high probability to the correct label
loss_func = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

#Training and Validation Loop
#1. Loop over the training dataset (in batches), calculate the loss, the gradient, and adjust the weights
#2. Loop over the validation (test) dataset, take the highest probability label as a models prediction, and see how many it gets right. Reports it as accuracy
#3. Run the steps above multiple times, each time we do so (an epoch). We continue doing this until the accuracy appears to level off

#5 epochs
for epoch in range(5):

    #Training Loop
    for batch, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(data) #prediction on train data

        loss = loss_func(torch.log(out), targets) #how correct we are

        loss.backward() #gradient calculation
        optimizer.step() #adjust the weight

    #Testing Loop
        correct = 0
    for batch, (data, targets) in enumerate(test_loader):
        out = model(data)
        best_guesses = out.argmax(dim = 1) #how many are correct
        correct_now = (targets == best_guesses)
        correct += correct_now.sum()

    print(correct/len(test_dataset))

# ---------------------------------------- PART 4 ---------------------------------------- Visualizing a NN model
#A model has been trained! (after 5 epochs), we can use it for handwritten digit recognition.

#Number to test and predict 
sample_image = train_dataset[13][0]

#Setting our prediction from our trained model
out = model(sample_image)
print(out.argmax(dim=1)) #Output Prediction 

#Plotting our image to see the handwritten one
show_image(sample_image.squeeze())