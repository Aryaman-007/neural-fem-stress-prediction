import torch
import torch.nn as nn
class StressMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128,3)
        self.relu = nn.ReLU()
    def forward(self,x):
        ##defining forward propagation basically
        x = self.fc1(x) #input enters in the first layer (input)
        x = self.relu(x) #activating the linear part

        x = self.fc2(x) #input enters 2nd layer now
        x = self.relu(x) #activating linear part

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        #no need to activate because this is the output layer

        return x


model = StressMLP()
x = torch.randn(1,5)
y = model(x)

# print("input shape: ",x.shape)
# print("output shape:", y.shape)
# print("Prediction: ",y)


        
    