from model import StressMLP
from tqdm import tqdm
from data_processing import train_loader,dev_loader
import torch
import torch.nn as nn

model = StressMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
train_losses = []
dev_losses = []
# print(len(train_loader))
best_dev_loss = float("inf")
for epoch in tqdm(range(50)):
    model.train()
    total_loss = 0.0
    for input,target in train_loader:
        output = model(input)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input.size(0)
    avg_train_loss = total_loss/len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    model.eval()
    total_dev_loss = 0.0
    with torch.no_grad():
        for input,target in dev_loader:
            output = model(input)
            loss = criterion(output,target)
            total_dev_loss+= loss.item()* input.size(0)
        avg_dev_loss = total_dev_loss/len(dev_loader.dataset)
    dev_losses.append(avg_dev_loss)
    if avg_dev_loss<best_dev_loss:
        best_dev_loss = avg_dev_loss
        torch.save(model.state_dict(),"best_model.pth")

    # print(f'Train Loss: {avg_train_loss} | Dev Loss: {avg_dev_loss}')
    

