
# STAGING FOR CHEST XRAY DATASET

## Importing Relevant Libraries and Models


```python
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import (
    models, 
    datasets, 
    transforms
)


```

## Transformation Pipelines


```python

sizing = 256

# We are keeping minimalistic transforms
# To get preserve effects in the xrays as much as possible
train_transform = transforms.Compose([
    transforms.RandomRotation(10, expand=True),
    transforms.Resize(sizing),
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.Resize(sizing),
    transforms.ToTensor()
])   
    
test_transform = transforms.Compose([
    transforms.Resize(sizing),
    transforms.ToTensor(),
])
    

```

## Setting Up Loaders


```python

# Setting Data Sets for Train, Test, Validation Generators
train_data = datasets.ImageFolder(
    '../input/x_ray_v3/content/x_ray/train',
    transform=train_transform
)

valid_data = datasets.ImageFolder(
    '../input/x_ray_v3/content/x_ray/validation',
    transform=valid_transform
)

test_data = datasets.ImageFolder(
    '../input/x_ray_v3/content/x_ray/test',
    transform=test_transform
)

```


```python
batch_size = 20
```


```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
```

## Validating Classes and Image Samples


```python
# Printing out the classes and assigned indexes

classes_to_idx = train_data.class_to_idx.items()
classes = []

print("--Classes & Numerical Labels--")
for key, value in classes_to_idx:
    print(value, key)
    classes.append(key)

print("\n", "No_of_classes: ", len(classes))
```


```python
def visualize(loader, classes, num_of_image=5, fig_size=(25, 5)):
    images, labels = next(iter(loader))
    
    fig = plt.figure(figsize=fig_size)
    for idx in range(num_of_image):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])

        img = images[idx]
        npimg = img.numpy()
        img = np.transpose(npimg, (1, 2, 0))  
        ax.imshow(img, cmap='gray')
        ax.set_title(classes[labels[idx]])
```


```python
visualize(train_loader, classes)

```

## Setting Up The Model


```python
# Setting up pre-trained model

model = models.resnext(pretrained=True)

```


```python
# Preventing adjustment of model weights above our custom classifier layer

for param in model.parameters():
    param.requires_grad = False
```


```python
model.classifier
```

### Setting up mila for possible use


```python

def milan(input, beta=-0.25):
    '''
    Applies the Mila function element-wise:
    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))
    See additional documentation for mila class.
    '''
    return input * torch.tanh(F.softplus(input+beta))

class mila(nn.Module):
    '''
    Applies the Mila function element-wise:
    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = mila(beta=1.0)
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self, beta=-0.25):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return milan(input, self.beta)
```


```python
class fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        x = self.logsoftmax(self.fc3(x))
        return x

```


```python
model.classifier = fc()
```

## Training The Model


```python
# setting up for possible use of GPU.
# sacrificing short code for readability.

def device():
    if torch.cuda.is_available():
        devtype = "cuda"
    else:
        devtype = "cpu"
    return torch.device(devtype)

```


```python
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.2)
device = device()

model.to(device);
```


```python
def training(model, epochs=5):
    running_loss = 0

    for epoch in range(epochs):

        print(f"EPOCH {epoch+1}/{epochs}...Training...")

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(
                    f"Complete --> "
                    f"Train loss: {running_loss/len(train_loader):.3f}.. "
                    f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(valid_loader):.3f} \n"
                )
                running_loss = 0
                model.train()
```


```python
# Actual Training of model

training(model, epochs=2)
```


```python

```
