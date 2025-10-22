import torch.nn.functional as F
import torch
import time

from torch.utils.data import DataLoader
from multilayer_nn import NeuralNetwork
from dataloader import ToyDataset

start_time = time.time()

# create dummy data
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])
y_test = torch.tensor([0, 1])

# create dataset and dataloaders
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0)

torch.manual_seed(123)
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)
print(f'Model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)
num_epochs = 3 

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch + 1:03d}/{num_epochs:03d}'
              f'| Batch {batch_idx:03d}/{len(train_loader):03d}'
              f'| Train Loss: {loss: .2f}')

# switch to eval for inference and to eval prediction
model.eval()
with torch.no_grad():
    outputs = model(X_train.to(device))
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

predictions = torch.argmax(probas, dim=1)
print(predictions)

predictions = torch.argmax(outputs, dim=1)
print(predictions)

print(predictions == y_train.to(device))

def compute_accuracy(model, data_loader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(data_loader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    
    return (correct / total_examples).item()

print(f'Compute accuracy')
print(f'Train accuracy {compute_accuracy(model, train_loader)}')
print(f'Test accuracy {compute_accuracy(model, test_loader)}')

end_time = time.time()
print(end_time - start_time)

'''print(f'Save model')
torch.save(model.state_dict(), "model.pth")

print(f'Load model')
model = NeuralNetwork(num_inputs=2, num_outputs=2)
model.load_state_dict(torch.load("model.pth"))'''