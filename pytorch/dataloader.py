import torch
from torch.utils.data import Dataset, DataLoader

# create a dataset
class ToyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    
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

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# in practice, having a substantially smaller batch as the last batch in the 
# training epoch can disturb convergence during training, so we set drop_last=True
# which will drop the last batch in each epoch
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True)

# its not necessary to shuffle a test dataset
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0)

if __name__ == "__main__":
    print(len(train_ds))
    torch.manual_seed(123)

    for idx, (x, y) in enumerate(train_loader):
        print(f'Batch {idx + 1}: {x}, {y}')