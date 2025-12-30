from torch import nn, optim
from torch.utils.data import DataLoader

def train_one_epoch(model, dataset, device="cuda"):
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"step={step}, loss={loss.item():.4f}")
