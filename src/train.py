import torch
from torch_geometric.loader import DataLoader
def train(model, train_data, val_data, optimizer, loss_fn, scaler):
    model.train()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    for epoch in range(1, 101):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}")