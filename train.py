import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Local modules' import

from models.resnet2 import ResNet18
from dataset.dataset import prepare_data, get_dataloaders

WANDB_KEY = "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"

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\n ---- {epoch} ----")

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        # Move data to GPU/CPU

        inputs,targets = inputs.to(device), targets.to(device)

        # Optimization steps

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistics

        running_loss += loss.item()
        _ , predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx%100 == 0:
            print(f" batch: {batch_idx}/{len(train_loader)} ; loss: {loss.item(): .4f}")

    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    print(f'Train Results: Loss: {avg_loss:.6f} | Acc: {acc:.2f}%')
        
    return avg_loss, acc
    
def validate(model, val_loader, criterion, device):   

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    print(f'Validation Results: Loss: {avg_loss:.6f} | Acc: {acc:.2f}%')

    return acc

def main(): 
     
    # 1. Setup device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Prepare data

    prepare_data()

    train_loader, val_loader = get_dataloaders(batch_size=64)

    # 3. Model initialization

    model = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr= 0.001, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, "max", patience=3,)

    # I' monitoring validation accuracy. I set scheduler criterion on "max", since the 
    # bigger the val accuracy's values, the better.

    # 4. Initialize wandb

    wandb.init(

        project="naive-resnet18-tiny-imagenet",

        config={
            "learning_rate": 0.001,
            "architecture": "ResNet18",
            "dataset": "TinyImageNet",
            "epochs": 10,
            "batch_size": 64,
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "scheduler_patience": 3,
            "scheduler_factor": 0.1

        }
    )

    best_acc = 0

    num_epochs = 10

    for epoch in range(1,num_epochs+1):
        
        train_loss, train_acc = train_one_epoch(epoch,model,train_loader,criterion,optimizer,device)

        val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            print(f"new best accuracy")
            best_acc = val_acc
            torch.save(model.state_dict(),'model_best.pth')


    #   Log metrics every epoch
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

    print(f"\n Training completed. Best validation accuracy: {best_acc}")

if __name__ == "__main__":
    main()