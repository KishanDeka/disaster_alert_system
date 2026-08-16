import torch
import torch.nn as nn

class ScratchCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ScratchCNN, self).__init__()
        self.features = nn.Sequential(
            # CNN Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            # CNN Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32
            
            # CNN Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 -> 16
            
            # CNN Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 16 -> 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def evaluate(self, data_loader, criterion, device):
        """Runs validation/testing on a given DataLoader."""
        self.eval()
        running_loss, running_corrects = 0.0, 0
        total_samples = len(data_loader.dataset)

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels).item()

        loss = running_loss / total_samples
        acc = running_corrects / total_samples
        return loss, acc

    def fit(self, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda', save_path="best_model.pth"):
        """Trains the model and saves the best weights based on validation loss."""
        self.to(device)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training Phase
            self.train()
            running_loss, running_corrects = 0.0, 0
            train_total = len(train_loader.dataset)

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels).item()

            train_loss = running_loss / train_total
            train_acc = running_corrects / train_total

            # Validation Phase
            val_loss, val_acc = self.evaluate(val_loader, criterion, device)

            # Checkpoint Save
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), save_path)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        print(f"\nTraining complete! Best model saved to {save_path}")
