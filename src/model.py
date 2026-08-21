from src.utils import *

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

    def evaluate(self, data_loader, criterion):
        """Runs validation/testing on a given DataLoader."""
        self.eval()
        running_loss, running_corrects = 0.0, 0
        total_samples = len(data_loader.dataset)
        
        device = get_device()
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

    def fit(self, train_loader, val_loader, criterion, optimizer, epochs=10, save_path="../data/best_model.pth"):
        """Trains the model and saves the best weights based on validation loss."""
        device = get_device()
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
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            # Checkpoint Save
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), save_path)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        print(f"\nTraining complete! Best model saved to {save_path}")
        

    def predict(self, image_input, mean, std):
        # Image Preprocessing (must match validation transforms)
        transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        device = get_device()
        # transform image
        input_tensor = transform(image_input).unsqueeze(0).to(device)
        
        # Run inference
        self.eval()
        with torch.no_grad():
            logits = self(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = probabilities.argmax()

        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probabilities[pred_idx] * 100)
        prob_dict = {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        '''
        print("\n================ INFERENCE RESULT ================")
        print(f"Image:            {image_path}")
        print(f"Prediction:       {predicted_label}")
        print(f"Confidence:       {confidence_score:.2f}%\n")
        
        print("Class Probabilities:")
        for class_name, prob in zip(CLASS_NAMES, probabilities):
            print(f"  - {class_name:<12}: {prob.item() * 100:.2f}%")
        '''
        return pred_class, confidence, prob_dict
        
        
    def run_and_save_evaluation(self, test_loader, output_dir="../data/summary/"):
        """Evaluates model and stores matplotlib figure, text report, and metrics dict."""
        os.makedirs(output_dir, exist_ok=True)
    
        fig_path = os.path.join(output_dir, "confusion_matrix.png")
        csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
        json_path = os.path.join(output_dir, "summary_metrics.json")
        
        device = get_device()
        self.to(device)
        self.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                images = inputs.to(device)
                outputs = self(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 1. Compute classification metrics dictionary
        report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

        # 2. Extract per-class rows into a DataFrame
        per_class_data = []
        for c_name in CLASS_NAMES:
            per_class_data.append({
                "Class Name": c_name,
                "Precision": f"{report_dict[c_name]['precision'] * 100:.2f}%",
                "Recall": f"{report_dict[c_name]['recall'] * 100:.2f}%",
                "F1-Score": f"{report_dict[c_name]['f1-score'] * 100:.2f}%",
                "Support (Total Samples)": int(report_dict[c_name]['support'])
            })
        
        metrics_df = pd.DataFrame(per_class_data)
        metrics_df.to_csv(csv_path, index=False)
    
        # 3. Compute Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES,
            ax=ax
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Test Set Confusion Matrix')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        summary_metrics = {
            "accuracy": report_dict["accuracy"] * 100,
            "macro_f1": report_dict["macro avg"]["f1-score"] * 100
        }
        with open(json_path, "w") as f:
            json.dump(summary_metrics, f, indent=4)
