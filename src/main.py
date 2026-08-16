from utils import *
from data_process import CSVDataset
from model import ScratchCNN

if __name__ == "__main__" :
    csv_file_path = CSVDataset.prepare_dataset(target_dir="../data")

    # 2. Calculate channel statistics
    dataset_mean, dataset_std = CSVDataset.calculate_mean_and_std(
        csv_file=csv_file_path, 
        split='train', 
        img_size=(128, 128)
    )

    # 3. Instantiate training dataset
    train_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    train_dataset = CSVDataset(csv_file=csv_file_path, split='train', transform=train_transform)
    print(f"Loaded train dataset with {len(train_dataset)} samples.")

        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScratchCNN(num_classes=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run full training pipeline inside the class
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=10,
        device=device,
        save_path="../data/best_model.pth"
    )    
