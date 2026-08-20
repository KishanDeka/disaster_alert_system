import argparse

# Imports from your src package
from src.utils import *
from src.dataset import CSVDataset
from src.model import ScratchCNN


def run_pipeline(data_dir, epochs, batch_size, lr, weights_path):
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Running pipeline on device: {device}")

    # Prepare dataset and index CSV
    print("\n--- Pre-processing data ---")
    csv_file = os.path.join(data_dir, 'data_list.csv')
    csvdata = CSVDataset(csv_file)
    csvdata.prepare_dataset(target_dir=os.path.abspath(data_dir))

    # Calculate dynamic mean and std for normalization
    print("Calculating dataset mean and standard deviation")
    mean, std = CSVDataset.calculate_mean_and_std(
        csv_file=os.path.abspath(csv_file), 
        split='train', 
        img_size=(128, 128)
    )
    
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    
    # Save to CSV
    pd.DataFrame({
        'channel': ['R', 'G', 'B'],
        'mean': mean,
        'std': std
        }).to_csv(os.path.join(data_dir, "data_stats.csv"), index=False)
    
    # Define Transforms
    train_transform = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    val_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    # Initialize Datasets & DataLoaders
    train_dataset = CSVDataset(csv_file=csv_file, split='train', transform=train_transform)
    val_dataset = CSVDataset(csv_file=csv_file, split='val', transform=val_transform)
    test_dataset = CSVDataset(csv_file=csv_file, split='test', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # Instantiate Model, Loss Function, and Optimizer
    print("\n--- Model Initialization ---")
    model = ScratchCNN(num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train Model
    print("\n--- Training ---")
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        save_path=weights_path
    )

    # Run evaluation and save to files
    print("\n--- Evaluation ---")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.run_and_save_evaluation(
            test_loader=test_loader,
            output_dir="data/summary/"
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Satellite Disaster Classification CNN")
    parser.add_argument("--data_dir", type=str, default="data/", help="Target directory for dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weights_path", type=str, default="data/model/best_model.pth", help="Output path for best model weights")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weights_path=args.weights_path
    )
    
    
