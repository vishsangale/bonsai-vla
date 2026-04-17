import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from src.vision.vit.vit import VisionTransformer
import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract hyperparameters from config
    img_size = cfg.dataset.image_size  # single source of truth — dataset owns image_size
    patch_size = cfg.model.patch_size
    in_channels = cfg.model.channels
    num_classes = cfg.model.num_classes
    emb_dim = cfg.model.dim
    depth = cfg.model.depth
    num_heads = cfg.model.heads
    mlp_ratio = cfg.model.mlp_dim / cfg.model.dim
    dropout = cfg.model.dropout
    
    batch_size = cfg.training.batch_size
    learning_rate = float(cfg.training.learning_rate)
    epochs = cfg.training.epochs
    log_interval = cfg.training.log_interval
    checkpoint_dir = cfg.training.checkpoint_dir
    checkpoint_name = cfg.training.checkpoint_name
    log_dir = cfg.training.log_dir
    weight_decay = cfg.training.weight_decay
    num_workers = cfg.training.num_workers
    
    data_path = cfg.dataset.path

    mean = list(cfg.dataset.mean)
    std = list(cfg.dataset.std)

    # Data Augmentation and Normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Ensure data directory exists
    os.makedirs(data_path, exist_ok=True)

    print(f"Loading {cfg.dataset.name} datasets...")
    if cfg.dataset.name.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: '{cfg.dataset.name}'. Add support for it in scripts/train.py.")
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Initializing Vision Transformer...")
    model = VisionTransformer(
        img_size=img_size,
        in_channels=in_channels,
        patch_size=patch_size,
        num_classes=num_classes,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    ).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training Loop
    print("Starting training...")
    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            global_step += 1
            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                # Per-interval accuracy: how well the model did on just these batches
                interval_acc = 100 * correct / total
                print(f"[Epoch {epoch + 1}, Batch {i + 1:3d}] loss: {avg_loss:.3f} | acc: {interval_acc:.2f}%")
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/acc", interval_acc, global_step)
                running_loss = 0.0
                # Reset per-interval counters so acc reflects only the current window
                correct = 0
                total = 0

        scheduler.step()

        # Validation Loop
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        val_acc = 100 * test_correct / test_total
        val_loss = test_loss / len(testloader)
        train_acc_epoch = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("val/loss", val_loss, epoch + 1)
        writer.add_scalar("val/acc", val_acc, epoch + 1)
        writer.add_scalar("train/acc_epoch", train_acc_epoch, epoch + 1)
        writer.add_scalar("train/lr", current_lr, epoch + 1)

        print(f"--- Epoch {epoch + 1} Summary ---")
        print(f"Train Acc: {train_acc_epoch:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.3f} | LR: {current_lr:.2e}")
        print("-" * 30)

        # Save checkpoint for this epoch
        os.makedirs(checkpoint_dir, exist_ok=True)
        stem, ext = os.path.splitext(checkpoint_name)
        epoch_checkpoint_name = f"{stem}_epoch{epoch + 1:03d}{ext}"
        checkpoint_path = os.path.join(checkpoint_dir, epoch_checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Finished Training")

if __name__ == '__main__':
    main()
