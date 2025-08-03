import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os

from dataset import GAFDataset

def train_model(model, train_loader, test_loader, epochs, learning_rate, device, model_save_dir):
    """
    Trains the Vision Transformer model and saves a checkpoint at each epoch.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    model.to(device)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Save a checkpoint after each epoch
        checkpoint_path = os.path.join(model_save_dir, f'vit_model_epoch_{epoch+1}_acc_{accuracy:.2f}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Also save the best model separately
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(model_save_dir, 'vit_model_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved to {best_model_path} with accuracy: {best_accuracy:.2f}% ***")

    print("\nTraining complete.")

if __name__ == '__main__':
    # Hyperparameters
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    IMAGE_SIZE = 224 # ViT expects 224x224 images

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = GAFDataset(root_dir='D:/gaf-vitrade/gaf_images/train', transform=transform)
    test_dataset = GAFDataset(root_dir='D:/gaf-vitrade/gaf_images/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=len(train_dataset.classes), ignore_mismatched_sizes=True)

    # Train the model
    train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, device, model_save_dir='D:/gaf-vitrade/models')
