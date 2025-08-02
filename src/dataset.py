
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GAFDataset(Dataset):
    """
    A PyTorch Dataset for loading GAF images.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): The root directory of the dataset (e.g., 'gaf_images/train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                image_paths.append((os.path.join(class_dir, img_name), self.class_to_idx[cls_name]))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # Example usage
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = GAFDataset(root_dir='D:/GOLDMAN SACHS PROJECT/gaf-vitrade/gaf_images/train', transform=transform)
    test_dataset = GAFDataset(root_dir='D:/GOLDMAN SACHS PROJECT/gaf-vitrade/gaf_images/test', transform=transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # You can now use these datasets with a PyTorch DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
