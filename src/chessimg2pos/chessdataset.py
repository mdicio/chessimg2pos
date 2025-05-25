from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChessTileDataset(Dataset):
    def __init__(self, image_paths, fen_chars, use_grayscale=True, transform=None):
        self.image_paths = image_paths
        self.fen_chars = fen_chars
        self.use_grayscale = use_grayscale
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        piece_type = image_path[-5]


        if piece_type not in self.fen_chars:
            print("piece not in fen", piece_type)
            print("image_path", image_path)
        # print(piece_type)
        assert piece_type in self.fen_chars
        label = self.fen_chars.index(piece_type)

        # Open image with PIL
        if self.use_grayscale:
            image = Image.open(image_path).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# def create_image_transforms(use_grayscale=True):
#     """Define the image transforms for preprocessing"""
#     if use_grayscale:
#         return transforms.Compose(
#             [
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5], std=[0.5]),
#             ]
#         )
#     else:
#         return transforms.Compose(
#             [
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ]
#         )

def create_image_transforms(use_grayscale=True):
    if use_grayscale:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # ← ensure single-channel
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
