import glob
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .chessclassifier import ChessPieceClassifier, EnhancedChessPieceClassifier, UltraEnhancedChessPieceClassifier
from .constants import DEFAULT_CLASSIFIER, DEFAULT_FEN_CHARS, DEFAULT_USE_GRAYSCALE
from torch.utils.data import DataLoader
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .chessdataset import ChessTileDataset, create_image_transforms
from .generate_tiles import generate_tiles_from_all_chessboards

DEFAULT_RATIO = 0.7  # ratio of training vs. test data
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

class ChessRecognitionTrainer:
    def __init__(
        self,
        images_dir,
        model_path,
        fen_chars=DEFAULT_FEN_CHARS,
        use_grayscale=DEFAULT_USE_GRAYSCALE,
        train_test_ratio=DEFAULT_RATIO,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        seed=1,
        verbose=True,
        overwrite = True,
        generate_tiles = True,
        tiles_dir=None,
    ):
        self.images_dir = images_dir
        self.generate_tiles = generate_tiles
        self.tiles_dir = tiles_dir if tiles_dir is not None else os.path.join(self.images_dir, "tiles")
        self.model_path = model_path
        self.overwrite = overwrite
        self.fen_chars = fen_chars
        self.use_grayscale = use_grayscale
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return running_loss / len(train_loader), correct / total

    def _validate(self, model, test_loader, criterion):
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        return val_loss / len(test_loader), val_correct / val_total

    def evaluate_model(self, model, test_loader):
        model.eval()
        correct, total = 0, 0
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1

        test_acc = correct / total
        logger.info(f"Overall Test Accuracy: {test_acc:.4f}")
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label]
            fen_char = self.fen_chars[label]
            logger.info(
                f"Class {label}: {fen_char} Accuracy: {acc:.4f} ({class_correct[label]}/{class_total[label]})"
            )
        return test_acc

    def train(self, classifier=DEFAULT_CLASSIFIER):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.generate_tiles:
            generate_tiles_from_all_chessboards(chessboards_dir=self.images_dir,  
                                                tiles_dir=self.tiles_dir,
                                                use_grayscale=self.use_grayscale,
                                                overwrite=self.overwrite)

        all_paths = np.array(
            glob.glob(os.path.join(self.tiles_dir, "**", "*.png"), recursive=True)
        )
        if len(all_paths) == 0:
            raise ValueError(f"No PNG files found in {self.tiles_dir}/**/*.png")

        np.random.shuffle(all_paths)
        divider = int(len(all_paths) * self.train_test_ratio)
        train_paths = all_paths[:divider]
        test_paths = all_paths[divider:]

        # Enhanced transforms based on classifier type
        if classifier == "enhanced":
            train_transform = create_image_transforms(self.use_grayscale)
            val_transform = create_image_transforms(self.use_grayscale)
        else:
            train_transform = create_image_transforms(self.use_grayscale)
            val_transform = create_image_transforms(self.use_grayscale)

        train_dataset = ChessTileDataset(
            train_paths, self.fen_chars, self.use_grayscale, train_transform
        )
        test_dataset = ChessTileDataset(
            test_paths, self.fen_chars, self.use_grayscale, val_transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Model selection based on classifier type
        if classifier == "enhanced":
            model = EnhancedChessPieceClassifier(
                num_classes=len(self.fen_chars), use_grayscale=self.use_grayscale
            ).to(self.device)
        elif classifier == "ultra":
            model = UltraEnhancedChessPieceClassifier(
                num_classes=len(self.fen_chars), use_grayscale=self.use_grayscale
            ).to(self.device)
        else:
            model = ChessPieceClassifier(
                num_classes=len(self.fen_chars), use_grayscale=self.use_grayscale
            ).to(self.device)

        # Enhanced loss function
        criterion_ce = nn.CrossEntropyLoss()
        criterion_smooth = LabelSmoothingLoss(classes=len(self.fen_chars), smoothing=0.1)
        criterion_focal = FocalLoss(alpha=1, gamma=2)
        
        # Combined loss function
        def combined_criterion(pred, target):
            return (0.5 * criterion_ce(pred, target) + 
                    0.3 * criterion_smooth(pred, target) + 
                    0.2 * criterion_focal(pred, target))

        # Enhanced optimizer and scheduler
        if classifier in ["enhanced", "ultra"]:
            initial_lr = self.learning_rate*2
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=initial_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            )
            criterion = combined_criterion
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = None
            criterion = criterion_ce

        if self.verbose:
            logger.info(f"Model architecture:\n{model}")
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Using classifier: {classifier}")

        best_val_acc, best_model_state = 0.0, None

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc = self._validate(model, test_loader, criterion)

            # Step scheduler if using enhanced training
            if scheduler is not None:
                scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                logger.info(
                    f"New best model with validation accuracy: {best_val_acc:.4f}"
                )

        if best_model_state:
            model.load_state_dict(best_model_state)

        logger.info("Evaluating final model on test data:")
        test_acc = self.evaluate_model(model, test_loader)

        logger.info(f"Saving model to {self.model_path}")
        torch.save(model.state_dict(), self.model_path)

        return model, self.device, test_acc

# Training improvements
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))





# Data augmentation suggestions (implement in your data loading)
def get_enhanced_transforms():
    """Enhanced data augmentation for chess pieces"""
    from torchvision import transforms
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=(-5, 5)),  # Small rotations
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translations
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Lighting variations
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Occlusion simulation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale
    ])
    
    return train_transforms