import glob
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from .chessclassifier import ChessPieceClassifier
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
DEFAULT_USE_GRAYSCALE = True

class ChessRecognitionTrainer:
    def __init__(
        self,
        images_dir,
        model_path,
        fen_chars="1RNBQKPrnbqkp",
        use_grayscale=DEFAULT_USE_GRAYSCALE,
        train_test_ratio=DEFAULT_RATIO,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        seed=1,
        verbose=True,
        overwrite = True,
        generate_tiles = True

    ):
        self.images_dir = images_dir
        self.generate_tiles = generate_tiles
        self.tiles_dir = os.path.join(os.path.dirname(self.images_dir), "tiles")
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
            logger.info(
                f"Class {label} Accuracy: {acc:.4f} ({class_correct[label]}/{class_total[label]})"
            )
        return test_acc

    def train(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        if self.generate_tiles:
            generate_tiles_from_all_chessboards(chessboards_dir = self.images_dir,  
                                                tiles_dir = self.tiles_dir,
                                                use_grayscale = self.use_grayscale,
                                                  overwrite = self.overwrite)

        all_paths = np.array(glob.glob(f"{self.tiles_dir}/*/*.png"))
        if len(all_paths) == 0:
            raise ValueError(f"No PNG files found in {self.tiles_dir}/*/*.png")

        np.random.shuffle(all_paths)
        divider = int(len(all_paths) * self.train_test_ratio)
        train_paths = all_paths[:divider]
        test_paths = all_paths[divider:]

        transform = create_image_transforms(self.use_grayscale)
        train_dataset = ChessTileDataset(
            train_paths, self.fen_chars, self.use_grayscale, transform
        )
        test_dataset = ChessTileDataset(
            test_paths, self.fen_chars, self.use_grayscale, transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        model = ChessPieceClassifier(
            num_classes=len(self.fen_chars), use_grayscale=self.use_grayscale
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.verbose:
            logger.info(f"Model architecture:\n{model}")
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")

        best_val_acc, best_model_state = 0.0, None

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc = self._validate(model, test_loader, criterion)

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
