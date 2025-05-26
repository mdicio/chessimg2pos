import logging
from functools import reduce
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from .chessboard_image import get_chessboard_tiles
from .chessclassifier import ChessPieceClassifier
from matplotlib.gridspec import GridSpec
from .utils import compressed_fen

# Set up logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .chessdataset import create_image_transforms

class ChessPositionPredictor:
    """An improved class to predict chess positions from images with better consistency"""

    def __init__(self, model_path, fen_chars="1RNBQKPrnbqkp", use_grayscale=True, verbose=False):
        """Initialize the predictor with a trained model

        Args:
            model_path: Path to the trained PyTorch model
            fen_chars: The FEN characters used during training (MUST match training configuration)
            use_grayscale: Whether to use grayscale images (must match training configuration)
            verbose: Whether to print debug information
        """
        self.fen_chars = fen_chars
        self.use_grayscale = use_grayscale
        self.verbose = verbose

        # Ensure model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            logger.info(f"Using device: {self.device}")

        # Create model and ensure it matches training configuration
        self.model = ChessPieceClassifier(
            num_classes=len(fen_chars), use_grayscale=use_grayscale
        )

        # Load model weights with error handling
        try:
            if self.verbose:
                logger.info(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Create transform with the same grayscale setting as used during training
        self.transform = create_image_transforms(use_grayscale=use_grayscale)
        
        if self.verbose:
            logger.info(f"Model initialized with FEN chars: {self.fen_chars}")
            logger.info(f"Using grayscale: {self.use_grayscale}")

    def predict_tile(self, tile_img):
        """Predict chess piece on a single tile

        Args:
            tile_img: PIL Image of a chess tile

        Returns:
            tuple: (predicted FEN char, confidence)
        """
        # Create a copy to avoid modifying the original
        tile_img_copy = tile_img.copy()
        
        # Apply grayscale conversion explicitly if needed to ensure consistency
        if self.use_grayscale:
            tile_img_copy = tile_img_copy.convert("L")
            
        # Apply transformation
        try:
            img_tensor = self.transform(tile_img_copy).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error transforming image: {str(e)}")
            # Return empty piece with low confidence as fallback
            return self.fen_chars[0], 0.0
        
        # Verify channels match model expectation
        expected_channels = 1 if self.use_grayscale else 3
        if img_tensor.shape[1] != expected_channels:
            logger.warning(f"Model expects {expected_channels} channels but got {img_tensor.shape[1]}")
            
            # Try to correct the channel mismatch
            if expected_channels == 1 and img_tensor.shape[1] == 3:
                # Convert RGB to grayscale by averaging channels
                img_tensor = img_tensor.mean(dim=1, keepdim=True)
                logger.info("Converted RGB to grayscale by averaging channels")
            elif expected_channels == 3 and img_tensor.shape[1] == 1:
                # Expand grayscale to RGB by repeating the channel
                img_tensor = img_tensor.expand(-1, 3, -1, -1)
                logger.info("Expanded grayscale to RGB by repeating the channel")

        # Get prediction
        with torch.no_grad():
            try:
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get the highest probability and its index
                max_prob, predicted_idx = torch.max(probabilities, 0)
                
                # Get top 3 predictions for debugging
                if self.verbose:
                    topk_probs, topk_indices = torch.topk(probabilities, min(3, len(self.fen_chars)))
                    top_predictions = [(self.fen_chars[idx.item()], prob.item()) for idx, prob in zip(topk_indices, topk_probs)]
                    logger.debug(f"Top 3 predictions: {top_predictions}")
                
                return self.fen_chars[predicted_idx.item()], max_prob.item()
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                return self.fen_chars[0], 0.0  # Return empty square with low confidence

    def predict_chessboard(self, image_path_or_object, return_tiles=False, fen_type = "standard"):
        """Predict chess position from an image file or PIL Image object

        Args:
            image_path_or_object: Path to the chessboard image file or PIL Image object
            return_tiles: Whether to return the detected tiles along with predictions

        Returns:
            dict: {
                'fen': predicted FEN string, 
                'confidence': overall confidence,
                'predictions': detailed predictions,
                'tiles': tile images if return_tiles=True
                'fen_type': fen notation style (standard or compressed)
            }
        """
        # Handle both file paths and PIL Image objects
        if isinstance(image_path_or_object, str):
            if not os.path.exists(image_path_or_object):
                raise FileNotFoundError(f"Image file not found: {image_path_or_object}")
            image_path = image_path_or_object
            original_image = Image.open(image_path)
        else:
            # Assume it's a PIL Image
            original_image = image_path_or_object
            image_path = None
        
        # Get tiles with the same grayscale setting as training
        try:
            tiles = get_chessboard_tiles(image_path_or_object, use_grayscale=self.use_grayscale)
        except Exception as e:
            logger.error(f"Error detecting chessboard tiles: {str(e)}")
            raise
            
        if len(tiles) != 64:
            raise ValueError(f"Expected 64 tiles, got {len(tiles)}")

        # Predict each tile
        predictions = []
        for i, tile in enumerate(tiles):
            try:
                fen_char, probability = self.predict_tile(tile)
                row = 7 - (i // 8)  # Chess board rows are inverted in FEN
                col = i % 8
                square = chr(97 + col) + str(row + 1)  # Convert to algebraic notation (a1, h8, etc.)
                predictions.append((square, fen_char, probability))
            except Exception as e:
                logger.error(f"Error predicting tile {i}: {str(e)}")
                # Use empty square as fallback
                row = 7 - (i // 8)
                col = i % 8
                square = chr(97 + col) + str(row + 1)
                predictions.append((square, self.fen_chars[0], 0.0))

        # Create the FEN string (from perspective of white)
        board_matrix = np.zeros((8, 8), dtype=object)
        for i, (square, fen_char, _) in enumerate(predictions):
            row = 7 - (ord(square[1]) - ord('1'))  # Convert back to 0-7 indices, inverted
            col = ord(square[0]) - ord('a')        # Convert a-h to 0-7 indices
            board_matrix[row, col] = fen_char
        
        # Construct FEN string (read from top to bottom)
        fen_rows = []
        for row in board_matrix:
            fen_row = "".join(row)
            fen_rows.append(fen_row)
        
        fen_notation = "/".join(fen_rows)
        if fen_type == "compressed":
            fen_out = compressed_fen(fen_notation)
        else:
            fen_out = fen_notation

        # Calculate overall confidence (product of individual confidences)
        confidence = reduce(lambda x, y: x * y, [p[2] for p in predictions])
        
        # Prepare result
        result = {
            'fen': fen_out,
            'confidence': confidence,
            'predictions': predictions,
        }
        
        if return_tiles:
            result['tiles'] = tiles
            result['original_image'] = original_image
            
        return result

    def visualize_prediction(self, result):
        """Visualize the prediction results

        Args:
            result: The result dictionary from predict_chessboard with return_tiles=True

        Returns:
            matplotlib Figure
        """
        if 'original_image' not in result or 'tiles' not in result:
            raise ValueError("The result dictionary must contain 'original_image' and 'tiles' keys. "
                            "Make sure to call predict_chessboard with return_tiles=True")
        
        chessboard_img = result['original_image']
        tiles = result['tiles']
        fen = result['fen']
        predictions = result['predictions']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1])

        # Plot original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(chessboard_img)
        ax1.set_title("Original Chessboard")
        ax1.axis("off")

        # Plot confidence heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        confidence_matrix = np.zeros((8, 8))
        for square, _, prob in predictions:
            row = ord(square[1]) - ord('1')  # Convert 1-8 to 0-7 indices
            col = ord(square[0]) - ord('a')  # Convert a-h to 0-7 indices
            confidence_matrix[7-row, col] = prob  # Invert rows for visualization

        im = ax2.imshow(confidence_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax2.set_title("Prediction Confidence")

        # Add square labels
        for i in range(8):  # Row (8->1 in chess)
            for j in range(8):  # Column (a->h in chess)
                # Find the prediction for this square
                square = chr(97 + j) + str(8 - i)  # Convert to algebraic notation
                pred = next((p for p in predictions if p[0] == square), None)
                
                if pred:
                    piece, conf = pred[1], pred[2]
                    text = f"{piece}\n{conf:.2f}"
                    ax2.text(
                        j, i, text,
                        ha="center", va="center",
                        color="black" if conf > 0.75 else "white",
                        fontsize=9,
                    )

        ax2.set_xticks(np.arange(8))
        ax2.set_yticks(np.arange(8))
        ax2.set_xticklabels(["a", "b", "c", "d", "e", "f", "g", "h"])
        ax2.set_yticklabels(["8", "7", "6", "5", "4", "3", "2", "1"])  # Reversed for chess notation
        fig.colorbar(im, ax=ax2, label="Confidence")

        # Plot FEN and overall confidence
        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(
            0.5, 0.6, f"Predicted FEN: {fen}", ha="center", va="center", fontsize=14
        )
        ax3.text(
            0.5, 0.3, f"Overall confidence: {result['confidence']:.6f}",
            ha="center", va="center", fontsize=12
        )
        ax3.text(
            0.5, 0.0, f"Lichess editor: https://lichess.org/editor/{fen}",
            ha="center", va="center", fontsize=12, color="blue"
        )
        ax3.axis("off")
        
        logger.info(f"Lichess editor: https://lichess.org/editor/{fen}")
        plt.tight_layout()
        
        return fig

    def compare_predictions(self, image_path, show_tiles=True):
        """Compare predictions with different processing methods to debug issues

        Args:
            image_path: Path to the chessboard image
            show_tiles: Whether to display individual tiles
            
        Returns:
            dict: Results of different prediction methods for comparison
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Try different grayscale settings
        original_grayscale = self.use_grayscale
        
        # 1. Standard prediction
        standard_result = self.predict_chessboard(image_path, return_tiles=True)
        logger.info(f"Standard prediction (grayscale={self.use_grayscale}): {standard_result['fen']}")
        
        # 2. Alternative grayscale setting
        self.use_grayscale = not original_grayscale
        logger.info(f"Trying with grayscale={self.use_grayscale}")
        try:
            alternative_result = self.predict_chessboard(image_path, return_tiles=True)
            logger.info(f"Alternative prediction: {alternative_result['fen']}")
        except Exception as e:
            logger.error(f"Alternative method failed: {str(e)}")
            alternative_result = None
        
        # Restore original setting
        self.use_grayscale = original_grayscale
        
        # Display tiles if requested
        if show_tiles and standard_result.get('tiles'):
            fig, axs = plt.subplots(8, 8, figsize=(10, 10))
            for i, tile in enumerate(standard_result['tiles']):
                row, col = i // 8, i % 8
                axs[row, col].imshow(tile)
                pred = standard_result['predictions'][i]
                axs[row, col].set_title(f"{pred[1]} ({pred[2]:.2f})", fontsize=8)
                axs[row, col].axis('off')
            plt.tight_layout()
            plt.show()
            
        return {
            'standard': standard_result,
            'alternative': alternative_result
        }

    def inspect_model(self):
        """Print model architecture and configuration details"""
        logger.info(f"Model architecture:\n{self.model}")
        logger.info(f"FEN characters: {self.fen_chars}")
        logger.info(f"Using grayscale: {self.use_grayscale}")
        logger.info(f"Device: {self.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return {
            'architecture': str(self.model),
            'fen_chars': self.fen_chars,
            'use_grayscale': self.use_grayscale,
            'device': str(self.device),
            'total_params': total_params,
            'trainable_params': trainable_params
        }