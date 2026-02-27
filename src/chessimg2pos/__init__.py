from .trainer import ChessRecognitionTrainer
from .predictor import ChessPositionPredictor
from .constants import DEFAULT_CLASSIFIER

from .model_loader import download_pretrained_model

__version__ = "0.1.4"

def predict_fen(image_path, output_type = "simple"):
    model_path = download_pretrained_model()
    predictor = ChessPositionPredictor(model_path = model_path, classifier = DEFAULT_CLASSIFIER)
    result = predictor.predict_chessboard(image_path)
    if output_type == "simple":
        return result["fen"]    
    else:
        return result