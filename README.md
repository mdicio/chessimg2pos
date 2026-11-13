# chessimg2pos — Chess Image to FEN in Python

[![PyPI version](https://img.shields.io/pypi/v/chessimg2pos.svg)](https://pypi.org/project/chessimg2pos/)
[![Python versions](https://img.shields.io/pypi/pyversions/chessimg2pos.svg)](https://pypi.org/project/chessimg2pos/)
[![Downloads](https://static.pepy.tech/badge/chessimg2pos)](https://pepy.tech/project/chessimg2pos)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![GitHub stars](https://img.shields.io/github/stars/mdicio/chessimg2pos?style=social)](https://github.com/mdicio/chessimg2pos)

**Convert a chessboard image or screenshot into a FEN string with one line of Python.**

`chessimg2pos` is a chess position recognition library: give it a picture of a 2D chessboard (a chess.com or lichess screenshot, a puzzle diagram, a book or website board image) and it returns the position in [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) notation, ready to paste into Lichess, Stockfish or `python-chess`. It uses a PyTorch CNN to classify all 64 squares, and a pretrained model downloads automatically on first use.

```bash
pip install chessimg2pos
```

```python
from chessimg2pos import predict_fen
print(predict_fen("screenshot.png"))
```

**Use cases:** analyse a position from a screenshot, digitise chess book diagrams, build chess bots and Discord/Telegram bots, generate training data, or convert chess puzzles into PGN/FEN datasets.

### Features

- 🖼️ **Image → FEN** with a single function call (`predict_fen`)
- 🎯 Trained on many board themes and piece sets from different sites
- ⚡ Pretrained weights downloaded automatically; runs on CPU or GPU
- 🔧 Full training pipeline (`ChessRecognitionTrainer`) to fine-tune on your own board styles
- 📊 Per-square confidence scores and prediction visualisation

This library vastly increases prediction accuracy over earlier approaches across a wide variety of chess image formats. For more advanced usage it also provides reusable components for training, inference, and data preparation.

Full credits to [linrock/chessboard-recognizer](https://github.com/linrock/chessboard-recognizer) for chess image data, preprocessing and basis for the training algorithm, originally a simple CNN architecture built on a no longer supported version of TensorFlow 2. This version transitions to PyTorch and vastly improves prediction accuracy on a wide variety of chess image formats.

---

## 🧪 Usage Example

Check the demo usage notebook 
for more advanced usages (training/inference) 
📓 `examples/demo_usage.ipynb`

### Predict from an image

```python
from chessimg2pos import predict_fen
fen = predict_fen("../images/chess_image.png")
print(fen)
```

### Output:

```text
8/8/8/4p1K1/2k1P3/8/8/8
```

That is standard FEN piece placement, so it works directly with Lichess, chess.com and `python-chess`:

```python
import chess
from chessimg2pos import predict_fen

fen = predict_fen("screenshot.png")
print(f"https://lichess.org/analysis/{fen}")
board = chess.Board(fen + " w - - 0 1")
```

Need one character per square instead (`1` = empty)? Pass `fen_type="expanded"`:

```python
predict_fen("screenshot.png", fen_type="expanded")
# 11111111/11111111/11111111/1111p1K1/11k1P111/11111111/11111111/11111111
```

> **Changed in 0.2.0:** earlier versions returned the expanded format by default. `fen_type="standard"` still returns it, with a deprecation warning.

Note: the image gives only piece placement. Side to move, castling rights and en passant are not in the picture, so add them yourself if needed (e.g. `fen + " w - - 0 1"`).

## 🖼️ Sample Results

<div align="center">

#### 📷 Input:
<!-- Replace the below link with your image or keep this as a placeholder -->
<img src="images/chess_image.png" width=240 />

#### 🎯 Predicted FEN:
`8/8/8/4p1K1/2k1P3/8/8/8`

</div>
---

## 🚀 Getting Started

### Requirements

- Python 3.10–3.14
- PyTorch
- Other dependencies in `requirements.txt`

```bash
pip install chessimg2pos
```
or 

```bash
git clone https://github.com/mdicio/chessimg2pos
pip install -r requirements.txt
```

## 🙏 Acknowledgements

This project is a continuation and modernization of:

- [linrock/chessboard-recognizer](https://github.com/linrock/chessboard-recognizer) — the original TensorFlow implementation
- [tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot) by [Elucidation](https://github.com/Elucidation)

Major thanks to these creators — this project wouldn’t exist without their work.

## 🧠 Core Classes

This project is centered around two powerful classes that handle training and prediction with a modern PyTorch-based architecture.

### 🔧 ChessRecognitionTrainer

Handles training and evaluation of the CNN-based chess piece classifier.

#### Example:

```python
from chessimg2pos import ChessRecognitionTrainer

trainer = ChessRecognitionTrainer(
    images_dir="../../training_images/chessboards", # replace with your path
    model_path="../../models/test_model.pt",# replace with path where you want models tgo be saved
    generate_tiles=False,  # Set to True if tiles need to be generated from boards
    epochs = 5,
    overwrite = False
)
model, device, accuracy = trainer.train(classifier="enhanced")
```

---

### 🔍 ChessPositionPredictor

Loads a trained model and predicts a FEN string from a chessboard image.

#### Example:

```python
from chessimg2pos import ChessPositionPredictor

predictor = ChessPositionPredictor("../../models/test_model.pt")
result = predictor.predict_chessboard("../images/ccom_1.png", return_tiles=True)

print("Predicted FEN:", result["fen"])
print("Confidence:", result["confidence"])
predictor.visualize_prediction(result)
```

---
