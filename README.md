# 🧠 Chessboard Recognizer (Convert your chess images to FEN positions with one click!)

This project uses a deep learning model implemented in PyTorch to recognize the positions of chess pieces on a chessboard image and convert it into [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) notation. This library introduces an easy and fast function to simply predict a fen from an image. For more advanced usage it also provides reusable components for training, inference, and data preparation.

Full credits to [linrock/chessboard-recognizer](https://github.com/linrock/chessboard-recognizer) for chess image data, preprocessing and training algorithm, originally built on a no longer supported version of TensorFlow 2. This version transitions to PyTorch, 

---

## 🧪 Usage Example

Check the demo usage notebook 
for more advanced usages (training/inference) 
📓 `examples/demo_usage.ipynb`

### Predict from an image

```python
from recognizer import predict_fen
fen = predict_fen("../images/chess_image.png")
print(fen)
```

### Output:

```text
3rkb1r/1pp2ppp/2n1q1n1/p3Pb2/2Pp4/PN3NB1/1P1QPPPP/3RKB1R
```

## 🖼️ Sample Results

<div align="center">

#### 📷 Input:
<!-- Replace the below link with your image or keep this as a placeholder -->
<img src="images/chess_image.png" width=240 />

#### 🎯 Predicted FEN:
`3rkb1r/1pp2ppp/2n1q1n1/p3Pb2/2Pp4/PN3NB1/1P1QPPPP/3RKB1R`

</div>
---

## 🚀 Getting Started

### Requirements

- Python 3.10+
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
model, device, accuracy = trainer.train()
```

---

### 🔍 ImprovedChessPositionPredictor

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
