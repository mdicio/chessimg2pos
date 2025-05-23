# 🧠 Chessboard Recognizer (Modernized PyTorch Edition)

This project uses a deep learning model implemented in PyTorch to recognize the positions of chess pieces on a chessboard image and convert it into [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) notation.

It is a modernized and modular adaptation of [linrock/chessboard-recognizer](https://github.com/linrock/chessboard-recognizer), originally built on TensorFlow 2. This version transitions to PyTorch, introduces notebook-based workflows, and provides reusable components for training, inference, and data preparation.

---

## 🧪 Usage Example

A typical workflow (including training and inference) is provided in the notebook:  
📓 `notebooks/demo_usage.ipynb` *(rename and place your actual notebook here)*

### Predict from an image

```python
from recognizer import predict_fen
fen = predict_fen("path/to/chessboard.png")
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
<img src="INSERT_YOUR_IMAGE_LINK_HERE" width=240 />

#### 🎯 Predicted FEN:
`INSERT_PREDICTED_FEN_HERE` (XX.XXX% confidence)

</div>

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch
- Other dependencies in `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Pretrained Model

You can:

- Download the pretrained model from [INSERT_YOUR_MODEL_LINK_HERE]
- Or train your own model from scratch using provided tools.

Place the pretrained model at:

```
models/
  └── your_model_file.pt
```

---

## 🏋️‍♀️ Training

Prepare a training set:

- Use our provided [dataset](INSERT_YOUR_TRAINING_DATASET_LINK_HERE) or
- Generate your own with:

```bash
python scripts/generate_chessboards.py
python scripts/generate_tiles.py
```

Train the model:

```bash
python train.py
```

---

## 🛠️ Tools

- `view_images.py`: Debug 32x32 tile alignment with source chessboard
- `save_chessboard.py chessboard.png <subdirectory> "<actual_fen>"`: Save misclassified examples to improve training
- `debug.py`: Visualize prediction confidences and FEN accuracy

---

## 📂 Directory Layout (Suggestion)

```
project-root/
│
├── models/
│   └── your_model.pt
├── notebooks/
│   └── demo_usage.ipynb
├── data/
│   └── training_images/
├── scripts/
│   ├── generate_chessboards.py
│   ├── generate_tiles.py
│   ├── view_images.py
│   ├── save_chessboard.py
│   └── debug.py
└── recognizer/
    └── __init__.py, model.py, utils.py, etc.
```

---

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
from trainer import ChessRecognitionTrainer

trainer = ChessRecognitionTrainer(
    images_dir="data/training_images",
    model_path="models/chess_model.pt",
    generate_tiles=True  # Set to True if tiles need to be generated from boards
)
model, device, accuracy = trainer.train()
```

---

### 🔍 ImprovedChessPositionPredictor

Loads a trained model and predicts a FEN string from a chessboard image.

#### Example:

```python
from predictor import ImprovedChessPositionPredictor

predictor = ImprovedChessPositionPredictor("models/chess_model.pt")
result = predictor.predict_chessboard("test_images/chessboard.png", return_tiles=True)

print("Predicted FEN:", result["fen"])
print("Confidence:", result["confidence"])
predictor.visualize_prediction(result)
```

---

## 🧪 Usage Example

This project supports both training from scratch and inference using pretrained models with clean, modular interfaces.

### 🏋️‍♂️ Training a Model

Train a chess piece classifier using your dataset of board images:

```python
from trainer import ChessRecognitionTrainer

trainer = ChessRecognitionTrainer(
    images_dir="data/training_images",  # Path to chessboard images
    model_path="models/chess_model.pt",
    generate_tiles=True  # Automatically tile the boards
)
model, device, accuracy = trainer.train()
print(f"Training complete. Test accuracy: {accuracy:.4f}")
```

---

### 🧠 Predicting from an Image

Use a trained model to predict the FEN string from a chessboard image:

```python
from predictor import ImprovedChessPositionPredictor

predictor = ImprovedChessPositionPredictor("models/chess_model.pt")
result = predictor.predict_chessboard("test_images/sample_board.png", return_tiles=True)

print("Predicted FEN:", result["fen"])
print("Confidence:", result["confidence"])
predictor.visualize_prediction(result)
```

---

### 🖼️ Sample Results

Below are example visualizations of prediction outputs.

#### 📷 Input Image:
![Input Chessboard](images/sample_chessboard.png)

#### 🧠 Prediction:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
```

#### 🔗 Lichess Editor:
[https://lichess.org/editor/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR](https://lichess.org/editor/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR)

#### 🔍 Confidence Visualization:
![Prediction Confidence Heatmap](images/confidence_heatmap.png)

---

### 📸 Image Template for README

To include your own examples, save your images and reference them like this:

```markdown
![Your Image Title](images/your_image_name.png)
```

Place the image in an `images/` folder at the root of your repository. For example:

- `images/sample_chessboard.png`
- `images/confidence_heatmap.png`
