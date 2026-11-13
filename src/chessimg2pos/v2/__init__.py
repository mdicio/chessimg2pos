"""Model v2: board locator + whole-board classifier + legality step.

PredictorV2 needs PyTorch; OnnxPredictorV2 needs only onnxruntime."""

__all__ = ["PredictorV2", "OnnxPredictorV2"]


def __getattr__(name):
    if name == "PredictorV2":
        from .predictor import PredictorV2
        return PredictorV2
    if name == "OnnxPredictorV2":
        from .onnx_predictor import OnnxPredictorV2
        return OnnxPredictorV2
    raise AttributeError(name)
