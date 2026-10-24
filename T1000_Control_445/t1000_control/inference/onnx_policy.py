from __future__ import annotations

import numpy as np


class OnnxPolicy:

    def __init__(self, model_path, *, providers=None):
        import onnxruntime as ort

        if providers is None:
            self.session = ort.InferenceSession(str(model_path))
        else:
            self.session = ort.InferenceSession(str(model_path), providers=providers)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.session.run(None, {'obs': observation.reshape(1, -1)})[0][0]
