import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        # pass
        z = np.matmul(x,w) + b
        if activation == "relu":
            return round(np.maximum(z,0),5)
        elif activation == "sigmoid":
            zclipped = np.clip(z, -500, 500)
            return round(1/(1 + np.exp(-zclipped)),5)
        else:
            return round(z,5)
