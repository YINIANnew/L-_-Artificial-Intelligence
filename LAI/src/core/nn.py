import math
import numpy as np
from .tensor import Tensor
from .abstractions import ModuleBase


class nn:

    class Module(ModuleBase):

        def __init__(self):
            self._parameters = {}
            self._modules = {}

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def backward(self, grad):
            pass

        def parameters(self):
            params = []
            for name, param in self._parameters.items():
                params.append(param)
            for name, module in self._modules.items():
                params.extend(module.parameters())
            return params

        def zero_grad(self):
            for name, param in self._parameters.items():
                if param.grad is not None:
                    param.grad = None
            for name, module in self._modules.items():
                module.zero_grad()

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def __setattr__(self, name, value):
            if isinstance(value, Tensor):
                self._parameters[name] = value
            elif isinstance(value, self.__class__):
                self._modules[name] = value
            super().__setattr__(name, value)

    class ReLU:

        def __call__(self, x):
            return F.relu(x)

    class GELU:

        def __call__(self, x):
            import math
            erf = np.vectorize(math.erf)
            sqrt = np.vectorize(math.sqrt)
            return Tensor(0.5 * x.data * (1 + erf(x.data / sqrt(2))))

    class Tanh:

        def __call__(self, x):
            return Tensor(np.tanh(x.data))

    class Sigmoid:

        def __call__(self, x):
            return Tensor(1.0 / (1.0 + np.exp(-x.data)))

    class Linear:

        def __init__(self, in_features, out_features, bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias

            self.weight = Tensor.randn(out_features, in_features).data
            if bias:
                self.bias = Tensor.randn(out_features).data

        def forward(self, x):
            if len(x.data.shape) == 3:
                batch_size, seq_len, _ = x.data.shape
                x_reshaped = x.data.reshape(-1, self.in_features)
                result = np.dot(x_reshaped, self.weight.T)
                if self.bias is not None:
                    result += self.bias
                result = result.reshape(batch_size, seq_len, self.out_features)
                return Tensor(result)
            else:
                result = np.dot(x.data, self.weight.T)
                if self.bias is not None:
                    result += self.bias
                return Tensor(result)

        def __call__(self, x):
            return self.forward(x)

    class LayerNorm:

        def __init__(self, normalized_shape, eps=1e-5):
            self.normalized_shape = normalized_shape
            self.eps = eps
            self.weight = np.ones(normalized_shape)
            self.bias = np.zeros(normalized_shape)

        def forward(self, x):
            if len(x.shape) == 3:
                mean = np.mean(x.data, axis=-1, keepdims=True)
                var = np.var(x.data, axis=-1, keepdims=True)
                normalized = (x.data - mean) / np.sqrt(var + self.eps)
                scaled = normalized * self.weight + self.bias
                return Tensor(scaled)
            else:
                mean = np.mean(x.data, axis=-1, keepdims=True)
                var = np.var(x.data, axis=-1, keepdims=True)
                normalized = (x.data - mean) / np.sqrt(var + self.eps)
                scaled = normalized * self.weight + self.bias
                return Tensor(scaled)

        def __call__(self, x):
            return self.forward(x)

    class Dropout:

        def __init__(self, p=0.5):
            self.p = p

        def forward(self, x):
            if self.p == 0:
                return x
            mask = np.random.rand(*x.shape) > self.p
            return Tensor(x.data * mask / (1 - self.p))

        def __call__(self, x):
            return self.forward(x)

    class Sequential:

        def __init__(self, *layers):
            self.layers = layers

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def __call__(self, x):
            return self.forward(x)


class F:

    @staticmethod
    def softmax(x, dim=-1):
        return x.softmax(dim)

    @staticmethod
    def relu(x):
        return Tensor(np.maximum(0, x.data))

    @staticmethod
    def normalize(x, dim=-1):
        norm = np.linalg.norm(x.data, axis=dim, keepdims=True)
        return Tensor(x.data / norm)

    @staticmethod
    def tanh(x):
        return Tensor(np.tanh(x.data))

    @staticmethod
    def sigmoid(x):
        return Tensor(1.0 / (1.0 + np.exp(-x.data)))

    @staticmethod
    def matmul(a, b):
        return Tensor.matmul(a, b)
