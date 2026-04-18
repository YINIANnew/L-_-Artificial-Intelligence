import numpy as np
from .abstractions import TensorBase


class Tensor(TensorBase):

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = np.array([data])
        self.shape = self.data.shape
        self.grad = None
        self._prev = []
        self._op = None

    def __getitem__(self, key):
        result = self.data[key]
        if isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def view(self, *shape):
        result = self.data.reshape(shape)
        return Tensor(result)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @staticmethod
    def matmul(a, b):
        result = np.matmul(a.data, b.data)
        return Tensor(result)

    @staticmethod
    def randn(*shape):
        data = np.random.randn(*shape)
        return Tensor(data)

    @staticmethod
    def ones(*shape):
        data = np.ones(shape)
        return Tensor(data)

    @staticmethod
    def tril(mat):
        result = np.tril(mat.data)
        return Tensor(result)

    def softmax(self, dim=-1):
        result = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        result = result / np.sum(result, axis=dim, keepdims=True)
        return Tensor(result)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        data = self.data
        if dim < 0:
            dim = data.ndim + dim
        
        sorted_indices = np.argsort(data, axis=dim, kind='quicksort')
        if largest:
            sorted_indices = np.take(sorted_indices, np.arange(data.shape[dim]-k, data.shape[dim]), axis=dim)
        else:
            sorted_indices = np.take(sorted_indices, np.arange(k), axis=dim)
        
        if largest and sorted:
            sorted_indices = np.flip(sorted_indices, axis=dim)
        
        sorted_values = np.take_along_axis(data, sorted_indices, axis=dim)
        
        return Tensor(sorted_values), Tensor(sorted_indices)

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data + other.data)
            result._prev = [self, other]
            result._op = "add"
        else:
            result = Tensor(self.data + other)
            result._prev = [self]
            result._op = "add_const"
        return result

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data - other.data)
            result._prev = [self, other]
            result._op = "sub"
        else:
            result = Tensor(self.data - other)
            result._prev = [self]
            result._op = "sub_const"
        return result

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(np.matmul(self.data, other.data))
            result._prev = [self, other]
            result._op = "matmul"
        else:
            result = Tensor(np.matmul(self.data, other))
            result._prev = [self]
            result._op = "matmul_const"
        return result

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data * other.data)
            result._prev = [self, other]
            result._op = "mul"
        else:
            result = Tensor(self.data * other)
            result._prev = [self]
            result._op = "mul_const"
        return result

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data / other.data)
            result._prev = [self, other]
            result._op = "truediv"
        else:
            result = Tensor(self.data / other)
            result._prev = [self]
            result._op = "truediv_const"
        return result

    def masked_fill(self, mask, value):
        result = self.data.copy()
        if isinstance(mask, Tensor):
            result[mask.data] = value
        else:
            result[mask] = value
        return Tensor(result)

    def squeeze(self, dim=None):
        result = np.squeeze(self.data, axis=dim)
        return Tensor(result)

    def unsqueeze(self, dim):
        result = np.expand_dims(self.data, axis=dim)
        return Tensor(result)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            result = self.data == other.data
        else:
            result = self.data == other
        return Tensor(result)

    def any(self, dim=None):
        return np.any(self.data, axis=dim)

    def clamp(self, min=None, max=None):
        result = np.clip(self.data, min, max)
        return Tensor(result)

    def dot(self, other):
        if isinstance(other, Tensor):
            return np.dot(self.data.flatten(), other.data.flatten())
        else:
            return np.dot(self.data.flatten(), other)

    def sum(self, dim=None, keepdim=False):
        result = np.sum(self.data, axis=dim, keepdims=keepdim)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result

    def mean(self, dim=None, keepdim=False):
        result = np.mean(self.data, axis=dim, keepdims=keepdim)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result

    def expand(self, *shape):
        result = np.broadcast_to(self.data, shape)
        return Tensor(result)

    def reshape(self, *shape):
        result = self.data.reshape(shape)
        return Tensor(result)

    def transpose(self, *dims):
        if not dims:
            result = self.data.T
        elif len(dims) == 2:
            result = self.data.swapaxes(dims[0], dims[1])
        else:
            result = self.data.transpose(*dims)
        return Tensor(result)

    def flatten(self):
        result = self.data.flatten()
        return Tensor(result)

    def contiguous(self):
        if not self.data.flags.contiguous:
            return Tensor(self.data.copy())
        return self

    @property
    def dtype(self):
        return self.data.dtype

    def to(self, dtype):
        result = self.data.astype(dtype)
        return Tensor(result)

    def max(self, dim=None, keepdim=False):
        result = np.max(self.data, axis=dim, keepdims=keepdim)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result

    def min(self, dim=None, keepdim=False):
        result = np.min(self.data, axis=dim, keepdims=keepdim)
        if not keepdim and dim is not None:
            return Tensor(result)
        elif isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result

    def relu(self):
        result = np.maximum(0, self.data)
        return Tensor(result)

    def sigmoid(self):
        result = 1 / (1 + np.exp(-self.data))
        return Tensor(result)

    def tanh(self):
        result = np.tanh(self.data)
        return Tensor(result)

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._op == "add":
            self._prev[0].backward(grad)
            self._prev[1].backward(grad)
        elif self._op == "add_const":
            self._prev[0].backward(grad)
        elif self._op == "sub":
            self._prev[0].backward(grad)
            self._prev[1].backward(-grad)
        elif self._op == "sub_const":
            self._prev[0].backward(grad)
        elif self._op == "mul":
            self._prev[0].backward(grad * self._prev[1].data)
            self._prev[1].backward(grad * self._prev[0].data)
        elif self._op == "mul_const":
            self._prev[0].backward(grad * self._prev[1])
        elif self._op == "truediv":
            self._prev[0].backward(grad / self._prev[1].data)
            self._prev[1].backward(
                -grad * self._prev[0].data / (self._prev[1].data ** 2)
            )
        elif self._op == "truediv_const":
            self._prev[0].backward(grad / self._prev[1])
        elif self._op == "matmul":
            self._prev[0].backward(np.matmul(grad, self._prev[1].data.T))
            self._prev[1].backward(np.matmul(self._prev[0].data.T, grad))
