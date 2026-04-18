from abc import ABC, abstractmethod

class TensorBase(ABC):
    @abstractmethod
    def __init__(self, data):
        pass
    
    @abstractmethod
    def __add__(self, other):
        pass
    
    @abstractmethod
    def __sub__(self, other):
        pass
    
    @abstractmethod
    def __mul__(self, other):
        pass
    
    @abstractmethod
    def __matmul__(self, other):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    @abstractmethod
    def reshape(self, *shape):
        pass
    
    @abstractmethod
    def transpose(self, dim0, dim1):
        pass
    
    @abstractmethod
    def mean(self, dim=None):
        pass
    
    @abstractmethod
    def sum(self, dim=None):
        pass
    
    @abstractmethod
    def max(self, dim=None):
        pass
    
    @abstractmethod
    def min(self, dim=None):
        pass
    
    @abstractmethod
    def softmax(self, dim=-1):
        pass
    
    @abstractmethod
    def relu(self):
        pass
    
    @abstractmethod
    def sigmoid(self):
        pass
    
    @abstractmethod
    def tanh(self):
        pass
    
    @abstractmethod
    def backward(self, grad=None):
        pass

class ModuleBase(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def backward(self, grad):
        pass
    
    @abstractmethod
    def parameters(self):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass

class OptimizerBase(ABC):
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass

class LossBase(ABC):
    @abstractmethod
    def __call__(self, predictions, targets):
        pass
    
    @abstractmethod
    def backward(self):
        pass

class AttentionBase(ABC):
    @abstractmethod
    def forward(self, query, key, value, mask=None):
        pass

class MoEBase(ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def route(self, x):
        pass
    
    @abstractmethod
    def expert_forward(self, x, expert_indices):
        pass

class TrainingBase(ABC):
    @abstractmethod
    def train_epoch(self, dataloader, optimizer, loss_fn):
        pass
    
    @abstractmethod
    def validate(self, dataloader, loss_fn):
        pass
    
    @abstractmethod
    def train(self, train_dataloader, val_dataloader, optimizer, loss_fn, epochs):
        pass

class DataLoaderBase(ABC):
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

class ConfigBase(ABC):
    @abstractmethod
    def load(self, config_path):
        pass
    
    @abstractmethod
    def save(self, config_path):
        pass
    
    @abstractmethod
    def get(self, key, default=None):
        pass
    
    @abstractmethod
    def set(self, key, value):
        pass

class MetricBase(ABC):
    @abstractmethod
    def compute(self, predictions, targets):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_score(self):
        pass