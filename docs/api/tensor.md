# PyTorchTensor
```python
PyTorchTensor(self, raw:'torch.Tensor')
```

# TensorFlowTensor
```python
TensorFlowTensor(self, raw:'tf.Tensor')
```

# JAXTensor
```python
JAXTensor(self, raw:'np.ndarray')
```

# NumPyTensor
```python
NumPyTensor(self, raw:'np.ndarray')
```

# Tensor
```python
Tensor(self, raw) -> None
```
Base class defining the common interface of all EagerPy Tensors
## __init__
```python
Tensor.__init__(self, raw) -> None
```

## raw

## dtype

## __repr__
```python
Tensor.__repr__(self:~TensorType) -> str
```

## __format__
```python
Tensor.__format__(self:~TensorType, format_spec) -> str
```

## __getitem__
```python
Tensor.__getitem__(self:~TensorType, index) -> ~TensorType
```

## __bool__
```python
Tensor.__bool__(self:~TensorType) -> bool
```

## __len__
```python
Tensor.__len__(self:~TensorType) -> int
```

## __abs__
```python
Tensor.__abs__(self:~TensorType) -> ~TensorType
```

## __neg__
```python
Tensor.__neg__(self:~TensorType) -> ~TensorType
```

## __add__
```python
Tensor.__add__(self:~TensorType, other) -> ~TensorType
```

## __radd__
```python
Tensor.__radd__(self:~TensorType, other) -> ~TensorType
```

## __sub__
```python
Tensor.__sub__(self:~TensorType, other) -> ~TensorType
```

## __rsub__
```python
Tensor.__rsub__(self:~TensorType, other) -> ~TensorType
```

## __mul__
```python
Tensor.__mul__(self:~TensorType, other) -> ~TensorType
```

## __rmul__
```python
Tensor.__rmul__(self:~TensorType, other) -> ~TensorType
```

## __truediv__
```python
Tensor.__truediv__(self:~TensorType, other) -> ~TensorType
```

## __rtruediv__
```python
Tensor.__rtruediv__(self:~TensorType, other) -> ~TensorType
```

## __floordiv__
```python
Tensor.__floordiv__(self:~TensorType, other) -> ~TensorType
```

## __rfloordiv__
```python
Tensor.__rfloordiv__(self:~TensorType, other) -> ~TensorType
```

## __mod__
```python
Tensor.__mod__(self:~TensorType, other) -> ~TensorType
```

## __lt__
```python
Tensor.__lt__(self:~TensorType, other) -> ~TensorType
```

## __le__
```python
Tensor.__le__(self:~TensorType, other) -> ~TensorType
```

## __eq__
```python
Tensor.__eq__(self:~TensorType, other) -> ~TensorType
```

## __ne__
```python
Tensor.__ne__(self:~TensorType, other) -> ~TensorType
```

## __gt__
```python
Tensor.__gt__(self:~TensorType, other) -> ~TensorType
```

## __ge__
```python
Tensor.__ge__(self:~TensorType, other) -> ~TensorType
```

## __pow__
```python
Tensor.__pow__(self:~TensorType, exponent) -> ~TensorType
```

## sign
```python
Tensor.sign(self:~TensorType) -> ~TensorType
```

## sqrt
```python
Tensor.sqrt(self:~TensorType) -> ~TensorType
```

## tanh
```python
Tensor.tanh(self:~TensorType) -> ~TensorType
```

## float32
```python
Tensor.float32(self:~TensorType) -> ~TensorType
```

## where
```python
Tensor.where(self:~TensorType, x, y) -> ~TensorType
```

## matmul
```python
Tensor.matmul(self:~TensorType, other) -> ~TensorType
```

## ndim

## numpy
```python
Tensor.numpy(self:~TensorType) -> Any
```

## item
```python
Tensor.item(self:~TensorType) -> Union[int, float]
```

## shape

## reshape
```python
Tensor.reshape(self:~TensorType, shape) -> ~TensorType
```

## astype
```python
Tensor.astype(self:~TensorType, dtype) -> ~TensorType
```

## clip
```python
Tensor.clip(self:~TensorType, min_, max_) -> ~TensorType
```

## square
```python
Tensor.square(self:~TensorType) -> ~TensorType
```

## arctanh
```python
Tensor.arctanh(self:~TensorType) -> ~TensorType
```

## sum
```python
Tensor.sum(self:~TensorType, axis=None, keepdims=False) -> ~TensorType
```

## mean
```python
Tensor.mean(self:~TensorType, axis=None, keepdims=False) -> ~TensorType
```

## min
```python
Tensor.min(self:~TensorType, axis=None, keepdims=False) -> ~TensorType
```

## max
```python
Tensor.max(self:~TensorType, axis=None, keepdims=False) -> ~TensorType
```

## minimum
```python
Tensor.minimum(self:~TensorType, other) -> ~TensorType
```

## maximum
```python
Tensor.maximum(self:~TensorType, other) -> ~TensorType
```

## argmin
```python
Tensor.argmin(self:~TensorType, axis=None) -> ~TensorType
```

## argmax
```python
Tensor.argmax(self:~TensorType, axis=None) -> ~TensorType
```

## argsort
```python
Tensor.argsort(self:~TensorType, axis=-1) -> ~TensorType
```

## uniform
```python
Tensor.uniform(self:~TensorType, shape, low=0.0, high=1.0) -> ~TensorType
```

## normal
```python
Tensor.normal(self:~TensorType, shape, mean=0.0, stddev=1.0) -> ~TensorType
```

## ones
```python
Tensor.ones(self:~TensorType, shape) -> ~TensorType
```

## zeros
```python
Tensor.zeros(self:~TensorType, shape) -> ~TensorType
```

## ones_like
```python
Tensor.ones_like(self:~TensorType) -> ~TensorType
```

## zeros_like
```python
Tensor.zeros_like(self:~TensorType) -> ~TensorType
```

## full_like
```python
Tensor.full_like(self:~TensorType, fill_value) -> ~TensorType
```

## onehot_like
```python
Tensor.onehot_like(self:~TensorType, indices, *, value=1) -> ~TensorType
```

## from_numpy
```python
Tensor.from_numpy(self:~TensorType, a) -> ~TensorType
```

## transpose
```python
Tensor.transpose(self:~TensorType, axes=None) -> ~TensorType
```

## bool
```python
Tensor.bool(self:~TensorType) -> ~TensorType
```

## all
```python
Tensor.all(self:~TensorType, axis=None, keepdims=False) -> ~TensorType
```

## any
```python
Tensor.any(self:~TensorType, axis=None, keepdims=False) -> ~TensorType
```

## logical_and
```python
Tensor.logical_and(self:~TensorType, other) -> ~TensorType
```

## logical_or
```python
Tensor.logical_or(self:~TensorType, other) -> ~TensorType
```

## logical_not
```python
Tensor.logical_not(self:~TensorType) -> ~TensorType
```

## exp
```python
Tensor.exp(self:~TensorType) -> ~TensorType
```

## log
```python
Tensor.log(self:~TensorType) -> ~TensorType
```

## log2
```python
Tensor.log2(self:~TensorType) -> ~TensorType
```

## log10
```python
Tensor.log10(self:~TensorType) -> ~TensorType
```

## log1p
```python
Tensor.log1p(self:~TensorType) -> ~TensorType
```

## tile
```python
Tensor.tile(self:~TensorType, multiples) -> ~TensorType
```

## softmax
```python
Tensor.softmax(self:~TensorType, axis=-1) -> ~TensorType
```

## log_softmax
```python
Tensor.log_softmax(self:~TensorType, axis=-1) -> ~TensorType
```

## squeeze
```python
Tensor.squeeze(self:~TensorType, axis=None) -> ~TensorType
```

## expand_dims
```python
Tensor.expand_dims(self:~TensorType, axis=None) -> ~TensorType
```

## full
```python
Tensor.full(self:~TensorType, shape, value) -> ~TensorType
```

## index_update
```python
Tensor.index_update(self:~TensorType, indices, values) -> ~TensorType
```

## arange
```python
Tensor.arange(self:~TensorType, start, stop=None, step=None) -> ~TensorType
```

## cumsum
```python
Tensor.cumsum(self:~TensorType, axis=None) -> ~TensorType
```

## flip
```python
Tensor.flip(self:~TensorType, axis=None) -> ~TensorType
```

## meshgrid
```python
Tensor.meshgrid(self:~TensorType, *tensors, indexing='xy') -> Tuple[~TensorType, ...]
```

## pad
```python
Tensor.pad(self:~TensorType, paddings, mode='constant', value=0) -> ~TensorType
```

## isnan
```python
Tensor.isnan(self:~TensorType) -> ~TensorType
```

## isinf
```python
Tensor.isinf(self:~TensorType) -> ~TensorType
```

## crossentropy
```python
Tensor.crossentropy(self:~TensorType, labels:~TensorType) -> ~TensorType
```

## T

## abs
```python
Tensor.abs(self:~TensorType) -> ~TensorType
```

## pow
```python
Tensor.pow(self:~TensorType, exponent) -> ~TensorType
```

## value_and_grad
```python
Tensor.value_and_grad(self:~TensorType, f, *args, **kwargs) -> Tuple[~TensorType, ~TensorType]
```

## value_aux_and_grad
```python
Tensor.value_aux_and_grad(self:~TensorType, f, *args, **kwargs) -> Tuple[~TensorType, Any, ~TensorType]
```

