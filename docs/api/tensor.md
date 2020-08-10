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
Tensor(self, raw:Any)
```
Base class defining the common interface of all EagerPy Tensors
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
Tensor.where(self:~TensorType, x:Union[_ForwardRef('Tensor'), int, float], y:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## matmul
```python
Tensor.matmul(self:~TensorType, other:~TensorType) -> ~TensorType
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
Tensor.reshape(self:~TensorType, shape:Union[Tuple[int, ...], int]) -> ~TensorType
```

## take_along_axis
```python
Tensor.take_along_axis(self:~TensorType, index:~TensorType, axis:int) -> ~TensorType
```

## astype
```python
Tensor.astype(self:~TensorType, dtype:Any) -> ~TensorType
```

## clip
```python
Tensor.clip(self:~TensorType, min_:float, max_:float) -> ~TensorType
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
Tensor.sum(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## prod
```python
Tensor.prod(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## mean
```python
Tensor.mean(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## min
```python
Tensor.min(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## max
```python
Tensor.max(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## minimum
```python
Tensor.minimum(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## maximum
```python
Tensor.maximum(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## argmin
```python
Tensor.argmin(self:~TensorType, axis:Union[int, NoneType]=None) -> ~TensorType
```

## argmax
```python
Tensor.argmax(self:~TensorType, axis:Union[int, NoneType]=None) -> ~TensorType
```

## argsort
```python
Tensor.argsort(self:~TensorType, axis:int=-1) -> ~TensorType
```

## uniform
```python
Tensor.uniform(self:~TensorType, shape:Union[Tuple[int, ...], int], low:float=0.0, high:float=1.0) -> ~TensorType
```

## normal
```python
Tensor.normal(self:~TensorType, shape:Union[Tuple[int, ...], int], mean:float=0.0, stddev:float=1.0) -> ~TensorType
```

## ones
```python
Tensor.ones(self:~TensorType, shape:Union[Tuple[int, ...], int]) -> ~TensorType
```

## zeros
```python
Tensor.zeros(self:~TensorType, shape:Union[Tuple[int, ...], int]) -> ~TensorType
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
Tensor.full_like(self:~TensorType, fill_value:float) -> ~TensorType
```

## onehot_like
```python
Tensor.onehot_like(self:~TensorType, indices:~TensorType, *, value:float=1) -> ~TensorType
```

## from_numpy
```python
Tensor.from_numpy(self:~TensorType, a:Any) -> ~TensorType
```

## transpose
```python
Tensor.transpose(self:~TensorType, axes:Union[Tuple[int, ...], NoneType]=None) -> ~TensorType
```

## bool
```python
Tensor.bool(self:~TensorType) -> ~TensorType
```

## all
```python
Tensor.all(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## any
```python
Tensor.any(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## logical_and
```python
Tensor.logical_and(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## logical_or
```python
Tensor.logical_or(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
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
Tensor.tile(self:~TensorType, multiples:Tuple[int, ...]) -> ~TensorType
```

## softmax
```python
Tensor.softmax(self:~TensorType, axis:int=-1) -> ~TensorType
```

## log_softmax
```python
Tensor.log_softmax(self:~TensorType, axis:int=-1) -> ~TensorType
```

## squeeze
```python
Tensor.squeeze(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None) -> ~TensorType
```

## expand_dims
```python
Tensor.expand_dims(self:~TensorType, axis:int) -> ~TensorType
```

## full
```python
Tensor.full(self:~TensorType, shape:Union[Tuple[int, ...], int], value:float) -> ~TensorType
```

## index_update
```python
Tensor.index_update(self:~TensorType, indices:Any, values:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## arange
```python
Tensor.arange(self:~TensorType, start:int, stop:Union[int, NoneType]=None, step:Union[int, NoneType]=None) -> ~TensorType
```

## cumsum
```python
Tensor.cumsum(self:~TensorType, axis:Union[int, NoneType]=None) -> ~TensorType
```

## flip
```python
Tensor.flip(self:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None) -> ~TensorType
```

## meshgrid
```python
Tensor.meshgrid(self:~TensorType, *tensors:~TensorType, indexing:str='xy') -> Tuple[~TensorType, ...]
```

## pad
```python
Tensor.pad(self:~TensorType, paddings:Tuple[Tuple[int, int], ...], mode:str='constant', value:float=0) -> ~TensorType
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
Tensor.pow(self:~TensorType, exponent:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## value_and_grad
```python
Tensor.value_and_grad(self:~TensorType, f:Callable[..., ~TensorType], *args:Any, **kwargs:Any) -> Tuple[~TensorType, ~TensorType]
```

## value_aux_and_grad
```python
Tensor.value_aux_and_grad(self:~TensorType, f:Callable[..., Tuple[~TensorType, Any]], *args:Any, **kwargs:Any) -> Tuple[~TensorType, Any, ~TensorType]
```

## flatten
```python
Tensor.flatten(self:~TensorType, start:int=0, end:int=-1) -> ~TensorType
```

## l0
```python
NormsMethods.l0(x:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## l1
```python
NormsMethods.l1(x:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## l2
```python
NormsMethods.l2(x:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## linf
```python
NormsMethods.linf(x:~TensorType, axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## lp
```python
NormsMethods.lp(x:~TensorType, p:Union[int, float], axis:Union[int, Tuple[int, ...], NoneType]=None, keepdims:bool=False) -> ~TensorType
```

## raw

## dtype

## __init__
```python
Tensor.__init__(self, raw:Any)
```

## __repr__
```python
Tensor.__repr__(self:~TensorType) -> str
```

## __format__
```python
Tensor.__format__(self:~TensorType, format_spec:str) -> str
```

## __getitem__
```python
Tensor.__getitem__(self:~TensorType, index:Any) -> ~TensorType
```

## __iter__
```python
Tensor.__iter__(self:~TensorType) -> Iterator[~TensorType]
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
Tensor.__add__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __radd__
```python
Tensor.__radd__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __sub__
```python
Tensor.__sub__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __rsub__
```python
Tensor.__rsub__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __mul__
```python
Tensor.__mul__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __rmul__
```python
Tensor.__rmul__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __truediv__
```python
Tensor.__truediv__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __rtruediv__
```python
Tensor.__rtruediv__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __floordiv__
```python
Tensor.__floordiv__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __rfloordiv__
```python
Tensor.__rfloordiv__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __mod__
```python
Tensor.__mod__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __lt__
```python
Tensor.__lt__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __le__
```python
Tensor.__le__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __eq__
```python
Tensor.__eq__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __ne__
```python
Tensor.__ne__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __gt__
```python
Tensor.__gt__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __ge__
```python
Tensor.__ge__(self:~TensorType, other:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

## __pow__
```python
Tensor.__pow__(self:~TensorType, exponent:Union[_ForwardRef('Tensor'), int, float]) -> ~TensorType
```

