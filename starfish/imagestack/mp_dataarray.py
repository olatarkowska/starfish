import copy
from collections import OrderedDict
from multiprocessing import Array as mp_array  # type: ignore
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from xarray import Variable

from starfish.types import Number


class MPDataArray:
    _initialized = False

    def __init__(
            self,
            backing_mp_array: mp_array,
            shaped_np_array: Union[np.ndarray, Variable],
            *args,
            **kwargs) -> None:
        self.__backing_mp_array = backing_mp_array
        self.__array = xr.DataArray(shaped_np_array, *args, **kwargs)
        self._initialized = True

    @classmethod
    def from_shape_and_dtype(
            cls,
            shape: Tuple[int, ...],
            dtype,
            fill_value: Optional[Number]=None,
            *args,
            **kwargs) -> "MPDataArray":
        backing_mp_array, shaped_np_array = _create_np_array_backed_by_mp_array(shape, dtype)
        if fill_value is not None:
            shaped_np_array.fill(fill_value)

        return MPDataArray(backing_mp_array, shaped_np_array, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.__array, item)

    def __setattr__(self, key, value):
        if self._initialized:
            return setattr(self.__array, key, value)
        else:
            object.__setattr__(self, key, value)

    def __getitem__(self, item):
        return self.__array[item]

    def __setitem__(self, key, value):
        self.__array[key] = value

    def __dir__(self):
        return dir(self.__array)

    @property
    def backing_mp_array(self):
        return self.__backing_mp_array

    def __deepcopy__(self, memodict={}):
        backing_mp_array, shaped_np_array = _create_np_array_backed_by_mp_array(
            self.__array.variable.shape, self.__array.variable.dtype)

        shaped_np_array[:] = self.__array.variable.data

        variable = Variable(
            copy.deepcopy(self.__array.variable.dims, memodict),
            shaped_np_array,
            # attrs and encoding are deep copied in the constructor.
            self.__array.variable.attrs,
            self.__array.variable.encoding,
        )
        coords = OrderedDict((k, v.copy(deep=True))
                             for k, v in self._coords.items())

        result = MPDataArray(
            backing_mp_array,
            variable,
            coords=coords,
            name=self.__array.name,
            fastpath=True,
        )

        return result


def _create_np_array_backed_by_mp_array(
        shape: Tuple[int, ...], dtype) -> Tuple[mp_array, np.ndarray]:
    ctype_type = np.ctypeslib.as_ctypes(np.empty((1,), dtype=np.dtype(dtype))).__class__
    length = int(np.product(shape))  # the cast to int is required by multiprocessing.Array.
    backing_array = mp_array(ctype_type, length)
    unshaped_np_array = np.frombuffer(backing_array.get_obj(), dtype)
    shaped_np_array = unshaped_np_array.reshape(shape)

    return backing_array, shaped_np_array


def deepcopy(source: xr.DataArray) -> MPDataArray:
    backing_array, shaped_np_array = _create_np_array_backed_by_mp_array(
        source.variable.shape, source.variable.dtype)

    shaped_np_array[:] = source.variable

    # the rest is basically copied from xr.DataArray's __deepcopy__ method.
    coords = OrderedDict((k, v.copy(deep=True))
                         for k, v in source._coords.items())
    return source._replace(shaped_np_array, coords)
