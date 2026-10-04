import json
import shutil
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal

import dask
import numpy as np
import pandas as pd

import xarray as xr

# Keys the reader interprets as type tags; dicts containing them stay tagged.
_RESERVED_TAGS = {"$type", "__tuple__", "__dict__", "__ndarray__", "__xnpy__"}


def open_xnpy_ndarray(
    path: str | Path, *, mmap_mode: Literal["r", "r+", "c"] | None = "r"
) -> np.ndarray:
    """Open an ndarray store.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`.

    Returns
    -------
    numpy.ndarray
        The loaded NumPy array.

    Raises
    ------
    TypeError
        If the store does not hold an ndarray.
    """
    obj = open_xnpy(path, mmap_mode=mmap_mode)
    if not isinstance(obj, np.ndarray):
        raise TypeError(f"Store {path} does not hold an ndarray.")
    return obj


def open_xnpy_dataframe(
    path: str | Path,
    variable: Hashable | None = None,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> pd.DataFrame | pd.Series:
    """Open a DataFrame or Series store, optionally restricted to one column.

    For a DataFrame store, only the payloads of the column labelled ``variable``
    are opened and the result is a one-column DataFrame carrying the stored
    index. ``variable`` is ignored for a Series store.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    variable : hashable, optional
        Column label to load for a DataFrame store.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The loaded pandas object.

    Raises
    ------
    TypeError
        If the store holds neither a DataFrame nor a Series.
    """
    obj = open_xnpy(path, variable, mmap_mode=mmap_mode)
    if not isinstance(obj, pd.DataFrame | pd.Series):
        raise TypeError(f"Store {path} does not hold a DataFrame or Series.")
    return obj


def open_xnpy_dataset(
    path: str | Path,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> xr.Dataset:
    """Open a Dataset store.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`.

    Returns
    -------
    xarray.Dataset
        The loaded xarray Dataset.

    Raises
    ------
    TypeError
        If the store does not hold a Dataset.
    """
    obj = open_xnpy(path, mmap_mode=mmap_mode)
    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Store {path} does not hold a Dataset.")
    return obj


def open_xnpy(
    path: str | Path,
    variable: Hashable | None = None,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | xr.Dataset:
    """Open a NumPy, pandas, or xarray object from a memory-mappable XNpy store.

    The return type is determined at runtime from the store manifest. Use
    :func:`open_xnpy_dataframe` or :func:`open_xnpy_dataset` for static typing.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    variable : hashable, optional
        Dataset variable or DataFrame column label to return. Ignored for
        ndarray, Series, and DataArray stores. For a DataFrame, only the
        selected column payloads are opened and a one-column DataFrame is
        returned. If omitted, returns the complete stored object.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`:
            * ``"r"``: Read-only, data is memory-mapped.
            * ``"r+"``: Read/write, data is memory-mapped in-place.
            * ``"c"``: Copy-on-write, accepts in-place edits without modifying files.
            * ``None``: Loads payloads fully into ordinary in-memory arrays.
        Note that text columns are always loaded into memory.

    Returns
    -------
    numpy.ndarray, pandas.Series, pandas.DataFrame, xarray.DataArray, or xarray.Dataset
        Reconstructed object. A Dataset store with ``variable`` specified returns a
        DataArray. Array payloads are memory-mapped unless ``mmap_mode=None``.

    Raises
    ------
    KeyError
        If the specified ``variable`` cannot be found in the store.
    ValueError
        If an unknown store kind or metadata type is encountered.
    """
    root = Path(path)

    metadata = decode_metadata(
        json.loads((root / "metadata.json").read_text()), root, mmap_mode
    )
    kind = metadata["kind"]

    def load(file: str) -> np.ndarray:
        return np.load(root / file, mmap_mode=mmap_mode, allow_pickle=False)

    def load_variable(info: dict[str, Any]) -> xr.Variable:
        return xr.Variable(info["dims"], load(info["file"]), attrs=info["attrs"])

    def load_column(spec: dict[str, Any]) -> Any:
        """Decode a pandas column from array, masked-array, or categorical storage."""
        values = load(spec["file"])
        encoding = spec["encoding"]

        if encoding == "array":
            return values

        if encoding == "categorical":
            return pd.Categorical.from_codes(
                values,
                categories=spec["categories"],
                ordered=spec["ordered"],
            )

        if encoding == "masked":
            dtype = pd.api.types.pandas_dtype(spec["dtype"])
            mask = np.asarray(load(spec["mask"]), dtype=bool)

            if dtype == object:
                out = np.asarray(values).astype(object)
                out[mask] = np.nan
                return pd.Index(out, dtype=object, copy=False)

            if isinstance(dtype, pd.DatetimeTZDtype):
                out = pd.to_datetime(
                    np.asarray(values), unit="ns", utc=True
                ).tz_convert(dtype.tz)
                out = pd.array(out, dtype=dtype)
                out[mask] = pd.NaT
                return out

            out = np.asarray(values).astype(object)
            out[mask] = dtype.na_value
            return pd.array(out, dtype=dtype)

        raise ValueError(f"Unknown pandas encoding: {encoding!r}.")

    def load_index() -> pd.Index:
        levels = [load_column(spec) for spec in metadata["index"]]
        names = metadata["index_names"]
        if len(levels) == 1:
            return pd.Index(levels[0], name=names[0])
        return pd.MultiIndex.from_arrays(levels, names=names)

    if kind == "ndarray":
        return load(metadata["file"])
    elif kind == "series":
        series = pd.Series(
            load_column(metadata["column"]),
            index=load_index(),
            name=metadata["name"],
            copy=False,
        )
        series.attrs = metadata["attrs"]
        return series
    elif kind == "dataframe":
        columns = [(c["label"], c["column"]) for c in metadata["columns"]]
        if variable is not None:
            columns = [column for column in columns if column[0] == variable]
            if not columns:
                raise KeyError(variable)

        names = metadata["columns_names"]
        labels = [label for label, _ in columns]
        frame = pd.DataFrame(
            {position: load_column(spec) for position, (_, spec) in enumerate(columns)},
            index=load_index(),
            copy=False,
        )
        frame.columns = (
            pd.MultiIndex.from_tuples(labels, names=names)
            if len(names) > 1
            else pd.Index(labels, name=names[0], tupleize_cols=False)
        )
        frame.attrs = metadata["attrs"]
        return frame
    elif kind in {"dataarray", "dataset"}:
        coords = {
            name: load_variable(info) for name, info in metadata["coords"].items()
        }
        if kind == "dataarray":
            return xr.DataArray(
                load_variable(metadata["variable"]),
                coords=coords,
                name=metadata["name"],
            )

        dataset = xr.Dataset(
            {name: load_variable(info) for name, info in metadata["variables"].items()},
            coords=coords,
            attrs=metadata["attrs"],
        )
        return dataset if variable is None else dataset[variable]
    else:
        raise ValueError(f"Unknown XNpy kind: {kind!r}.")


def to_xnpy(
    obj: np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    mode: Literal["w", "w-"] = "w-",
    scheduler: Literal["threads", "synchronous"] = "threads",
    num_workers: int | None = None,
) -> None:
    """Write a NumPy, pandas, or xarray object to a memory-mappable XNpy store.

    In-memory arrays are written with :func:`numpy.save`. Large lazily indexed
    arrays are streamed in bounded, chunk-aligned slabs. All array writes (variables,
    coordinates, columns, masks) are built as :func:`dask.delayed` tasks and
    Dask-backed variables as :func:`dask.array.store` tasks. They execute in a
    single :func:`dask.compute` call, so columns, variables, and chunks run
    concurrently under the selected scheduler.

    Metadata (``attrs``, labels, names) is encoded with type tags so that tuples,
    dicts with non-string keys, and nested pandas or xarray objects (stored as
    sub-stores under ``attrs/``) round-trip. Dicts with string keys are stored
    as native, nested JSON objects.
    Nullable and object-backed columns are stored as typed NumPy arrays
    with separate missing-value masks; categorical columns store integer
    codes and their categories. Every column and index level stores its
    pandas dtype, which the reader restores.

    Parameters
    ----------
    obj : numpy.ndarray, pandas.Series, pandas.DataFrame, xarray.DataArray, or xarray.Dataset
        Object to store. Labels and attributes must be built from standard Python
        types (str, int, float, bool, None, tuples, lists, dicts), NumPy scalars
        and arrays, or nested pandas and xarray objects.
    path : str or pathlib.Path
        Destination directory for the XNpy store.
    mode : {"w", "w-"}, default="w-"
        Write mode:
            * ``"w"``: Replaces an existing store if present.
            * ``"w-"``: Requires that the destination directory does not exist.
    scheduler : {"threads", "synchronous"}, default="threads"
        Dask scheduler used to execute all write tasks:
            * ``"threads"``: Writes concurrently using threads.
            * ``"synchronous"``: Writes serially in the calling thread.
    num_workers : int, optional
        Number of workers passed to :func:`dask.compute`. If omitted, the Dask
        default is used.

    Returns
    -------
    None
        The store is written to ``path``.

    Raises
    ------
    ValueError
        If an unsupported ``mode`` or ``scheduler`` is specified.
    TypeError
        If ``obj`` is not one of the supported types.
    FileExistsError
        If ``path`` already exists and ``mode="w-"``.
    """
    if mode not in {"w", "w-"}:
        raise ValueError(f"Unsupported XNpy mode: {mode!r}.")

    if not isinstance(
        obj, (np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(f"Unsupported XNpy object type: {type(obj)!r}.")
    if scheduler not in {"threads", "synchronous"}:
        raise ValueError(f"Unsupported XNpy scheduler: {scheduler!r}.")

    root = Path(path)
    if root.exists():
        if mode == "w-":
            raise FileExistsError(root)
        shutil.rmtree(root)

    jobs: list[tuple[Any, str]] = []
    nested: list[Any] = []

    def add(source: Any, folder: str, name: Any) -> str:
        file = f"{folder}/{name}.npy"
        jobs.append((source, file))
        return file

    def add_variable(source: xr.DataArray, folder: str, name: Any) -> dict[str, Any]:
        return {
            "dims": list(source.dims),
            "attrs": dict(source.attrs),
            "file": add(source, folder, name),
        }

    def add_column(values: Any, folder: str, name: Any) -> dict[str, Any]:
        """Encode a pandas column into array, masked-array, or categorical storage."""
        series = pd.Series(values)
        dtype = series.dtype
        dtype_name = str(dtype)

        def masked(data: np.ndarray) -> dict[str, Any]:
            return {
                "encoding": "masked",
                "dtype": dtype_name,
                "file": add(data, folder, name),
                "mask": add(series.isna().to_numpy(), folder, f"{name}.mask"),
            }

        if isinstance(dtype, pd.CategoricalDtype):
            return {
                "encoding": "categorical",
                "dtype": dtype_name,
                "file": add(series.cat.codes.to_numpy(), folder, name),
                "categories": series.cat.categories.tolist(),
                "ordered": bool(series.cat.ordered),
            }

        if isinstance(dtype, pd.BooleanDtype):
            return masked(series.fillna(False).to_numpy(dtype=bool))

        if isinstance(dtype, pd.StringDtype):
            return masked(series.fillna("").to_numpy(dtype=str))

        if isinstance(series.array, pd.arrays.IntegerArray | pd.arrays.FloatingArray):
            numpy_dtype = dtype.numpy_dtype
            data = series.to_numpy(
                dtype=numpy_dtype,
                na_value=numpy_dtype.type(0),
            )
            return masked(data)

        if isinstance(dtype, pd.DatetimeTZDtype):
            return masked(series.array.asi8)

        if dtype == object:
            mask = series.isna().to_numpy()
            valid = series[~mask]

            if valid.empty:
                return masked(np.zeros(len(series), dtype=np.uint8))

            inferred = pd.api.types.infer_dtype(valid, skipna=True)

            if inferred == "boolean":
                return masked(series.fillna(False).to_numpy(dtype=bool))

            if inferred in {"integer", "floating", "mixed-integer-float"}:
                valid_values = np.asarray(valid.tolist())

                if valid_values.dtype == object:
                    raise TypeError(
                        "Object column contains numeric values that cannot be represented by a NumPy dtype."
                    )

                data = np.zeros(len(series), dtype=valid_values.dtype)
                data[~mask] = valid_values
                return masked(data)

            if inferred in {"string", "unicode"}:
                return masked(series.fillna("").to_numpy(dtype=str))

            if inferred == "bytes":
                valid_values = np.asarray(valid.tolist())

                if valid_values.dtype == object:
                    raise TypeError(
                        "Object column contains bytes that cannot be represented by a NumPy dtype."
                    )

                data = np.zeros(len(series), dtype=valid_values.dtype)
                data[~mask] = valid_values
                return masked(data)

            raise TypeError(
                f"Unsupported object column with inferred dtype {inferred!r}."
            )

        data = series.to_numpy(copy=False)

        if data.dtype == object:
            raise TypeError(f"Unsupported pandas dtype: {dtype!r}.")

        return {
            "encoding": "array",
            "dtype": dtype_name,
            "file": add(data, folder, name),
        }

    def add_index(index: pd.Index) -> list[dict[str, Any]]:
        return [
            add_column(index.get_level_values(i), "coords", f"index.{i}")
            for i in range(index.nlevels)
        ]

    if isinstance(obj, xr.Dataset):
        metadata: dict[str, Any] = {
            "kind": "dataset",
            "attrs": dict(obj.attrs),
            "variables": {
                name: add_variable(var, "data", name)
                for name, var in obj.data_vars.items()
            },
            "coords": {
                name: add_variable(coord, "coords", name)
                for name, coord in obj.coords.items()
            },
        }
    elif isinstance(obj, xr.DataArray):
        metadata = {
            "kind": "dataarray",
            "name": obj.name,
            "variable": add_variable(
                obj,
                "data",
                "array" if obj.name is None else obj.name,
            ),
            "coords": {
                name: add_variable(coord, "coords", name)
                for name, coord in obj.coords.items()
            },
        }
    elif isinstance(obj, pd.DataFrame):
        metadata = {
            "kind": "dataframe",
            "attrs": dict(obj.attrs),
            "index_names": list(obj.index.names),
            "index": add_index(obj.index),
            "columns_names": list(obj.columns.names),
            "columns": [
                {"label": label, "column": add_column(column, "data", position)}
                for position, (label, column) in enumerate(obj.items())
            ],
        }
    elif isinstance(obj, pd.Series):
        metadata = {
            "kind": "series",
            "name": obj.name,
            "attrs": dict(obj.attrs),
            "index_names": list(obj.index.names),
            "index": add_index(obj.index),
            "column": add_column(obj, "data", "series"),
        }
    else:
        metadata = {"kind": "ndarray", "file": add(obj, "data", "array")}

    slab_bytes = 64 * 1024**2

    def write_array(source: Any, path: Path) -> None:
        np.save(path, np.asarray(source), allow_pickle=False)

    def write_slab(source: xr.Variable, path: Path, offset: int) -> None:
        slab = np.ascontiguousarray(source)
        with path.open("r+b") as file:
            file.seek(offset)
            file.write(slab)

    def flush(target: np.ndarray, *_: Any) -> None:
        target.flush()

    def make_task(source: Any, path: Path) -> list[Any]:
        if source.dtype == object:
            source = np.asarray(source).astype(str)

        if getattr(source, "chunks", None) is not None:
            import dask.array as da

            target = np.lib.format.open_memmap(
                path, "w+", dtype=source.dtype, shape=source.shape
            )
            data = source.data if isinstance(source, xr.DataArray) else source
            stored = da.store(data, target, lock=False, compute=False)
            return [dask.delayed(flush, pure=False)(target, stored)]

        if (
            isinstance(source, np.ndarray)
            or source.nbytes <= slab_bytes
            or not source.ndim
        ):
            return [dask.delayed(write_array, pure=False)(source, path)]

        offset = np.lib.format.open_memmap(
            path, "w+", dtype=source.dtype, shape=source.shape
        ).offset
        strides = [
            source.dtype.itemsize * int(np.prod(source.shape[axis + 1 :]))
            for axis in range(source.ndim)
        ]
        axis = next(axis for axis, size in enumerate(strides) if size <= slab_bytes)
        chunk = source.encoding.get("preferred_chunks", {}).get(source.dims[axis], 1)
        step = max(slab_bytes // strides[axis] // chunk, 1) * chunk
        return [
            dask.delayed(write_slab, pure=False)(
                source.variable[(*lead, slice(start, start + step))],
                path,
                offset
                + sum(i * size for i, size in zip(lead, strides, strict=False))
                + start * strides[axis],
            )
            for lead in np.ndindex(*source.shape[:axis])
            for start in range(0, source.shape[axis], step)
        ]

    try:
        metadata = encode_metadata(metadata, nested)
        root.mkdir(parents=True)
        for folder in {(root / file).parent for _, file in jobs}:
            folder.mkdir(exist_ok=True)

        tasks = [
            task for source, file in jobs for task in make_task(source, root / file)
        ]
        dask.compute(*tasks, scheduler=scheduler, num_workers=num_workers)

        for position, value in enumerate(nested):
            to_xnpy(
                value,
                root / "attrs" / str(position),
                scheduler=scheduler,
                num_workers=num_workers,
            )

        with (root / "metadata.json").open("w") as file:
            json.dump(metadata, file, indent=2)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def encode_metadata(value: Any, nested: list[Any]) -> Any:
    """Convert metadata to JSON using a single tagged-value schema."""
    if isinstance(value, np.dtype):
        if value.fields is None:
            return {"$type": "numpy.dtype", "value": value.str}
        return {
            "$type": "numpy.dtype",
            "descr": encode_metadata(value.descr, nested),
        }
    if isinstance(value, np.datetime64):
        return {
            "$type": "numpy.datetime64",
            "dtype": value.dtype.str,
            "value": value.view("i8").item(),
        }
    if isinstance(value, np.timedelta64):
        return {
            "$type": "numpy.timedelta64",
            "dtype": value.dtype.str,
            "value": value.view("i8").item(),
        }
    if isinstance(value, np.generic):
        return {
            "$type": "numpy.scalar",
            "dtype": encode_metadata(value.dtype, nested),
            "value": encode_metadata(value.item(), nested),
        }
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, bytes):
        return {"$type": "bytes", "value": value.hex()}
    if isinstance(value, complex):
        return {"$type": "complex", "value": [value.real, value.imag]}
    if isinstance(value, np.ndarray):
        data = (
            value.view("i8").tolist()
            if value.dtype.kind in {"M", "m"}
            else encode_metadata(value.tolist(), nested)
        )
        return {
            "$type": "numpy.ndarray",
            "dtype": encode_metadata(value.dtype, nested),
            "shape": list(value.shape),
            "value": data,
        }
    if isinstance(value, pd.DataFrame | pd.Series | xr.DataArray | xr.Dataset):
        nested.append(value)
        return {"$type": "xnpy", "path": f"attrs/{len(nested) - 1}"}
    if isinstance(value, tuple):
        return {
            "$type": "tuple",
            "value": [encode_metadata(item, nested) for item in value],
        }
    if isinstance(value, list):
        return [encode_metadata(item, nested) for item in value]
    if isinstance(value, dict):
        if all(type(key) is str for key in value) and _RESERVED_TAGS.isdisjoint(value):
            return {key: encode_metadata(item, nested) for key, item in value.items()}
        return {
            "$type": "dict",
            "value": [
                [
                    encode_metadata(key, nested),
                    encode_metadata(item, nested),
                ]
                for key, item in value.items()
            ],
        }
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}.")


def decode_metadata(
    value: Any, root: Path, mmap_mode: Literal["r", "r+", "c"] | None
) -> Any:
    if isinstance(value, list):
        return [decode_metadata(item, root, mmap_mode) for item in value]
    if not isinstance(value, dict):
        return value

    type_name = value.get("$type")
    if type_name is None:
        if "__tuple__" in value:
            return tuple(
                decode_metadata(item, root, mmap_mode) for item in value["__tuple__"]
            )
        if "__dict__" in value:
            return {
                decode_metadata(key, root, mmap_mode): decode_metadata(
                    item, root, mmap_mode
                )
                for key, item in value["__dict__"]
            }
        if "__ndarray__" in value:
            return np.array(value["__ndarray__"], dtype=value["dtype"])
        if "__xnpy__" in value:
            return open_xnpy(root / value["__xnpy__"], mmap_mode=mmap_mode)
        return {
            key: decode_metadata(item, root, mmap_mode) for key, item in value.items()
        }
    if type_name == "tuple":
        return tuple(decode_metadata(item, root, mmap_mode) for item in value["value"])
    if type_name == "dict":
        return {
            decode_metadata(key, root, mmap_mode): decode_metadata(
                item, root, mmap_mode
            )
            for key, item in value["value"]
        }
    if type_name == "bytes":
        return bytes.fromhex(value["value"])
    if type_name == "complex":
        return complex(*value["value"])
    if type_name == "numpy.dtype":
        return np.dtype(
            decode_metadata(value["descr"], root, mmap_mode)
            if "descr" in value
            else value["value"]
        )
    if type_name in {"numpy.datetime64", "numpy.timedelta64"}:
        dtype = np.dtype(value["dtype"])
        return np.asarray(value["value"], dtype="i8").view(dtype)[()]
    if type_name == "numpy.scalar":
        dtype = decode_metadata(value["dtype"], root, mmap_mode)
        return np.asarray(
            decode_metadata(value["value"], root, mmap_mode), dtype=dtype
        )[()]
    if type_name == "numpy.ndarray":
        dtype = decode_metadata(value["dtype"], root, mmap_mode)
        data = decode_metadata(value["value"], root, mmap_mode)
        array = (
            np.asarray(data, dtype="i8").view(dtype)
            if dtype.kind in {"M", "m"}
            else np.asarray(data, dtype=dtype)
        )
        return array.reshape(value["shape"])
    if type_name == "xnpy":
        return open_xnpy(root / value["path"], mmap_mode=mmap_mode)
    raise ValueError(f"Unknown metadata type: {type_name!r}.")
