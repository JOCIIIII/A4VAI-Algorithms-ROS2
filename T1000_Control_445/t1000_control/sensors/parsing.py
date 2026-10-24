from __future__ import annotations

import numpy as np


_XYZ_DTYPE_CACHE: dict[int, np.dtype] = {}


def _xyz_dtype(point_step: int) -> np.dtype:
    dt = _XYZ_DTYPE_CACHE.get(point_step)
    if dt is None:
        dt = np.dtype({"names": ["x", "y", "z"],
                       "formats": ["<f4", "<f4", "<f4"],
                       "offsets": [0, 4, 8],
                       "itemsize": int(point_step)})
        _XYZ_DTYPE_CACHE[point_step] = dt
    return dt


def _records_to_xyz(rec) -> np.ndarray:
    xyz = np.empty((len(rec), 3), dtype=np.float32)
    xyz[:, 0] = rec["x"]
    xyz[:, 1] = rec["y"]
    xyz[:, 2] = rec["z"]

    keep = ~np.isnan(xyz).any(axis=1)
    if keep.all():
        return xyz
    return xyz[keep]


def _pointcloud2_count(msg, point_step: int) -> int | None:
    if getattr(msg, "is_bigendian", False):
        raise ValueError("PointCloud2 big-endian data is not supported")
    if point_step < 12:
        raise ValueError("PointCloud2 point_step must be at least 12 bytes")

    dimensions = [getattr(msg, name, None) for name in ("width", "height", "row_step")]
    if all(value is None for value in dimensions):
        return None
    if any(value is None for value in dimensions):
        raise ValueError("PointCloud2 width, height and row_step must be present together")

    width, height, row_step = map(int, dimensions)
    if width < 0 or height <= 0 or row_step != width * point_step:
        raise ValueError("PointCloud2 row_step does not match width * point_step")
    if len(msg.data) != row_step * height:
        raise ValueError("PointCloud2 data size does not match row_step * height")
    return width * height


def pointcloud2_to_xyz(msg) -> np.ndarray:
    step = int(msg.point_step)
    if step <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    declared_count = _pointcloud2_count(msg, step)
    raw = np.frombuffer(memoryview(msg.data).cast("B"), dtype=np.uint8)
    n = raw.size // step if declared_count is None else declared_count
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rec = raw[: n * step].view(_xyz_dtype(step))
    return _records_to_xyz(rec)


def check_pointcloud2_layout(msg) -> bool:
    try:
        _pointcloud2_count(msg, int(msg.point_step))
    except (AttributeError, TypeError, ValueError):
        return False
    want = {"x": 0, "y": 4, "z": 8}
    got = {f.name: f.offset for f in msg.fields if f.name in want}
    if got != want:
        return False
    return all(f.datatype == 7 and getattr(f, "count", 1) == 1
               for f in msg.fields if f.name in want)


CUSTOM_POINT_STRIDE = 20


CUSTOM_POINT_XYZ_OFFSET = 4


_CUSTOM_DTYPE_CACHE: dict[int, np.dtype] = {}


def _custom_xyz_dtype(stride: int) -> np.dtype:
    dt = _CUSTOM_DTYPE_CACHE.get(stride)
    if dt is None:
        o = CUSTOM_POINT_XYZ_OFFSET
        dt = np.dtype({"names": ["x", "y", "z"],
                       "formats": ["<f4", "<f4", "<f4"],
                       "offsets": [o, o + 4, o + 8],
                       "itemsize": int(stride)})
        _CUSTOM_DTYPE_CACHE[stride] = dt
    return dt


def custommsg_to_xyz(msg) -> np.ndarray:
    pts = msg.points
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.empty((n, 3), dtype=np.float32)
    for i, p in enumerate(pts):
        xyz[i, 0] = p.x
        xyz[i, 1] = p.y
        xyz[i, 2] = p.z
    keep = ~np.isnan(xyz).any(axis=1)
    return xyz if keep.all() else xyz[keep]


def custommsg_raw_to_xyz(data, point_num: int | None = None) -> np.ndarray:
    buf = np.frombuffer(memoryview(data).cast("B"), dtype=np.uint8)
    if buf.size < 8:
        raise ValueError(f"CustomMsg CDR is too short: {buf.size} bytes")

    raw = memoryview(data).cast("B")
    base = 4
    off = base + 8

    def _u32(o: int) -> int:
        return int(np.frombuffer(raw[o:o + 4], dtype="<u4")[0])

    def _align(o: int, a: int) -> int:
        rem = (o - base) % a
        return o if rem == 0 else o + (a - rem)

    slen = _u32(off)
    off += 4 + slen

    off = _align(off, 8)
    off += 8
    point_num_hdr = _u32(off); off += 4
    off += 4
    seq_len = _u32(off); off += 4

    n = int(point_num if point_num is not None else seq_len)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if seq_len != point_num_hdr:
        raise ValueError(f"point_num({point_num_hdr}) != sequence length({seq_len})")

    avail = buf.size - off
    stride, rem = divmod(avail, n)
    if not (16 <= stride <= 32) or rem != 0:
        raise ValueError(
            f"bad CustomPoint stride: {avail}B / {n} points = {stride}B, remainder {rem}. "
            f"Expected {CUSTOM_POINT_STRIDE}B. Check one real Orin message."
        )

    rec = buf[off: off + n * stride].view(_custom_xyz_dtype(stride))
    return _records_to_xyz(rec)
