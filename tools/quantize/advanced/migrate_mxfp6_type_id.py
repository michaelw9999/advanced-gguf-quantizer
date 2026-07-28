#!/usr/bin/env python3
"""Inspect or migrate legacy MXFP6 GGUF type IDs without rewriting tensor data."""

from __future__ import annotations

import argparse
import dataclasses
import json
import mmap
import struct
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

SCALAR_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


class NeedMoreData(Exception):
    pass


class InvalidGGUF(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class MetadataValue:
    value_type: int
    offset: int
    value: object | None


@dataclasses.dataclass(frozen=True)
class GGUFHeader:
    version: int
    tensor_count: int
    metadata_count: int
    metadata: dict[str, MetadataValue]
    tensor_type_offsets: tuple[tuple[str, int, int], ...]
    header_end: int

    @property
    def tensor_types(self) -> Counter[int]:
        return Counter(tensor_type for _, _, tensor_type in self.tensor_type_offsets)


class Cursor:
    def __init__(self, data: bytes | mmap.mmap):
        self.data = data
        self.pos = 0

    def take(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.data):
            raise NeedMoreData
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def unpack(self, fmt: str) -> tuple[object, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))

    def string(self) -> str:
        (size,) = self.unpack("<Q")
        if size > 1 << 31:
            raise InvalidGGUF(f"unreasonable GGUF string length: {size}")
        return self.take(size).decode("utf-8")


def skip_value(cursor: Cursor, value_type: int, *, capture_scalar: bool = False) -> tuple[int, object | None]:
    offset = cursor.pos
    if value_type in SCALAR_SIZES:
        raw = cursor.take(SCALAR_SIZES[value_type])
        if not capture_scalar:
            return offset, None
        formats = {
            GGUF_TYPE_UINT8: "<B",
            GGUF_TYPE_INT8: "<b",
            GGUF_TYPE_UINT16: "<H",
            GGUF_TYPE_INT16: "<h",
            GGUF_TYPE_UINT32: "<I",
            GGUF_TYPE_INT32: "<i",
            GGUF_TYPE_FLOAT32: "<f",
            GGUF_TYPE_BOOL: "<?",
            GGUF_TYPE_UINT64: "<Q",
            GGUF_TYPE_INT64: "<q",
            GGUF_TYPE_FLOAT64: "<d",
        }
        return offset, struct.unpack(formats[value_type], raw)[0]
    if value_type == GGUF_TYPE_STRING:
        value = cursor.string()
        return offset, value if capture_scalar else None
    if value_type == GGUF_TYPE_ARRAY:
        (element_type,) = cursor.unpack("<I")
        (count,) = cursor.unpack("<Q")
        if count > 1 << 40:
            raise InvalidGGUF(f"unreasonable GGUF array length: {count}")
        if element_type in SCALAR_SIZES:
            cursor.take(SCALAR_SIZES[element_type] * count)
        else:
            for _ in range(count):
                skip_value(cursor, element_type)
        return offset, None
    raise InvalidGGUF(f"unknown GGUF metadata type: {value_type}")


def parse_header(data: bytes | mmap.mmap) -> GGUFHeader:
    cursor = Cursor(data)
    if cursor.take(4) != GGUF_MAGIC:
        raise InvalidGGUF("not a GGUF file")
    (version,) = cursor.unpack("<I")
    if version not in (2, 3):
        raise InvalidGGUF(f"unsupported GGUF version: {version}")
    tensor_count, metadata_count = cursor.unpack("<QQ")

    metadata: dict[str, MetadataValue] = {}
    for _ in range(metadata_count):
        key = cursor.string()
        (value_type,) = cursor.unpack("<I")
        offset, value = skip_value(cursor, value_type, capture_scalar=True)
        metadata[key] = MetadataValue(value_type, offset, value)

    tensor_type_offsets: list[tuple[str, int, int]] = []
    for _ in range(tensor_count):
        name = cursor.string()
        (n_dims,) = cursor.unpack("<I")
        if n_dims > 4:
            raise InvalidGGUF(f"unreasonable tensor rank {n_dims} for {name!r}")
        cursor.take(8 * n_dims)
        type_offset = cursor.pos
        (tensor_type,) = cursor.unpack("<I")
        cursor.take(8)
        tensor_type_offsets.append((name, type_offset, tensor_type))

    return GGUFHeader(
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        metadata=metadata,
        tensor_type_offsets=tuple(tensor_type_offsets),
        header_end=cursor.pos,
    )


def read_remote_header(url: str) -> tuple[bytes, GGUFHeader]:
    size = 4 * 1024 * 1024
    while size <= 256 * 1024 * 1024:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
        with urllib.request.urlopen(request) as response:
            data = response.read(size)
        try:
            return data, parse_header(data)
        except NeedMoreData:
            size *= 2
    raise InvalidGGUF("GGUF header exceeds the 256 MiB remote inspection limit")


def inspect_path(path: Path, *, writable: bool = False) -> tuple[BinaryIO, mmap.mmap, GGUFHeader]:
    handle = path.open("r+b" if writable else "rb")
    try:
        access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
        mapped = mmap.mmap(handle.fileno(), 0, access=access)
        try:
            return handle, mapped, parse_header(mapped)
        except Exception:
            mapped.close()
            raise
    except Exception:
        handle.close()
        raise


def summary(source: str, header: GGUFHeader) -> dict[str, object]:
    file_type = header.metadata.get("general.file_type")
    return {
        "source": source,
        "gguf_version": header.version,
        "tensor_count": header.tensor_count,
        "metadata_count": header.metadata_count,
        "general.file_type": file_type.value if file_type else None,
        "tensor_types": dict(sorted(header.tensor_types.items())),
        "header_end": header.header_end,
    }


def migrate(path: Path, *, source_type: int, target_type: int, source_file_type: int) -> dict[str, object]:
    handle, mapped, header = inspect_path(path, writable=True)
    try:
        matching = [(name, offset) for name, offset, tensor_type in header.tensor_type_offsets if tensor_type == source_type]
        if not matching:
            raise InvalidGGUF(f"no tensor descriptors use legacy type {source_type}")
        if header.tensor_types[target_type]:
            raise InvalidGGUF(
                f"file already contains tensor type {target_type}; refusing a possibly partial migration"
            )

        file_type = header.metadata.get("general.file_type")
        if file_type and file_type.value == source_file_type:
            if file_type.value_type not in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32):
                raise InvalidGGUF("general.file_type is not a 32-bit integer")
            mapped[file_type.offset:file_type.offset + 4] = struct.pack("<I", target_type)
            file_type_changed = True
        else:
            file_type_changed = False

        packed_target = struct.pack("<I", target_type)
        for _, offset in matching:
            mapped[offset:offset + 4] = packed_target
        mapped.flush()

        migrated = parse_header(mapped)
        if migrated.tensor_types[source_type] != 0:
            raise InvalidGGUF("legacy tensor types remain after migration")
        if migrated.tensor_types[target_type] != len(matching):
            raise InvalidGGUF("migrated tensor count does not match the preflight count")
        return {
            **summary(str(path), migrated),
            "migrated_tensor_count": len(matching),
            "general.file_type_changed": file_type_changed,
        }
    finally:
        mapped.close()
        handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local GGUF path or an HTTP(S) URL for inspection")
    parser.add_argument("--migrate", action="store_true", help="rewrite a local legacy MXFP6 GGUF in place")
    parser.add_argument(
        "--confirm-legacy-mxfp6",
        action="store_true",
        help="confirm that tensor type 42 in this artifact is legacy MXFP6, not official Q2_0",
    )
    parser.add_argument("--source-type", type=int, default=42)
    parser.add_argument("--target-type", type=int, default=50)
    parser.add_argument("--source-file-type", type=int, default=41)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.source.startswith(("http://", "https://")):
            if args.migrate:
                raise InvalidGGUF("remote sources are inspection-only; download before migration")
            _, header = read_remote_header(args.source)
            result = summary(args.source, header)
        else:
            path = Path(args.source)
            if args.migrate:
                if not args.confirm_legacy_mxfp6:
                    raise InvalidGGUF(
                        "--migrate requires --confirm-legacy-mxfp6 because official Q2_0 also uses type 42"
                    )
                result = migrate(
                    path,
                    source_type=args.source_type,
                    target_type=args.target_type,
                    source_file_type=args.source_file_type,
                )
            else:
                handle, mapped, header = inspect_path(path)
                try:
                    result = summary(str(path), header)
                finally:
                    mapped.close()
                    handle.close()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (InvalidGGUF, NeedMoreData, OSError, struct.error, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
