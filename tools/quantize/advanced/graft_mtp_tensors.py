#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf import GGMLQuantizationType
from gguf.constants import GGML_QUANT_SIZES


LOCAL_QUANT_TYPES = {
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
    43: ("MXFP8", 256, 264),
    50: ("MXFP6_E2M3", 64, 52),
}


def register_local_quant_types() -> None:
    def missing(cls, value):
        if value not in LOCAL_QUANT_TYPES:
            return None
        name, block_size, type_size = LOCAL_QUANT_TYPES[value]
        obj = int.__new__(cls, value)
        obj._name_ = name
        obj._value_ = value
        cls._value2member_map_[value] = obj
        cls._member_map_[name] = obj
        if name not in cls._member_names_:
            cls._member_names_.append(name)
        GGML_QUANT_SIZES[obj] = (block_size, type_size)
        return obj

    GGMLQuantizationType._missing_ = classmethod(missing)
    for value in LOCAL_QUANT_TYPES:
        quant_type = GGMLQuantizationType(value)
        if quant_type not in GGML_QUANT_SIZES:
            _, block_size, type_size = LOCAL_QUANT_TYPES[value]
            GGML_QUANT_SIZES[quant_type] = (block_size, type_size)


def add_metadata(
    writer: GGUFWriter,
    reader: GGUFReader,
    *,
    name: str | None,
    mtp_donor: str | None = None,
    mtp_policy: str | None = None,
) -> None:
    skip = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
        "qwen35.block_count",
        "qwen35.nextn_predict_layers",
        "advanced_gguf_quantizer.mtp_donor",
        "advanced_gguf_quantizer.mtp_policy",
    }
    if name:
        skip.add("general.name")

    for key, field in reader.fields.items():
        if key in skip:
            continue
        if not field.types:
            continue
        vtype = field.types[0]
        subtype = field.types[1] if vtype == GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), vtype, subtype)

    if name:
        writer.add_string("general.name", name)
    if mtp_donor:
        writer.add_string("advanced_gguf_quantizer.mtp_donor", mtp_donor)
    if mtp_policy:
        writer.add_string("advanced_gguf_quantizer.mtp_policy", mtp_policy)
    writer.add_uint32("qwen35.block_count", 65)
    writer.add_uint32("qwen35.nextn_predict_layers", 1)


def add_tensor(writer: GGUFWriter, tensor) -> None:
    writer.add_tensor(
        tensor.name,
        tensor.data,
        raw_dtype=tensor.tensor_type,
    )


def gguf_align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def reader_alignment(reader: GGUFReader) -> int:
    field = reader.fields.get("general.alignment")
    if field is None:
        return 32
    return int(field.contents())


def tensor_payload_nbytes(reader: GGUFReader, path: Path, index: int) -> int:
    tensor = reader.tensors[index]
    alignment = reader_alignment(reader)
    offset = int(tensor.field.parts[-1][0])
    data_start = int(tensor.data_offset) - offset
    if index + 1 < len(reader.tensors):
        next_offset = int(reader.tensors[index + 1].field.parts[-1][0])
        padded_span = next_offset - offset
    else:
        padded_span = path.stat().st_size - (data_start + offset)

    base_nbytes = int(tensor.n_bytes)
    if gguf_align(base_nbytes, alignment) == padded_span:
        return base_nbytes

    # MXFP6_E2M3 with tensor-scale stores a 4-byte scale trailer that the
    # Python gguf reader does not include in tensor.data/n_bytes.
    if int(tensor.tensor_type) == 42 and gguf_align(base_nbytes + 4, alignment) == padded_span:
        return base_nbytes + 4

    return base_nbytes


def add_tensor_exact(writer: GGUFWriter, reader: GGUFReader, path: Path, index: int) -> None:
    tensor = reader.tensors[index]
    payload_nbytes = tensor_payload_nbytes(reader, path, index)
    if payload_nbytes == int(tensor.n_bytes):
        add_tensor(writer, tensor)
        return

    raw = np.memmap(path, mode="r", dtype=np.uint8, offset=int(tensor.data_offset), shape=(payload_nbytes,))
    writer.add_tensor_info(
        tensor.name,
        tensor.data.shape,
        np.dtype(np.uint8),
        payload_nbytes,
        raw_dtype=tensor.tensor_type,
    )
    writer.tensors[-1][tensor.name].tensor = raw


def tensor_type_name(tensor) -> str:
    return getattr(tensor.tensor_type, "name", str(tensor.tensor_type)).lower()


def graft_mtp(base: Path, donor: Path, output: Path, *, name: str | None, mtp_policy: str | None) -> None:
    base_reader = GGUFReader(str(base))
    donor_reader = GGUFReader(str(donor))

    base_names = {tensor.name for tensor in base_reader.tensors}
    donor_tensor_indexes = [
        index for index, tensor in enumerate(donor_reader.tensors) if tensor.name.startswith("blk.64.")
    ]
    donor_tensors = [donor_reader.tensors[index] for index in donor_tensor_indexes]
    if not donor_tensors:
        raise RuntimeError("donor GGUF has no blk.64 MTP tensors")

    for tensor in donor_tensors:
        if tensor.name in base_names:
            raise RuntimeError(f"donor tensor already exists in base: {tensor.name}")

    arch_field = base_reader.fields.get("general.architecture")
    if arch_field is None:
        raise RuntimeError("base GGUF is missing general.architecture")
    arch = str(arch_field.contents())

    writer = GGUFWriter(str(output), arch=arch, use_temp_file=False)
    add_metadata(
        writer,
        base_reader,
        name=name,
        mtp_donor=str(donor),
        mtp_policy=mtp_policy or "graft Qwen3.6 blk.64 MTP block after core blend quantization",
    )

    for index, _ in enumerate(base_reader.tensors):
        add_tensor_exact(writer, base_reader, base, index)
    for index in donor_tensor_indexes:
        add_tensor_exact(writer, donor_reader, donor, index)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"wrote {output}")
    print(f"base_tensors={len(base_reader.tensors)} donor_tensors={len(donor_tensors)} total={len(base_reader.tensors) + len(donor_tensors)}")


def sidecar_names(weight_name: str) -> set[str]:
    if not weight_name.endswith(".weight"):
        return set()
    stem = weight_name[:-len(".weight")]
    return {f"{stem}.scale", f"{stem}.input_scale"}


def graft_replacements(
    base: Path,
    donor: Path,
    output: Path,
    *,
    donor_type: str,
    name: str | None,
    drop_sidecars: bool,
) -> None:
    base_reader = GGUFReader(str(base))
    donor_reader = GGUFReader(str(donor))

    wanted_type = donor_type.lower()
    donor_tensors = {
        tensor.name: index
        for index, tensor in enumerate(donor_reader.tensors)
        if tensor_type_name(tensor) == wanted_type
    }
    donor_tensor_names = {
        tensor.name
        for tensor in donor_reader.tensors
        if tensor_type_name(tensor) == wanted_type
    }
    if not donor_tensors:
        raise RuntimeError(f"donor GGUF has no tensors of type {donor_type}")

    base_names = {tensor.name for tensor in base_reader.tensors}
    replace_names = sorted(name for name in donor_tensor_names if name in base_names)
    if not replace_names:
        raise RuntimeError(f"donor type {donor_type} has no names matching base tensors")

    drop_names: set[str] = set()
    if drop_sidecars:
        for tensor_name in replace_names:
            drop_names.update(sidecar_names(tensor_name))

    arch_field = base_reader.fields.get("general.architecture")
    if arch_field is None:
        raise RuntimeError("base GGUF is missing general.architecture")
    arch = str(arch_field.contents())

    writer = GGUFWriter(str(output), arch=arch, use_temp_file=False)
    add_metadata(writer, base_reader, name=name)

    replaced = 0
    dropped = 0
    for index, tensor in enumerate(base_reader.tensors):
        if tensor.name in drop_names:
            dropped += 1
            continue
        if tensor.name in donor_tensors:
            add_tensor_exact(writer, donor_reader, donor, donor_tensors[tensor.name])
            replaced += 1
        else:
            add_tensor_exact(writer, base_reader, base, index)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"wrote {output}")
    print(
        f"base_tensors={len(base_reader.tensors)} donor_type={donor_type} "
        f"replaced={replaced} dropped_sidecars={dropped} total={len(base_reader.tensors) - dropped}"
    )


def main() -> None:
    register_local_quant_types()

    parser = argparse.ArgumentParser(description="Append MTP tensors or graft matching tensors from a donor GGUF")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--donor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--mtp-policy", default=None, help="metadata string for the grafted MTP policy")
    parser.add_argument("--replace-donor-type", default=None, help="replace matching base tensors with donor tensors of this GGML type")
    parser.add_argument("--drop-sidecars", action="store_true", help="drop .scale/.input_scale sidecars for replaced .weight tensors")
    args = parser.parse_args()

    if args.replace_donor_type:
        graft_replacements(
            args.base,
            args.donor,
            args.output,
            donor_type=args.replace_donor_type,
            name=args.name,
            drop_sidecars=args.drop_sidecars,
        )
    else:
        graft_mtp(args.base, args.donor, args.output, name=args.name, mtp_policy=args.mtp_policy)


if __name__ == "__main__":
    main()
