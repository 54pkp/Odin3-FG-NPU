"""Specialize the published ANVIL-S ONNX graph using its original weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


REFERENCE_SHA256 = "65e871aed25b331e3a3747cf9019942121a94af71997476f5804a24b768cd0d8"


# The official export captured image size only in the pad arithmetic and the final two slices.
def specialize(published: Path, width: int, height: int) -> onnx.ModelProto:
    if not (4 <= width <= 1920 and width % 4 == 0 and 4 <= height <= 1080 and height % 4 == 0):
        raise ValueError("Use image dimensions divisible by four within the published 1920x1080 size")
    sizes = {"/Constant": (1080, height), "/Constant_4": (1920, width),
             "/Constant_22": (1080, height), "/Constant_26": (1920, width)}
    content = published.read_bytes()
    if hashlib.sha256(content).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Expected the pinned published ANVIL-S 1080p ONNX")
    model = onnx.load_model_from_string(content)
    updated = set()
    for node in model.graph.node:
        if node.name in sizes:
            value = next(attribute for attribute in node.attribute if attribute.name == "value")
            original = numpy_helper.to_array(value.t)
            old, new = sizes[node.name]
            if original.size != 1 or original.item() != old:
                raise ValueError(f"Unexpected size constant: {node.name}")
            value.t.CopyFrom(numpy_helper.from_array(np.full(original.shape, new, dtype=original.dtype)))
            updated.add(node.name)
    if updated != sizes.keys():
        raise ValueError("Expected four shape-related constants")
    shape = model.graph.input[0].type.tensor_type.shape
    if [dim.dim_value for dim in shape.dim] != [1, 6, 1080, 1920]:
        raise ValueError("Unexpected published input shape")
    shape.dim[2].dim_value = height
    shape.dim[3].dim_value = width
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("published", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    model = specialize(args.published, args.width, args.height)
    with args.output.open("xb") as result:
        result.write(model.SerializeToString())


if __name__ == "__main__":
    main()
