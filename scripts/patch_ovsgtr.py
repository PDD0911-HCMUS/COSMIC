"""Apply API-only compatibility fixes for PyTorch 2.x to the vendored repo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "data/OvSGTR/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn"


def main():
    for filename in ("ms_deform_attn.h", "ms_deform_attn_cuda.cu"):
        path = CSRC / filename
        source = path.read_text(encoding="utf-8")
        updated = source.replace(".type().is_cuda()", ".is_cuda()")
        updated = updated.replace("AT_DISPATCH_FLOATING_TYPES(value.type(),", "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),")
        updated = updated.replace(".data<", ".data_ptr<")
        if updated != source:
            path.write_text(updated, encoding="utf-8")
            print(f"Updated PyTorch tensor API: {filename}")
    attention_path = CSRC.parents[1] / "ms_deform_attn.py"
    source = attention_path.read_text(encoding="utf-8")
    updated = source.replace("            grad_output,\n", "            grad_output.contiguous(),\n")
    if updated != source:
        attention_path.write_text(updated, encoding="utf-8")
        print("Ensure contiguous native backward gradient")
    print("OvSGTR compatibility patches ready; source paths are loaded by osg_builder.py")


if __name__ == "__main__":
    main()
