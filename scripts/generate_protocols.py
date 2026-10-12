"""Generate FlightGear protocol XML from the recorded schema."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landing.core import FIELDS


def build(direction, nodes):
    root = ET.Element("PropertyList")
    io = ET.SubElement(ET.SubElement(root, "generic"), direction)
    for name, value in (
        ("binary_mode", "false"),
        ("var_separator", ","),
        ("line_separator", "newline"),
    ):
        ET.SubElement(io, name).text = value
    for name, node in nodes.items():
        chunk = ET.SubElement(io, "chunk")
        ET.SubElement(chunk, "name").text = name
        ET.SubElement(chunk, "type").text = "double"
        ET.SubElement(chunk, "node").text = node
        if direction == "output":
            ET.SubElement(chunk, "format").text = "%.10f"
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "config/protocols"
    (root / "physical-ai-state.xml").write_text(build("output", FIELDS))
    nodes = {k: FIELDS[k] for k in ("aileron", "elevator", "rudder")}
    nodes.update(
        {f"throttle{i}": f"/controls/engines/engine[{i}]/throttle" for i in range(4)}
    )
    (root / "physical-ai-action.xml").write_text(build("input", nodes))
