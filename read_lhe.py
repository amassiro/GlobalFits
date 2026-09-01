import gzip
import json
import re

PARTICLE_FIELDS = [
    "id", "status", "mother1", "mother2", "color1", "color2",
    "px", "py", "pz", "e", "m", "lifetime", "spin",
]

EVENT_RE = re.compile(r"<event(?:\s[^>]*)?>(.*?)</event>", re.DOTALL)
RWGT_RE = re.compile(r"<rwgt>(.*?)</rwgt>", re.DOTALL)
WGT_RE = re.compile(r"<wgt\s+id=['\"]([^'\"]+)['\"]\s*>([^<]*)</wgt>")


def _open_lhe(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _is_eft_weight(wid):
    return wid == "SM" or wid.startswith("c")


def _parse_event(block, eft_only=True):
    lines = [ln for ln in block.splitlines() if ln.strip()]
    header = lines[0].split()
    nup = int(float(header[0]))

    particles = []
    for line in lines[1: nup + 1]:
        vals = line.split()
        ints = [int(float(v)) for v in vals[:6]]
        floats = [float(v) for v in vals[6:13]]
        particles.append(dict(zip(PARTICLE_FIELDS, ints + floats)))

    weights = {}
    rwgt_match = RWGT_RE.search(block)
    if rwgt_match:
        for wid, val in WGT_RE.findall(rwgt_match.group(1)):
            if not eft_only or _is_eft_weight(wid):
                weights[wid] = float(val)

    return {
        "NUP": nup,
        "IDPRUP": int(float(header[1])),
        "XWGTUP": float(header[2]),
        "SCALUP": float(header[3]),
        "AQEDUP": float(header[4]),
        "AQCDUP": float(header[5]),
        "particles": particles,
        "weights": weights,
    }


def lhe_to_json(lhe_path, json_path=None, eft_only=True):
    if json_path is None:
        json_path = re.sub(r"\.lhe(\.gz)?$", ".json", lhe_path)

    with _open_lhe(lhe_path) as f:
        text = f.read()

    events = [_parse_event(block, eft_only) for block in EVENT_RE.findall(text)]

    with open(json_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"wrote {len(events)} events to {json_path}")
    return json_path


if __name__ == "__main__":
    import sys
    lhe_to_json(sys.argv[1])
