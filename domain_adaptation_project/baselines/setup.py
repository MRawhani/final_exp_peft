import sys
from pathlib import Path

def setup_src_path():
    # in jupyter (lab / notebook), based on notebook path
    module_path = str(Path.cwd().parents[0] / "modules")
    if module_path not in sys.path:
        sys.path.append(module_path)

    return sys.path
