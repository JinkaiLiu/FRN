try:
    import hdf5plugin
    import blosc
    HDF5_PLUGINS_AVAILABLE = True
except ImportError:
    HDF5_PLUGINS_AVAILABLE = False
    print("Warning: hdf5plugin or blosc not available. Some HDF5 files may not load correctly.")

import h5py
from pathlib import Path
import sys

print("✅ Checking t_offset units in all events_2x.h5...")

root = Path(sys.argv[1])
for file in root.rglob("events_2x.h5"):
    try:
        with h5py.File(file, "r") as f:
            if 't_offset' in f:
                offset = f['t_offset'][()]
                print(f"✔️ {file} -> t_offset: {offset}")
            else:
                print(f"⚠️ {file} -> no t_offset")
    except Exception as e:
        print(f"❌ {file} -> {e}")
