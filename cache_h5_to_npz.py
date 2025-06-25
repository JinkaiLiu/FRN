import os
from pathlib import Path
import numpy as np

try:
    import hdf5plugin
    import blosc
    HDF5_PLUGINS_AVAILABLE = True
except ImportError:
    HDF5_PLUGINS_AVAILABLE = False
    print("Warning: hdf5plugin or blosc not available. Some HDF5 files may not load correctly.")

import h5py

from tqdm import tqdm


def convert_h5_to_npz(events_h5_path: Path, output_npz_path: Path):
    try:
        with h5py.File(events_h5_path, 'r') as f:
            x = np.array(f['events/x'], copy=True)
            y = np.array(f['events/y'], copy=True)
            t = np.array(f['events/t'], copy=True)
            p = np.array(f['events/p'], copy=True)

            if 't_offset' in f:
                t_offset = f['t_offset'][()]
                print(f"[{events_h5_path.name}] ✅ Apply t_offset: {t_offset}")
                t += t_offset

        np.savez_compressed(output_npz_path, x=x, y=y, t=t, p=p)
        return True
    except Exception as e:
        print(f"❌ Failed to convert {events_h5_path.name}: {e}")
        return False


def process_dataset(root_dir):
    root = Path(root_dir)
    splits = ['train', 'test']
    total_converted = 0

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue

        for sequence in split_dir.iterdir():
            events_h5 = sequence / 'events' / 'left' / 'events_2x.h5'
            output_npz = events_h5.with_suffix('.npz')

            if not events_h5.exists():
                print(f"⚠️  Missing: {events_h5}")
                continue

            if output_npz.exists():
                print(f"✅ Already exists: {output_npz.name}")
                continue

            print(f"🔁 Converting: {events_h5}")
            success = convert_h5_to_npz(events_h5, output_npz)
            if success:
                print(f"✅ Saved: {output_npz.name}")
                total_converted += 1

    print(f"\n🎉 Total converted: {total_converted}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Path to root of DSEC-Det dataset")
    args = parser.parse_args()

    process_dataset(args.root_dir)
