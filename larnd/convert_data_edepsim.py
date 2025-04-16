import h5py
import numpy as np
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import os

INPUT_DIR="/share/lustre/awilkins/contrastive_learning/extra_100kpp_for_new_electhrows/edep_h5_30kpp_alt"
OUTPUT_DIR="/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/edepsim/all"

def read_file(filename):
    f = h5py.File(filename, "r")
    segments = f["segments"]
    vertices = f["vertices"]
    event_ids = vertices["eventID"]

    return segments, vertices, event_ids

def save_event(event_id, i_event, filename, segments, vertices, for_training):
    segments_ev = segments[segments["eventID"] == event_id]
    vertices_ev = vertices[vertices["eventID"] == event_id]
    energy = np.array(segments_ev["dE"])

    coordinates = np.stack(
        [np.array(segments_ev["x"]), np.array(segments_ev["y"]), np.array(segments_ev["z"])],
        axis=1
    )
    vertex = np.stack(
        [vertices_ev["x_vert"], vertices_ev["y_vert"], vertices_ev["z_vert"]]
    ) / 10 # cm

    particle_name = filename.stem.split("_")[0]
    assert particle_name in ["electron", "gamma", "muon", "pion", "proton"]
    target_folder = os.path.join(output_path, particle_name)

    if len(coordinates) > 3:
        np.savez(f"{target_folder}/{filename.stem}_eventID_{event_id}.npz",
            adc=energy,
            coordinates=coordinates,
            charge=energy,
            vertex=vertex
        )

def process_file(filename, for_training=False):
    try:
        segments, vertices, event_ids = read_file(filename)
        for i_event, event_id in enumerate(event_ids):
            save_event(event_id, i_event, filename, segments, vertices, for_training)
    except ValueError as e:
        print(f"Error in file {filename}: {e}")
    except KeyError as e:
        print(f"Key error in file {filename}: {e}")
    except OSError as e:
        print(f"Failed to open {filename} with error:\n{e}")

if __name__ == "__main__":
    larnd_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    if not os.path.exists(output_path):
        raise ValueError(f"{output_path} does not exist!")
    if not os.path.exists(larnd_path):
        raise ValueError(f"{larnd_path} does not exist!")

    for particle_type in ["proton","electron", "pion", "muon", "gamma"]:

        if not os.path.exists(os.path.join(output_path, particle_type)):
            os.makedirs(os.path.join(output_path, particle_type))

        print(f"Processing {particle_type} files")
        print(larnd_path)
        files = list(larnd_path.glob(f"{particle_type}*.h5"))
        assert len(files) > 0, f"No files found for {particle_type}"
        process_map(process_file, files, max_workers=8)
