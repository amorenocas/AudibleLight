#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates an example dataset for SELD, similar to DCASE 2023 Task 3 and [this repo](https://zenodo.org/records/6406873):

By default, this script can generate:
- 4-seconds long spatial recordings
- Sampling rate of 48kHz
- One 8-channel recording format, 8-linear microphone array (MIC) with [2, 2, 2, 14, 2, 2, 2] cm inter-microphone spacing
- Spatial events spatialized in N unique spaces, using ray-traced RIRs
- Maximum polyphony of 1 (with possible same-class events overlapping)

A single augmentation can be to every audio file, sampled randomly from:
- Pitch shifting (+/- up to half an octave)
- Time stretching (between 0.9 and 1.1x)
- Distortion (up to +10 dB gain)
- Reverse
- Phase inversion

The backend is fixed to RLR (ray-tracing). When using this backend,
materials can also be added to the simulation.

Many other parameters can be controlled using flags passed in to the script.
"""

import argparse
import gc
import json
import os
import random
from pathlib import Path
from time import time

import numpy as np
from loguru import logger
from scipy import stats
from tqdm import tqdm

from audiblelight import config, utils
from audiblelight.augmentation import Distortion, Invert, PitchShift, Reverse, SpeedUp
from audiblelight.core import Scene
from audiblelight.worldstate import MATERIALS_JSON
from scripts.seld.seld_dataset_assets import MESHES

# For reproducible randomisation
utils.seed_everything(utils.SEED)

# Filepaths, directories, etc.
FG_DIR = "/data/train-clean-100"
MESH_DIR = utils.get_project_root() / "resources/meshes/gibson"
OUTPUT_DIR = utils.get_project_root() / "spatial_scenes_dcase_synthetic"

# Parameters
DURATION = 4
SAMPLE_RATE = 48000

# Valid materials for the ray-tracing engine
with open(MATERIALS_JSON, "r") as js_in:
    js_out = json.load(js_in)
VALID_MATERIALS = list({mat["name"] for mat in js_out["materials"]})

AUGMENTATIONS = {
    "pitchshift": (
        PitchShift,
        dict(sample_rate=SAMPLE_RATE, semitones=stats.uniform(-7, 0)),
    ),
    "speedup": (
        SpeedUp,
        dict(sample_rate=SAMPLE_RATE, stretch_factor=stats.uniform(0.9, 0.2)),
    ),
    "reverse": Reverse,
    "invert": Invert,
    "distortion": (
        Distortion,
        dict(sample_rate=SAMPLE_RATE, drive_db=stats.uniform(0.0, 10.0)),
    ),
}


def get_augmentations(augmentation_names: list[str]) -> list:
    """
    Given a list of augmentation names as strings, grab them from the AUGMENTATIONS dictionary
    """
    grabbed = []
    for aug in augmentation_names:
        if aug in AUGMENTATIONS.keys():
            grabbed.append(AUGMENTATIONS[aug])
        else:
            raise ValueError(
                f"Augmentation {aug} is not a valid parameter for this script!"
            )
    return grabbed


def generate(
    asset_name: str,
    split: str,
    scene_num: int,
    scape_num: int,
    output_dir: Path,
    augmentations: list[str],
    materials: bool,
    max_overlap: int,
    min_events_static,
    max_events_static: int,
) -> None:
    """
    Make a single generation with required arguments
    """
    # Output filepaths
    fold = 1 if split == "train" else 2
    common = f"dev-{split}-alight/fold{fold}_scene{scene_num}_{str(scape_num).zfill(3)}"
    audio_path = output_dir / f"dev/{common}.wav"
    metadata_path = output_dir / f"metadata_dev/{common}.csv"
    metadata_doa_path = output_dir / f"metadata_dev/{common}_doa.json"

    # Skip over this generation if files already exist
    if (
        audio_path.with_name(audio_path.stem + ".wav").exists()
        and metadata_path.with_name(metadata_path.stem + ".csv").exists()
    ):
        return None

    # Choose a noise floor for the scene
    scene_ref_db = np.random.uniform(config.MIN_REF_DB, config.MAX_REF_DB)

    # Use augmentations
    if augmentations:
        use_augmentations = get_augmentations(augmentations)
    else:
        use_augmentations = None

    # Resolve full kwargs for backend
    backend_kwargs = dict(
        add_to_context=False,
        material=random.choice(VALID_MATERIALS) if materials else "Default",
        mesh=MESH_DIR / asset_name,
    )

    scene = Scene(
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        backend="rlr",
        scene_start_dist=lambda size=None: 0.0,
        # Audio files will always start from 0 seconds in
        event_start_dist=None,
        # Events capped to 10 seconds
        event_duration_dist=lambda size=None: DURATION,
        # Events have speed between 0.5 and 2.0 metres-per-second
        event_velocity_dist=lambda size=None: 0.0,
        # Events have resolution between 1.0 and 4.0 Hz
        event_resolution_dist=stats.uniform(
            config.MIN_EVENT_RESOLUTION,
            config.MAX_EVENT_RESOLUTION - config.MIN_EVENT_RESOLUTION,
        ),
        # Events have SNR between 5 and 30 dB
        snr_dist=stats.uniform(
            config.MIN_EVENT_SNR, config.MAX_EVENT_SNR - config.MIN_EVENT_SNR
        ),
        # Event augmentations will sample from this list
        event_augmentations=use_augmentations,
        fg_path=Path(FG_DIR),
        max_overlap=max_overlap,
        ref_db=scene_ref_db,
        backend_kwargs=backend_kwargs,
        allow_duplicate_audios=False,
    )

    # Initialise the Scene with all arguments
    for attempt_microphone in range(10):
        try:
            # Add the microphone to ray-tracing/parameterized backends
            scene.clear_microphones()
            scene.add_microphone(
                microphone_type="spatialcodec",
                alias="mic",
                # position=scene.state.mesh.bounding_box.centroid,
            )

            for attempt_event in range(20):
                # Sample random event position in polar coordinates
                event_azimuth = np.random.uniform(-90, 90)
                event_elevation = np.clip(np.random.normal(0, 5), -30, 30)
                event_distance = np.random.uniform(1, 3)

                try:
                    scene.add_event(
                        event_type="static",
                        augmentations=1 if use_augmentations else None,
                        ensure_direct_path=True,
                        max_place_attempts=1,
                        scene_start=0.0,
                        event_start=0.0,
                        duration=DURATION,
                        position=[event_azimuth, event_elevation, event_distance],
                        polar=True,
                    )
                    # If we reach here, event was added successfully
                    break  # stop trying more positions for this mic
                except ValueError as e:
                    logger.warning(
                        f"Event positioning attempt {attempt_event + 1} / {20} failed: {e}"
                    )
                    continue

            # If no events added successfully, try again by calling the function recursively
            if len(scene.get_events()) != 0:
                # Always add gaussian noise
                scene.add_ambience(noise="gaussian")
                scene.generate(
                    audio_fname=audio_path,
                    metadata_fname=metadata_path,
                    audio=True,
                    metadata_json=True,
                    metadata_dcase=False,
                )
                with open(metadata_doa_path, "w") as f:
                    json.dump(
                        {
                            "azimuth": event_azimuth,
                            "elevation": event_elevation,
                            "distance": event_distance,
                        },
                        f,
                        indent=2,
                    )
                break
            continue
        except ValueError as e:
            logger.warning(
                f"Microphone positioning attempt {attempt_microphone + 1} / {10} failed: {e}"
            )
    del scene
    gc.collect()
    return None


def get_assets(asset_split: str) -> dict:
    """
    Get the train + test meshes for this backend and split
    """
    if str(asset_split) not in MESHES.keys():
        raise ValueError(
            f"Expected meshes in {list(MESHES.keys())} but got {asset_split}"
        )
    return MESHES[str(asset_split)]


def main(
    augmentations: bool,
    materials: bool,
    assets: str,
    outdir: str,
    max_overlap: int,
    min_events_static: int,
    max_events_static: int,
):
    """
    Runs the generation across all training + test rooms
    """

    # Create the output folders if they don't currently exist
    outdir = Path(outdir)
    for fp in [
        outdir / "metadata_dev/dev-train-alight",
        outdir / "metadata_dev/dev-test-alight",
        outdir / "dev/dev-train-alight",
        outdir / "dev/dev-test-alight",
    ]:
        if not fp.exists():
            os.makedirs(fp)

    # Get the train + test meshes for this run
    chosen = get_assets(assets)
    train_rooms = chosen["train"]
    test_rooms = chosen["test"]
    train_recordings_per_room = chosen["scapes_per_train_mesh"]
    test_recordings_per_room = chosen["scapes_per_test_mesh"]

    # Start iterating to create the required number of training scenes
    logger.info("Generating training scenes...")
    full_start = time()
    for train_room_idx, train_room in enumerate(train_rooms):
        for train_scape_idx in tqdm(
            range(train_recordings_per_room),
            desc=f"Generating for train room {train_room_idx + 1}/{len(train_rooms)}, name {train_room}...",
        ):
            generate(
                train_room,
                "train",
                train_room_idx,
                train_scape_idx,
                outdir,
                augmentations,
                materials,
                max_overlap,
                min_events_static,
                max_events_static,
            )

    logger.info("Generating testing scenes...")
    for test_room_idx, test_room in enumerate(test_rooms):
        for test_scape_idx in tqdm(
            range(test_recordings_per_room),
            desc=f"Generating for test room {test_room_idx + 1}/{len(test_rooms)}, name {test_room}...",
        ):
            generate(
                test_room,
                "test",
                test_room_idx,
                test_scape_idx,
                outdir,
                augmentations,
                materials,
                max_overlap,
                min_events_static,
                max_events_static,
            )

    # Log the time taken
    full_end = time() - full_start
    logger.info(f"Finished in {full_end:.4f} seconds.")


if __name__ == "__main__":
    # Use module docstring for the help text
    parser = argparse.ArgumentParser(description=__doc__)

    # Here come the user parameters
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        help=f"The name of augmentations to use: supported are {', '.join(AUGMENTATIONS.keys())}",
        default=None,
    )
    parser.add_argument(
        "--materials",
        action="store_true",
        help="Add this flag to use materials with 'backend=rlr'",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="9A",
        help="The data files (.glb meshes or .sofa) to use: see `seld_dataset_assets.py`. "
        "Note that the total number of scapes to generate will remain fixed at 1200.",
    )
    parser.add_argument(
        "--outdir",
        type=int,
        default=OUTPUT_DIR,
        help=f"Path to save generated outputs, defaults to {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=1,
        help="Maximum number of overlapping events, defaults to 1",
    )
    parser.add_argument(
        "--min-events-static",
        type=int,
        default=1,
        help="Minimum number of static events per scene, defaults to 1",
    )
    parser.add_argument(
        "--max-events-static",
        type=int,
        default=1,
        help="Maximum number of static events per scene, defaults to 1",
    )

    # Parse args and start generating
    args = vars(parser.parse_args())
    logger.info("Generating with args: {}".format(args))

    main(**args)
