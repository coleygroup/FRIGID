""" assign_subformulae.py

Given a set of spectra and candidates from a labels file, assign subformulae and save to JSON files.

"""

from pathlib import Path
import argparse
from functools import partial
import numpy as np
import pandas as pd
import json
from pathos import multiprocessing as pmp

from tqdm import tqdm
from mist import utils
from mist.utils import get_subdir

def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-id",
        default="ID",
        help="ID key in mgf input"
    )
    parser.add_argument(
        "--spec-files",
        default="data/paired_spectra/canopus_train/spec_files/",
        help="Spec files; either MGF or directory.",
    )
    parser.add_argument("--output-dir", default=None,
                        help="Name of output dir.")
    parser.add_argument(
        "--labels-file",
        default="data/paired_spectra/canopus_train/labels.tsv",
        help="Labels file",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug flag."
    )
    parser.add_argument(
        "--mass-diff-type",
        default="ppm",
        type=str,
        help="Type of mass difference - absolute differece (abs) or relative difference (ppm).",
    )
    parser.add_argument(
        "--mass-diff-thresh",
        action="store",
        default=20,
        type=float,
        help="Threshold of mass difference.",
    )
    parser.add_argument(
        "--inten-thresh",
        action="store",
        default=0.001,
        type=float,
        help="Threshold of MS2 subpeak intensity (normalized to 1).",
    )
    parser.add_argument(
        "--max-formulae",
        action="store",
        default=50,
        type=int,
        help="Max number of peaks to keep",
    )
    parser.add_argument(
        "--num-workers", action="store", default=32, type=int, help="num workers"
    )
    return parser.parse_args()


def process_spec_file(spec_name: str, spec_files: str, max_inten=0.001, max_peaks=60):
    """_summary_

    Args:
        spec_name (str): _description_
        spec_files (str): _description_
        max_inten (float, optional): _description_. Defaults to 0.001.
        max_peaks (int, optional): _description_. Defaults to 60.

    Returns:
        _type_: _description_
    """
    # Try sharded path first, fall back to flat layout
    subdir = get_subdir(spec_name)
    spec_file = Path(spec_files) / subdir / f"{spec_name}.ms"
    if not spec_file.exists():
        spec_file = Path(spec_files) / f"{spec_name}.ms"

    meta, tuples = utils.parse_spectra(spec_file)
    spec = utils.process_spec_file(meta, tuples)
    return spec_name, spec


def assign_subforms(spec_files, labels_file,
                    mass_diff_thresh: int = 20,
                    mass_diff_type: str = "ppm",
                    inten_thresh: float = 0.001,
                    output_dir=None,
                    num_workers: int = 32,
                    feature_id="ID",
                    max_formulae: int = 50,
                    debug=False):
    """_summary_

    Args:
        spec_files (_type_): _description_
        labels_file (_type_): _description_
        mass_diff_thresh (int, optional): _description_. Defaults to 20.
        mass_diff_type (str, optional): _description_. Defaults to "ppm".
        inten_thresh (float, optional): _description_. Defaults to 0.001.
        output_dir (_type_, optional): _description_. Defaults to None.
        num_workers (int, optional): _description_. Defaults to 32.
        feature_id (str, optional): _description_. Defaults to "ID".
        max_formulae (int, optional): _description_. Defaults to 50.
        debug (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
    """
    spec_files = Path(spec_files)
    label_path = Path(labels_file)

    # Read in labels
    labels_df = pd.read_csv(label_path, sep="\t").astype(str)
    if debug:
        labels_df = labels_df[:50]

    # Define output directory name
    output_dir = Path(output_dir)
    if output_dir is None:
        subform_dir = label_path.parent / "subformulae"
        output_dir_name = f"subform_{max_formulae}"
        output_dir = subform_dir / output_dir_name

    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(256):
        (output_dir / f"{i:03d}").mkdir(exist_ok=True)

    cpus = min(pmp.cpu_count(), max(num_workers, 1))

    if spec_files.suffix == ".mgf":
        # Input specs
        parsed_specs = utils.parse_spectra_mgf(spec_files)
        input_specs = [utils.process_spec_file(*i) for i in parsed_specs]
        spec_names = [i[0][feature_id] for i in parsed_specs]
        input_specs = list(zip(spec_names, input_specs))
    elif spec_files.is_dir():
        spec_fn_lst = labels_df["spec"].to_list()
        proc_spec_full = partial(
            process_spec_file,
            spec_files=spec_files,
            max_inten=inten_thresh,
            max_peaks=max_formulae,
        )
        # input_specs = [proc_spec_full(i) for i in tqdm(spec_fn_lst)]
        pool = pmp.Pool(processes=cpus)
        input_specs = list(tqdm(
            pool.imap(proc_spec_full, spec_fn_lst, chunksize=32768),
            total=len(spec_fn_lst),
            desc="Loading spectra",
        ))
        pool.close()
        pool.join()
    else:
        raise ValueError(f"Spec files arg {spec_files} is not a dir or mgf")

    # input_specs contains a list of tuples (spec name, [msz, intens] list)) where msz/intens have been filtered
    input_specs_dict = {tup[0]: tup[1] for tup in input_specs}
    export_dicts, spec_names = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Building export dicts"):
        spec = str(row["spec"])
        new_entry = {
            "spec": input_specs_dict[spec],
            "form": row["formula"],
            "mass_diff_type": mass_diff_type,
            "spec_name": spec,
            "mass_diff_thresh": mass_diff_thresh,
            "ion_type": row["ionization"],
        }
        spec_names.append(spec)
        export_dicts.append(new_entry)

    # Assign subformulae
    print(f"There are {len(export_dicts)} spec-cand pairs this spec files")
    def export_wrapper(x): return utils.get_output_dict(**x)
    if debug:
        output_dict_lst = [export_wrapper(i) for i in tqdm(export_dicts[:10], desc="Assigning subformulae (debug)")]
    else:
        pool = pmp.Pool(processes=cpus)
        output_dict_lst = list(tqdm(
            pool.imap(export_wrapper, export_dicts, chunksize=500),
            total=len(export_dicts),
            desc="Assigning subformulae",
        ))
        pool.close()
        pool.join()
    assert len(export_dicts) == len(output_dict_lst)

    # Write all output jsons to sharded subdirectories
    for output_dict, spec_name in tqdm(zip(output_dict_lst, spec_names), total=len(spec_names), desc="Writing JSON files"):
        subdir = get_subdir(spec_name)
        with open(output_dir / subdir / f"{spec_name}.json", "w") as f:
            json.dump(output_dict, f, indent=4)

if __name__ == "__main__":
    args = get_args()
    assign_subforms(spec_files=args.spec_files, 
                    labels_file=args.labels_file,
                    mass_diff_thresh=args.mass_diff_thresh,
                    mass_diff_type=args.mass_diff_type,
                    inten_thresh=args.inten_thresh,
                    output_dir=args.output_dir,
                    num_workers=args.num_workers,
                    feature_id=args.feature_id,
                    max_formulae=args.max_formulae,
                    debug=args.debug)