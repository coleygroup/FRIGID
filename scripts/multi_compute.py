import os
import glob
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pulp
from myopic_mces import MCES

try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform('1,3 heteroatom H shift',
                          '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator  # newer rdkit
    _RD_TAUTOMER_CANONICALIZER = 'v2'


def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi
    Note that this function may be 50 times slower than Chem.MolFromInchi"""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == 'v1':
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def is_valid(mol):
    smiles = mol2smiles(mol)
    if smiles is None:
        return False

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except:
        return False
    if len(mol_frags) > 1:
        return False
    
    return True


def remove_stereo_from_mol(mol):
    """Remove stereochemistry information from a molecule."""
    if mol is None:
        return None
    mol = Chem.RWMol(mol)
    Chem.RemoveStereochemistry(mol)
    return mol.GetMol()


def mol_to_inchi_no_stereo(mol):
    """Convert molecule to InChI without stereochemistry."""
    if mol is None:
        return None
    mol_no_stereo = remove_stereo_from_mol(mol)
    return Chem.MolToInchi(mol_no_stereo)


def get_molecular_formula(mol):
    """Get molecular formula from RDKit mol object."""
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.CalcMolFormula(mol)
    except:
        return None


def compute_metrics_for_one(t_smiles, p_smiles_list, solver=None, doMCES=False, doFull=False, filter_formula=False):
    """
    Worker function to compute metrics for a single true molecule and its predictions.
    Now handles preprocessing (Mol conversion, InChI generation, Filtering) internally.
    """
    RDLogger.DisableLog('rdApp.*')

    # --- Preprocessing Step (Moved inside worker) ---
    # Process True Molecule
    if t_smiles is None:
        return defaultdict(float), [], [], [], []
        
    t_mol_init = Chem.MolFromSmiles(t_smiles)
    if t_mol_init is None:
        return defaultdict(float), [], [], [], []

    t_inchi = Chem.MolToInchi(t_mol_init)
    
    true_formula = None
    if filter_formula:
        true_formula = get_molecular_formula(t_mol_init)

    # Process Predictions
    p_inchi = []
    p_inchi_set = set()

    for ps in p_smiles_list:
        if pd.isna(ps) or ps is None: continue
        
        pmol_init = Chem.MolFromSmiles(ps)
        if pmol_init is None or not is_valid(pmol_init):
            continue
            
        if filter_formula and true_formula is not None:
            p_formula = get_molecular_formula(pmol_init)
            if p_formula != true_formula:
                continue

        pi = Chem.MolToInchi(pmol_init)
        if pi not in p_inchi_set:
            p_inchi_set.add(pi)
            p_inchi.append(pi)
    
    if not doFull:
        p_inchi = p_inchi[:11]

    # --- Metric Computation Step ---
    rdkit_topological_fgpts = dict()
    fpgen = AllChem.GetRDKitFPGenerator()

    def get_rdkit_fingerprint(mol):    
        if mol not in rdkit_topological_fgpts:
            return fpgen.GetFingerprint(mol)
        else:
            return rdkit_topological_fgpts[mol]

    true_mol = canonical_mol_from_inchi(t_inchi)
    # Remove stereochemistry for all comparisons
    true_mol_no_stereo = remove_stereo_from_mol(true_mol)
    true_inchi_no_stereo = Chem.MolToInchi(true_mol_no_stereo) if true_mol_no_stereo else None
    
    true_smi = Chem.MolToSmiles(true_mol_no_stereo)
    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol_no_stereo, 2, nBits=2048)
    true_rdkit_fp = get_rdkit_fingerprint(true_mol_no_stereo)
    true_num_bonds = true_mol_no_stereo.GetNumBonds()

    # Precompute metrics for each predicted molecule
    p_mces = []
    p_tanimoto = []
    p_rdkit_tanimoto = []
    p_cosine = []
    p_inchi_no_stereo = []  # Store stereo-free InChIs for exact match comparison
    
    for pi in p_inchi:
        try:
            pmol = canonical_mol_from_inchi(pi)
            # Remove stereochemistry for all comparisons
            pmol_no_stereo = remove_stereo_from_mol(pmol)
            pi_no_stereo = Chem.MolToInchi(pmol_no_stereo) if pmol_no_stereo else None
            p_inchi_no_stereo.append(pi_no_stereo)
            
            try:
                pmol_smi = Chem.MolToSmiles(pmol_no_stereo)
                if doMCES:
                    p_mces.append(MCES(true_smi, pmol_smi, solver=solver, threshold=15, always_stronger_bound=True, solver_options=dict(msg=0, timeLimit=600))[1])
                else:
                    p_mces.append(true_num_bonds + pmol_no_stereo.GetNumBonds())
            except:
                try:
                    p_mces.append(true_num_bonds + pmol_no_stereo.GetNumBonds())
                except:
                    p_mces.append(2*true_num_bonds)

            try:
                pmol_fp = AllChem.GetMorganFingerprintAsBitVect(pmol_no_stereo, 2, nBits=2048)
                pred_rdkit_fp = get_rdkit_fingerprint(pmol_no_stereo)
                p_rdkit_tanimoto.append(DataStructs.TanimotoSimilarity(true_rdkit_fp, pred_rdkit_fp))
                p_tanimoto.append(DataStructs.TanimotoSimilarity(true_fp, pmol_fp))
                p_cosine.append(DataStructs.CosineSimilarity(true_fp, pmol_fp))
            except:
                p_tanimoto.append(0.0)
                p_rdkit_tanimoto.append(0.0)
                p_cosine.append(0.0)
        except:
            p_tanimoto.append(0.0)
            p_rdkit_tanimoto.append(0.0)
            p_cosine.append(0.0)
            p_mces.append(2*true_num_bonds)
            p_inchi_no_stereo.append(None)

    # Build prefix arrays for best (min) MCES, best (max) Tanimoto, best (max) Cosine
    prefix_min_mces = [100]
    prefix_max_tanimoto = [0.0]
    prefix_max_rdkit_tanimoto = [0.0]
    prefix_max_cosine = [0.0]
    for j in range(len(p_inchi)):
        prefix_min_mces.append(min(prefix_min_mces[-1], p_mces[j]))
        prefix_max_tanimoto.append(max(prefix_max_tanimoto[-1], p_tanimoto[j]))
        prefix_max_rdkit_tanimoto.append(max(prefix_max_rdkit_tanimoto[-1], p_rdkit_tanimoto[j]))
        prefix_max_cosine.append(max(prefix_max_cosine[-1], p_cosine[j]))

    # Earliest index of true InChI (without stereochemistry), if present
    earliest_idx = -1
    for idx, pi_no_stereo in enumerate(p_inchi_no_stereo):
        if pi_no_stereo is not None and pi_no_stereo == true_inchi_no_stereo:
            earliest_idx = idx
            break

    if doFull:
        # Compute metrics using prefix arrays
        m_local = defaultdict(float)
        for k in range(1, 101):
            m_local[f'acc@{k}'] = 1.0 if (earliest_idx != -1 and earliest_idx < k) else 0.0
            idx = min(k, len(p_inchi))
            m_local[f'mces@{k}'] = prefix_min_mces[idx]
            m_local[f'tanimoto@{k}'] = prefix_max_tanimoto[idx]
            m_local[f'rdkit_tanimoto@{k}'] = prefix_max_rdkit_tanimoto[idx]
            m_local[f'cosine@{k}'] = prefix_max_cosine[idx]
            m_local[f'close_match@{k}'] = 1.0 if (prefix_max_rdkit_tanimoto[idx] >= 0.675) else 0.0
            m_local[f'meaningful_match@{k}'] = 1.0 if (prefix_max_rdkit_tanimoto[idx] >= 0.4) else 0.0
    else:
        m_local = defaultdict(float)
        for k in range(1, 11):
            m_local[f'acc@{k}'] = 1.0 if (earliest_idx != -1 and earliest_idx < k) else 0.0
            idx = min(k, len(p_inchi))
            m_local[f'mces@{k}'] = prefix_min_mces[idx]
            m_local[f'tanimoto@{k}'] = prefix_max_tanimoto[idx]
            m_local[f'rdkit_tanimoto@{k}'] = prefix_max_rdkit_tanimoto[idx]
            m_local[f'cosine@{k}'] = prefix_max_cosine[idx]
            m_local[f'close_match@{k}'] = 1.0 if (prefix_max_rdkit_tanimoto[idx] >= 0.675) else 0.0
            m_local[f'meaningful_match@{k}'] = 1.0 if (prefix_max_rdkit_tanimoto[idx] >= 0.4) else 0.0

    return m_local, prefix_min_mces, prefix_max_tanimoto, prefix_max_rdkit_tanimoto, prefix_max_cosine


def compute_metrics(true_smiles, pred_smiles_lists, csv_path, doMCES=False, doFull=False, filter_formula=False):
    """
    Compute metrics for predictions.
    
    Args:
        true_smiles: List of true molecule SMILES strings
        pred_smiles_lists: List of lists of prediction SMILES strings
        csv_path: Path to save metrics CSV
        doMCES: Whether to compute MCES (slow)
        doFull: Whether to compute full metrics (k=1 to 100)
        filter_formula: Whether to filter predictions by formula match
    
    Returns:
        dict: Final metrics
    """
    
    if doMCES:
        solver = pulp.listSolvers(onlyAvailable=True)[0]
    else:
        solver = None

    # Parallelize including the expensive preprocessing
    with tqdm_joblib(tqdm(total=len(true_smiles))) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(compute_metrics_for_one)(
                true_smiles[i],
                pred_smiles_lists[i],
                solver,
                doMCES=doMCES,
                doFull=doFull,
                filter_formula=filter_formula
            )
            for i in range(len(true_smiles))
        )

    # Separate per-sample metrics from aggregate metrics
    per_sample_mces = []
    per_sample_tanimoto = []
    per_sample_rdkit_tanimoto = []
    per_sample_cosine = []
    
    # aggregate results
    final_metrics = defaultdict(float)
    for r in results:
        m_local, prefix_mces, prefix_tanimoto, prefix_rdkit_tanimoto, prefix_cosine = r
        per_sample_mces.append(prefix_mces)
        per_sample_tanimoto.append(prefix_tanimoto)
        per_sample_rdkit_tanimoto.append(prefix_rdkit_tanimoto)
        per_sample_cosine.append(prefix_cosine)
        
        for key, val in m_local.items():
            final_metrics[key] += val

    num_samples = len(true_smiles)
    if doFull:
        for k in range(1, 101):
            final_metrics[f'acc@{k}'] /= num_samples
            final_metrics[f'mces@{k}'] /= num_samples
            final_metrics[f'tanimoto@{k}'] /= num_samples
            final_metrics[f'rdkit_tanimoto@{k}'] /= num_samples
            final_metrics[f'cosine@{k}'] /= num_samples
            final_metrics[f'close_match@{k}'] /= num_samples
            final_metrics[f'meaningful_match@{k}'] /= num_samples
            
            # Compute medians
            final_metrics[f'median_mces@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_mces])
            final_metrics[f'median_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_tanimoto])
            final_metrics[f'median_rdkit_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_rdkit_tanimoto])
            final_metrics[f'median_cosine@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_cosine])
    else:
        for k in range(1, 11):
            final_metrics[f'acc@{k}'] /= num_samples
            final_metrics[f'mces@{k}'] /= num_samples
            final_metrics[f'tanimoto@{k}'] /= num_samples
            final_metrics[f'rdkit_tanimoto@{k}'] /= num_samples
            final_metrics[f'cosine@{k}'] /= num_samples
            final_metrics[f'close_match@{k}'] /= num_samples
            final_metrics[f'meaningful_match@{k}'] /= num_samples
            
            # Compute medians
            final_metrics[f'median_mces@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_mces])
            final_metrics[f'median_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_tanimoto])
            final_metrics[f'median_rdkit_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_rdkit_tanimoto])
            final_metrics[f'median_cosine@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_cosine])

    df = pd.DataFrame(final_metrics, index=[0])
    df.to_csv(csv_path, index=False)

    filter_str = " (formula filtered)" if filter_formula else ""
    print(f"Metrics{filter_str}:")
    print(df[["acc@1", "mces@1", "tanimoto@1", "meaningful_match@1", "close_match@1"]])
    print(df[["acc@10", "mces@10", "tanimoto@10", "meaningful_match@10", "close_match@10"]])
    print("\nMedian metrics:")
    print(df[["median_mces@1", "median_tanimoto@1", "median_rdkit_tanimoto@1"]])
    print(df[["median_mces@10", "median_tanimoto@10", "median_rdkit_tanimoto@10"]])

    return final_metrics


def find_subdirs(base_path):
    """Find all subdirectories in the base path."""
    subdirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    return sorted(subdirs)


def find_round_files(base_path):
    """
    Find predictions_round_K.csv files across subdirectories.
    Implements fallback logic: if round K is missing in a subdir, use max round < K.
    
    Returns:
        dict: Mapping from round number (int) to list of file paths
    """
    subdirs = find_subdirs(base_path)
    print(f"Found {len(subdirs)} subdirectories")
    
    # 1. Index all available files
    # subdir_path -> {round_num: file_path}
    subdir_files = defaultdict(dict)
    all_seen_rounds = set()
    
    for subdir in subdirs:
        pattern = os.path.join(subdir, "predictions_round_*.csv")
        files = glob.glob(pattern)
        for f in files:
            try:
                # Extract K from predictions_round_K.csv
                r = int(os.path.basename(f).split('_')[-1].split('.')[0])
                subdir_files[subdir][r] = f
                all_seen_rounds.add(r)
            except ValueError:
                continue
                
    # 2. Build results using fallback logic
    round_files_map = defaultdict(list)
    sorted_rounds = sorted(list(all_seen_rounds))
    
    for r in sorted_rounds:
        for subdir in subdirs:
            # Try to find exact round match
            if r in subdir_files[subdir]:
                round_files_map[r].append(subdir_files[subdir][r])
            else:
                # Fallback: Find largest available round less than r
                available_rounds = [k for k in subdir_files[subdir].keys() if k < r]
                if available_rounds:
                    fallback_r = max(available_rounds)
                    round_files_map[r].append(subdir_files[subdir][fallback_r])
                    # Optional: Print warning if debugging needed
                    # print(f"Subdir {os.path.basename(subdir)} missing round {r}, using {fallback_r}")
                # If no previous round exists (e.g. requesting round 0 but only round 5 exists?), skip
    
    return round_files_map


def concat_round_predictions(file_paths):
    """
    Concatenate predictions from multiple CSV files for the same round.
    
    Args:
        file_paths: List of paths to predictions_round_K.csv files
    
    Returns:
        DataFrame with concatenated predictions
    """
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    if not dfs:
        return None
    
    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined {len(file_paths)} files -> {len(combined)} total spectra")
    
    return combined


def process_round(round_num, file_paths, output_dir, doFull=False, enable_formula_filter=False):
    """
    Process predictions for a single round.
    """
    print(f"\n{'='*70}")
    print(f"Processing Round {round_num}")
    print(f"{'='*70}")
    
    # Concatenate all predictions for this round
    df = concat_round_predictions(file_paths)
    if df is None:
        print(f"No data found for round {round_num}")
        return None, None
    
    # Extract true SMILES and predicted SMILES as raw strings
    # We delay Mol conversion to the parallel workers
    true_smiles = df['true_smiles'].tolist()
    pred_cols = [col for col in df.columns if col.startswith('pred_smiles_')]
    pred_smiles_lists = df[pred_cols].values.tolist()
    
    # Compute metrics without formula filtering
    print("\n--- Without formula filtering ---")
    output_csv_no_filter = os.path.join(output_dir, f'metrics_round_{round_num}.csv')
    metrics_no_filter = compute_metrics(true_smiles, pred_smiles_lists, output_csv_no_filter, 
                                        doMCES=False, doFull=doFull, 
                                        filter_formula=False)
    print(f"Saved metrics to: {output_csv_no_filter}")
    
    metrics_formula = None
    if enable_formula_filter:
        # Compute metrics with formula filtering
        print("\n--- With formula filtering ---")
        output_csv_formula = os.path.join(output_dir, f'metrics_round_{round_num}_formula_filtered.csv')
        metrics_formula = compute_metrics(true_smiles, pred_smiles_lists, output_csv_formula, 
                                         doMCES=False, doFull=doFull, 
                                         filter_formula=True)
        print(f"Saved metrics to: {output_csv_formula}")
    
    return metrics_no_filter, metrics_formula


def main():
    # Hardcoded path - change this to your base directory
    BASE_PATH = ".../path/to/run_scaling_multi_gpu/output/"

    # Flag to enable/disable formula filtered computations
    ENABLE_FORMULA_FILTER = False 
    
    # Create output directory for metrics
    output_dir = os.path.join(BASE_PATH, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing multi-process outputs from: {BASE_PATH}")
    print(f"Output directory: {output_dir}\n")
    print(f"Formula filtering enabled: {ENABLE_FORMULA_FILTER}")
    
    # Find all round files across subdirectories
    round_files = find_round_files(BASE_PATH)
    
    if not round_files:
        print("No predictions_round_*.csv files found!")
        return
    
    print(f"\nFound predictions for rounds: {sorted(round_files.keys())}\n")
    
    # Process each round
    all_metrics_no_filter = {}
    all_metrics_formula = {}
    for round_num in sorted(round_files.keys()):
        if round_num != 12: continue
        file_paths = round_files[round_num]
        metrics_no_filter, metrics_formula = process_round(round_num, file_paths, output_dir, 
                                                         doFull=False, 
                                                         enable_formula_filter=ENABLE_FORMULA_FILTER)
        if metrics_no_filter:
            all_metrics_no_filter[round_num] = metrics_no_filter
        if metrics_formula:
            all_metrics_formula[round_num] = metrics_formula
    
    # Save summary of all rounds (no filtering)
    if all_metrics_no_filter:
        print(f"\n{'='*70}")
        print("SUMMARY OF ALL ROUNDS (No Formula Filtering)")
        print(f"{'='*70}\n")
        
        summary_rows = []
        for round_num in sorted(all_metrics_no_filter.keys()):
            metrics = all_metrics_no_filter[round_num]
            row = {'round': round_num}
            row.update(metrics)
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'summary_all_rounds.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Saved summary to: {summary_path}\n")
        
        # Print key metrics for each round
        print("Key metrics by round:")
        print(summary_df[['round', 'acc@1', 'tanimoto@1', 'meaningful_match@1', 'close_match@1']])
        print()
        print(summary_df[['round', 'acc@10', 'tanimoto@10', 'meaningful_match@10', 'close_match@10']])
    
    # Save summary of all rounds (formula filtering)
    if all_metrics_formula:
        print(f"\n{'='*70}")
        print("SUMMARY OF ALL ROUNDS (Formula Filtering)")
        print(f"{'='*70}\n")
        
        summary_rows = []
        for round_num in sorted(all_metrics_formula.keys()):
            metrics = all_metrics_formula[round_num]
            row = {'round': round_num}
            row.update(metrics)
            summary_rows.append(row)
        
        summary_df_formula = pd.DataFrame(summary_rows)
        summary_path_formula = os.path.join(output_dir, 'summary_all_rounds_formula_filtered.csv')
        summary_df_formula.to_csv(summary_path_formula, index=False)
        
        print(f"Saved summary to: {summary_path_formula}\n")
        
        # Print key metrics for each round
        print("Key metrics by round (formula filtered):")
        print(summary_df_formula[['round', 'acc@1', 'tanimoto@1', 'meaningful_match@1', 'close_match@1']])
        print()
        print(summary_df_formula[['round', 'acc@10', 'tanimoto@10', 'meaningful_match@10', 'close_match@10']])


if __name__ == "__main__":
    main()