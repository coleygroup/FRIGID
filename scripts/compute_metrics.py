'''
Compute metrics for the predictions.csv outputs from benchmark_spec2mol.py
'''

import pickle
import os
from collections import Counter, defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

try:
    import pulp
    from myopic_mces import MCES
except ModuleNotFoundError:
    print("'pulp' or 'myopic_mces' not found. MCES metric will not be available.")


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

def compute_metrics_for_one(t_inchi, p_inchi, solver, doMCES=False, doFull=False):
    RDLogger.DisableLog('rdApp.*')

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

def compute_metrics(true, pred, csv_path, doMCES=False, doFull=False):
    true_inchi = []
    pred_inchi = []
    for i in tqdm(range(len(true)), desc="Preprocessing", leave=False):
        local_pred_inchi = []
        local_pred_inchi_set = set()
        for j in range(len(pred[i])):
            if pred[i][j] is not None and is_valid(pred[i][j]):
                pi = Chem.MolToInchi(pred[i][j])
                if pi not in local_pred_inchi_set:
                    local_pred_inchi_set.add(pi)
                    local_pred_inchi.append(pi)

        if not doFull:
            local_pred_inchi = local_pred_inchi[:11]

        pred_inchi.append(local_pred_inchi)
        true_inchi.append(Chem.MolToInchi(true[i]))

    if doMCES:
        solver = pulp.listSolvers(onlyAvailable=True)[0]
    else:
        solver = None

    with tqdm_joblib(tqdm(total=len(true_inchi))) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(compute_metrics_for_one)(
                true_inchi[i],
                pred_inchi[i],
                solver,
                doMCES=doMCES,
                doFull=doFull
            )
            for i in range(len(true_inchi))
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

    if doFull:
        for k in range(1, 101):
            final_metrics[f'acc@{k}'] /= len(true_inchi)
            final_metrics[f'mces@{k}'] /= len(true_inchi)
            final_metrics[f'tanimoto@{k}'] /= len(true_inchi)
            final_metrics[f'rdkit_tanimoto@{k}'] /= len(true_inchi)
            final_metrics[f'cosine@{k}'] /= len(true_inchi)
            final_metrics[f'close_match@{k}'] /= len(true_inchi)
            final_metrics[f'meaningful_match@{k}'] /= len(true_inchi)
            
            # Compute medians
            final_metrics[f'median_mces@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_mces])
            final_metrics[f'median_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_tanimoto])
            final_metrics[f'median_rdkit_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_rdkit_tanimoto])
            final_metrics[f'median_cosine@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_cosine])
    else:
        for k in range(1, 11):
            final_metrics[f'acc@{k}'] /= len(true_inchi)
            final_metrics[f'mces@{k}'] /= len(true_inchi)
            final_metrics[f'tanimoto@{k}'] /= len(true_inchi)
            final_metrics[f'rdkit_tanimoto@{k}'] /= len(true_inchi)
            final_metrics[f'cosine@{k}'] /= len(true_inchi)
            final_metrics[f'close_match@{k}'] /= len(true_inchi)
            final_metrics[f'meaningful_match@{k}'] /= len(true_inchi)
            
            # Compute medians
            final_metrics[f'median_mces@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_mces])
            final_metrics[f'median_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_tanimoto])
            final_metrics[f'median_rdkit_tanimoto@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_rdkit_tanimoto])
            final_metrics[f'median_cosine@{k}'] = np.median([prefix[min(k, len(prefix)-1)] for prefix in per_sample_cosine])

    df = pd.DataFrame(final_metrics, index=[0])

    print(df[["acc@1", "mces@1", "tanimoto@1", "meaningful_match@1", "close_match@1"]])
    print(df[["acc@10", "mces@10", "tanimoto@10", "meaningful_match@10", "close_match@10"]])
    print("\nMedian metrics:")
    print(df[["median_mces@1", "median_tanimoto@1", "median_rdkit_tanimoto@1"]])
    print(df[["median_mces@10", "median_tanimoto@10", "median_rdkit_tanimoto@10"]])

PATH = "/path/to/predictions.csv"

df = pd.read_csv(PATH)
true = df['true_smiles'].apply(lambda x: Chem.MolFromSmiles(x)).tolist()
pred = df.drop(columns=['true_smiles'])
if 'name' in pred.columns:
    pred = pred.drop(columns=['name'])

# Convert predicted SMILES to RDKit Mol objects
for col in pred.columns:
    pred[col] = pred[col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)

pred = pred.values.tolist()

print(PATH)
compute_metrics(true, pred, "metrics.csv", doMCES=False, doFull=False)