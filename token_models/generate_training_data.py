import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
import concurrent.futures

# --- Configuration ---
PUBCHEM_PATH = "data/CID-SMILES"
TOTAL_PUBCHEM_LINES = 123467442
CHUNK_SIZE = 32768  # Number of SMILES sent to each CPU core at once
MAX_PENDING_FUTURES = os.cpu_count() * 4 # Prevents memory bloat
MAX_WORKERS = os.cpu_count()

def init_worker():
    """Silence RDKit warnings in every new child process."""
    RDLogger.DisableLog('rdApp.*')

def process_chunk(smiles_chunk, can_forms, can_inchi, msg_forms, msg_inchi):
    """
    Function executed by worker processes to filter SMILES batches.
    """
    can_batch = []
    msg_batch = []
    
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        Chem.RemoveStereochemistry(mol)
        clean_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        # Extract metadata
        form = rdMolDescriptors.CalcMolFormula(mol)

        try:
            inchi_key = Chem.MolToInchiKey(mol).split("-")[0]
        except Exception: # sometimes InChI generation fails, just ignore those molecules
            continue
        
        # Filtering logic
        if form in can_forms and inchi_key not in can_inchi:
            can_batch.append(clean_smi)
        if form in msg_forms and inchi_key not in msg_inchi:
            msg_batch.append(clean_smi)
            
    return can_batch, msg_batch

def main():
    # Disable RDKit in main process
    RDLogger.DisableLog('rdApp.*')

    canopus_forms = set()
    canopus_test_inchi = set()
    msg_forms = set()
    msg_test_inchi = set()

    ########### load canopus train/test set ###########
    print("Loading Canopus data...")
    canopus_labels = pd.read_csv("../data/canopus/labels.tsv", sep="\t")
    canopus_split = pd.read_csv("../data/canopus/splits/canopus_hplus_100_0.tsv", sep="\t")

    canopus_test_names = canopus_split[canopus_split['split'] != 'train'].name
    canopus_test_smiles = canopus_labels[canopus_labels['spec'].isin(canopus_test_names)]['smiles']

    for smi in tqdm(canopus_test_smiles, desc="Processing Canopus SMILES"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        canopus_test_inchi.add(Chem.MolToInchiKey(mol).split("-")[0])
        canopus_forms.add(rdMolDescriptors.CalcMolFormula(mol))

    ########### load msg train/test set ###########
    print("Loading MSG data...")
    msg_labels = pd.read_csv("../data/msg/labels.tsv", sep="\t")
    msg_split = pd.read_csv("../data/msg/split.tsv", sep="\t")
    msg_test_names = msg_split[msg_split['split'] != 'train'].name
    msg_test_smiles = msg_labels[msg_labels['spec'].isin(msg_test_names)]['smiles']

    for smi in tqdm(msg_test_smiles, desc="Processing MSG SMILES"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        msg_test_inchi.add(Chem.MolToInchiKey(mol).split("-")[0])
        msg_forms.add(rdMolDescriptors.CalcMolFormula(mol))

    ########### Parallel Processing of PubChem ###########
    can_train_smi = []
    msg_train_smi = []
    
    print(f"Starting parallel processing with {os.cpu_count()} cores...")
    
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        futures = []
        
        with open(PUBCHEM_PATH, "r") as f:
            current_chunk = []
            pbar = tqdm(total=TOTAL_PUBCHEM_LINES, desc="Parsing PubChem File")
            
            for line in f:
                try:
                    # Basic string splitting is cheap; do it in the main thread
                    parts = line.strip().split("\t")
                    if len(parts) < 2: continue
                    current_chunk.append(parts[1])
                except Exception:
                    continue

                if len(current_chunk) >= CHUNK_SIZE:
                    # Dispatch chunk to a worker
                    futures.append(executor.submit(
                        process_chunk, current_chunk, canopus_forms, 
                        canopus_test_inchi, msg_forms, msg_test_inchi
                    ))
                    current_chunk = []
                    pbar.update(CHUNK_SIZE)
                    
                    # Memory Management: Collect results if the queue gets too long
                    if len(futures) > MAX_PENDING_FUTURES:
                        done, futures = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            can_res, msg_res = fut.result()
                            can_train_smi.extend(can_res)
                            msg_train_smi.extend(msg_res)
                        # Explicitly convert futures back to a list for appending
                        futures = list(futures)

            # Process remaining SMILES in the last chunk
            if current_chunk:
                futures.append(executor.submit(
                    process_chunk, current_chunk, canopus_forms, 
                    canopus_test_inchi, msg_forms, msg_test_inchi
                ))

            # Final collection of remaining futures
            for fut in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(futures), desc="Finalizing Results"):
                can_res, msg_res = fut.result()
                can_train_smi.extend(can_res)
                msg_train_smi.extend(msg_res)
            
            pbar.close()

    # --- Save Output ---
    print(f"Saving results: {len(can_train_smi)} Canopus and {len(msg_train_smi)} MSG SMILES.")
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(can_train_smi, columns=['smiles']).to_csv("data/canopus_train.csv", index=False)
    pd.DataFrame(msg_train_smi, columns=['smiles']).to_csv("data/msg_train.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()