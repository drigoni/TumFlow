import os
import sys
# for linux env.
sys.path.insert(0,'..')
import pandas as pd
from tqdm import tqdm
import argparse
import time
import tqdm
import mflow.utils.environment as env
from rdkit import Chem


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='melanoma_skmel28',
                        choices=['melanoma_skmel28'],
                        help='dataset to be use')
    args = parser.parse_args()
    return args


start_time = time.time()
args = parse()
data_name = args.data_name
print('args', vars(args))
data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'melanoma_skmel28':
    print('Preprocessing melanoma_skmel28 data')
    df_cancer = pd.read_csv('melanoma_skmel28.csv', index_col=None)
    cancer_dataset = df_cancer.to_numpy()
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

# transform to molecule structure
smiles = cancer_dataset[:, 0]
eff = cancer_dataset[:, 1]
mols = []
for tmp in smiles:
    mm = Chem.MolFromSmiles(tmp)
    mols.append(mm)
assert len(smiles) == len(eff) == len(mols)

# calculate other properties
qed = []
plogp = []
energy = []
wt = []
charge = []
n_atoms = []
tmp_count = 0
for idx in tqdm.tqdm(range(len(smiles))):
    smi = smiles[idx]
    mol = mols[idx]
    # print("Processing molecule: {} .".format(smi))
    if mol is not None:
        qed.append(env.qed(mol))
        plogp.append(env.penalized_logp(mol))
        tmp_energy = env.calculate_mol_energy(mol)
        energy.append(tmp_energy)
        if tmp_energy is None:
            tmp_count += 1
        wt.append(env.calculate_mol_wt(mol))
        charge.append(env.calculate_mol_charge(mol))
        n_atoms.append(mol.GetNumAtoms())
    else:
        qed.append(None)
        plogp.append(None)
        energy.append(None)
        wt.append(None)
        charge.append(None)
        n_atoms.append(None)

print("Energy non found for {}/{} molecules. ".format(tmp_count, len(energy)))

# make final csv
cancer_properties_dataframe = pd.DataFrame({
    "qed": qed,
    "plogp": plogp,
    "energy": energy,
    "wt": wt,
    "charge": charge,
    "eff": eff,
    "n_atoms": n_atoms,
    "smiles": smiles
})
cancer_properties_dataframe.to_csv("./melanoma_skmel28_property.csv", index=False)

# print final time requested
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )
