import numpy as np
import torch
from rdkit import Chem

from data.smile_to_graph import GGNNPreprocessor
from data.transform_cancer import transform_fn_cancer
from mflow.models.model import MoFlow as Model


def load_model(snapshot_path, model_params, debug=False):
    print("loading snapshot: {}".format(snapshot_path))
    if debug:
        print("Hyper-parameters:")
        model_params.print()
    model = Model(model_params)

    device = torch.device("cpu")
    model.load_state_dict(torch.load(snapshot_path, map_location=device))
    return model


def smiles_to_adj(mol_smiles, data_name="melanoma_skmel28"):
    if data_name == "melanoma_skmel28":
        out_size = 80
        transform_fn = transform_fn_cancer

    preprocessor = GGNNPreprocessor(out_size=out_size, kekulize=True)
    canonical_smiles, mol = preprocessor.prepare_smiles_and_mol(Chem.MolFromSmiles(mol_smiles))
    atoms, adj = preprocessor.get_input_features(mol)
    atoms, adj, _ = transform_fn((atoms, adj, None))
    adj = np.expand_dims(adj, axis=0)
    atoms = np.expand_dims(atoms, axis=0)

    adj = torch.from_numpy(adj)
    atoms = torch.from_numpy(atoms)
    return adj, atoms


def get_latent_vec(model, mol_smiles, data_name="melanoma_skmel28"):
    adj, atoms = smiles_to_adj(mol_smiles, data_name)
    with torch.no_grad():
        z = model(adj, atoms)
    z = np.hstack([z[0][0].cpu().numpy(), z[0][1].cpu().numpy()]).squeeze(0)  # change later !!! according to debug
    return z
