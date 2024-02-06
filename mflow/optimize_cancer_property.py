import argparse

# for linux env.
import copy
import functools
import os
import time
from distutils.util import strtobool

import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from data import transform_cancer
from data.data_loader import NumpyTupleDataset
from data.transform_cancer import cancer_atomic_num_list
from mflow.models.hyperparams import Hyperparameters
from mflow.models.model import MoFlow, rescale_adj
from mflow.models.utils import adj_to_smiles, check_cancer_validity
from mflow.utils.model_utils import load_model, smiles_to_adj
from mflow.utils.sascorer import calculateScore
from mflow.utils.timereport import TimeReport

print = functools.partial(print, flush=True)


class MoFlowProp(nn.Module):
    def __init__(self, model: MoFlow, hidden_size):
        super(MoFlowProp, self).__init__()
        self.model = model
        self.latent_size = model.b_size + model.a_size
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh) - 1):
            modules.append(nn.Linear(vh[i], vh[i + 1]))
            if i < len(vh) - 2:
                modules.append(nn.LeakyReLU())
        self.propNN = nn.Sequential(*modules)

    def encode(self, adj, x):
        with torch.no_grad():
            self.model.eval()
            adj_normalized = rescale_adj(adj).to(adj)
            z, sum_log_det_jacs = self.model(adj, x, adj_normalized)  # z = [h, adj_h]
            h = torch.cat([z[0].reshape(z[0].shape[0], -1), z[1].reshape(z[1].shape[0], -1)], dim=1)
        return h, sum_log_det_jacs

    def reverse(self, z):
        with torch.no_grad():
            self.model.eval()
            adj, x = self.model.reverse(z, true_adj=None)
        return adj, x

    def forward(self, adj, x):
        h, sum_log_det_jacs = self.encode(adj, x)
        output = self.propNN(h)
        return output, h, sum_log_det_jacs


def fit_model(
    model,
    atomic_num_list,
    train_dataloader,
    train_prop,
    valid_dataloader,
    valid_prop,
    device,
    property_name="AVERAGE_GI50",
    max_epochs=10,
    learning_rate=1e-3,
    weight_decay=1e-5,
):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    model = model.to(device)
    model.train()
    # Loss and optimizer
    metrics = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    assert len(train_prop) == len(train_dataloader.dataset)
    assert len(valid_prop) == len(valid_dataloader.dataset)

    iter_per_epoch = len(train_dataloader)
    log_step = 20
    tr = TimeReport(total_iter=max_epochs * iter_per_epoch)
    if property_name == "AVERAGE_GI50":
        col = 0
    elif property_name == "AVERAGE_LC50":
        col = 1
    elif property_name == "AVERAGE_IC50":
        col = 2
    elif property_name == "AVERAGE_TGI":
        col = 3
    else:
        raise ValueError("Wrong property_name {}".format(property_name))

    best_validation_score = 10 ^ 5
    best_model = None
    for epoch in range(max_epochs):
        print("Training epoch {}, Time: {}".format(epoch + 1, time.ctime()))
        model.train()
        training_loss = 0
        for i, batch in enumerate(train_dataloader):
            x = batch[0].to(device)
            adj = batch[1].to(device)
            bs = x.shape[0]

            ps = i * bs
            pe = min((i + 1) * bs, len(train_dataloader.dataset))
            true_y = [[tt[col]] for tt in train_prop[ps:pe]]
            true_y = torch.tensor(true_y).float().cuda()
            # model and loss
            optimizer.zero_grad()

            y, z, sum_log_det_jacs = model(adj, x)

            loss = metrics(y, true_y)
            loss.backward()
            optimizer.step()
            tr.update()
            training_loss += loss.item() * bs
            # Print log info
            if (i + 1) % log_step == 0:
                print(
                    "Epoch [{}/{}], Iter [{}/{}], loss: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: ".format(
                        epoch + 1,
                        max_epochs,
                        i + 1,
                        iter_per_epoch,
                        loss.item(),
                        tr.get_avg_time_per_iter(),
                        tr.get_avg_iter_per_sec(),
                    )
                )
                tr.print_summary()
        mean_train_loss = training_loss / len(train_dataloader.dataset)
        print("Train loss: {:.7f} .".format(mean_train_loss))

        print("Valid epoch {}, Time: {}".format(epoch + 1, time.ctime()))
        model.eval()
        valid_loss = 0
        for i, batch in enumerate(valid_dataloader):
            x = batch[0].to(device)
            adj = batch[1].to(device)
            bs = x.shape[0]

            ps = i * bs
            pe = min((i + 1) * bs, len(valid_dataloader.dataset))
            true_y = [[tt[col]] for tt in valid_prop[ps:pe]]
            true_y = torch.tensor(true_y).float().cuda()
            # model and loss
            optimizer.zero_grad()
            y, z, sum_log_det_jacs = model(adj, x)
            loss = metrics(y, true_y)
            valid_loss += loss.item() * bs
            # Print log info
        mean_valid_loss = valid_loss / len(valid_dataloader.dataset)
        print("Valid loss: {:.7f} .".format(mean_valid_loss))
        if mean_valid_loss < best_validation_score:
            best_validation_score = mean_valid_loss
            best_model = (epoch + 1, copy.deepcopy(model))
    tr.print_summary()
    tr.end()
    print(
        "[fit_model Ends], Start at {}, End at {}, Total {}".format(
            time.ctime(start), time.ctime(), time.time() - start
        )
    )
    print("Best model obtained ad epoch {}.".format(best_model[0]))
    return best_model[1]


def load_property_csv(add_clinic=False):
    filename = "./data/melanoma_skmel28_property.csv"
    df = pd.read_csv(filename)  #
    df = df.drop(["qed", "plogp"], axis=1)  # ['AVERAGE_GI50', 'AVERAGE_LC50', 'AVERAGE_IC50', 'AVERAGE_TGI','smiles']
    if add_clinic:
        df_clinic = pd.read_csv("./data/melanoma_skmel28_clinic.csv")
        # filtering
        df_clinic = df_clinic[df_clinic["Outcome"] == "effective"]
        # rename columns
        df_clinic = df_clinic.rename(columns={"Canonical_SMILES": "smile"}, errors="raise")
        # fake values to be the best according to training set. In this way, normalization is not a problem.
        best_values_gi50 = df["AVERAGE_GI50"].min()
        best_values_lc50 = df["AVERAGE_LC50"].min()
        best_values_ic50 = df["AVERAGE_IC50"].min()
        best_values_tgi = df["AVERAGE_TGI"].min()
        df_clinic["AVERAGE_GI50"] = best_values_gi50
        df_clinic["AVERAGE_LC50"] = best_values_lc50
        df_clinic["AVERAGE_IC50"] = best_values_ic50
        df_clinic["AVERAGE_TGI"] = best_values_tgi
        # reordering and selecting just useful columns
        df_clinic = df_clinic[["AVERAGE_GI50", "AVERAGE_LC50", "AVERAGE_IC50", "AVERAGE_TGI", "smile"]]
        # concatenation
        df = pd.concat([df, df_clinic], axis=0)

    tuples = [tuple(x) for x in df.values]
    print("Load {} done, length: {}".format(filename, len(tuples)))
    return tuples


def find_lower_score_smiles(
    model,
    property_model,
    device,
    data_name,
    property_name,
    train_prop,
    topk,
    atomic_num_list,
    debug,
    model_dir,
    model_suffix,
):
    start_time = time.time()
    if property_name == "AVERAGE_GI50":
        col = 0
    elif property_name == "AVERAGE_LC50":
        col = 1
    elif property_name == "AVERAGE_IC50":
        col = 2
    elif property_name == "AVERAGE_TGI":
        col = 3
    print("Finding lower {} score".format(property_name))
    train_prop_sorted = list(
        sorted(train_prop, key=lambda tup: tup[col])
    )  # for melanoma_skmel28 properties lower is better
    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 10 == 0:
            print("Optimization {}/{}, time: {:.2f} seconds".format(i, topk, time.time() - start_time))
        smile = r[-1]  # last column is smile
        try:
            results = optimize_mol(
                model,
                property_model,
                smile,
                device,
                sim_cutoff=0,
                lr=0.005,
                num_iter=100,
                data_name=data_name,
                atomic_num_list=atomic_num_list,
                property_name=property_name,
                random=False,
                debug=debug,
            )
        except:
            print(
                "Exceptions with the molecule: {}. See if it encodes all its atoms. Skipping the molecule. ".format(
                    smile
                )
            )
            continue
        result_list.extend(results)

    result_list.sort(key=lambda tup: tup[1], reverse=False)

    # check novelty
    list_of_training_smiles = set()
    for i, r in enumerate(train_prop_sorted):
        smile = r[-1]
        list_of_training_smiles.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol)
        list_of_training_smiles.add(smile2)

    result_list_novel = []
    for i, r in enumerate(result_list):
        smile_new, predicted_property, smiles_original, similarity_score = r
        if smile_new not in list_of_training_smiles:
            result_list_novel.append(r)

    # dump results
    f = open("{}/{}_{}_discovered_sorted{}.csv".format(model_dir, data_name, property_name, model_suffix), "w")
    f.write(
        "{},{},{},{},{},{}\n".format(
            "New SMILES", "Predicted Property", "SAS", "Weight", "Starting SMILES", "Similarity Score"
        )
    )
    for r in result_list_novel:
        smile_new, predicted_property, smiles_original, similarity_score = r = r
        mol = Chem.MolFromSmiles(smile_new)
        try:
            mol_sas_score = calculateScore(mol)
        except:
            mol_sas_score = -1
        try:
            mol_weight = Chem.Descriptors.MolWt(mol)
        except:
            mol_weight = -1
        f.write(
            "{},{},{},{},{},{}\n".format(
                smile_new, predicted_property, mol_sas_score, mol_weight, smiles_original, similarity_score
            )
        )
        f.flush()
    f.close()
    print("Dump done!")


def optimize_mol(
    model: MoFlow,
    property_model: MoFlowProp,
    mol_start_smiles,
    device,
    sim_cutoff,
    lr=2.0,
    num_iter=20,
    data_name="melanoma_skmel28",
    atomic_num_list=[6, 7, 8, 9, 0],
    property_name="AVERAGE_GI50",
    debug=True,
    random=False,
):
    property_model.eval()
    with torch.no_grad():
        bond, atoms = smiles_to_adj(mol_start_smiles, data_name)
        bond = bond.to(device)
        atoms = atoms.to(device)
        mol_vec, sum_log_det_jacs = property_model.encode(bond, atoms)

        # check main model and property model results
        if debug:
            model.eval()
            adj_rev, x_rev = property_model.reverse(mol_vec)
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            print(mol_start_smiles, reverse_smiles)
            optimize_mol, adj_normalized = rescale_adj(bond).to(device)
            z, sum_log_det_jacs = model(bond, atoms, adj_normalized)
            z0 = z[0].reshape(z[0].shape[0], -1)
            z1 = z[1].reshape(z[1].shape[0], -1)
            adj_rev, x_rev = model.reverse(torch.cat([z0, z1], dim=1))
            reverse_smiles2 = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            train_smiles2 = adj_to_smiles(bond.cpu(), atoms.cpu(), atomic_num_list)
            print(train_smiles2, reverse_smiles2)

    mol_start = Chem.MolFromSmiles(mol_start_smiles)
    mol_start_fingerprint = AllChem.GetMorganFingerprint(mol_start, 2)

    # optimize the molecule
    cur_vec = mol_vec.clone().detach().requires_grad_(True).to(device)
    visited = []
    visited_prop = []
    for step in range(num_iter):
        prop_val = property_model.propNN(cur_vec).squeeze()
        grad = torch.autograd.grad(prop_val, cur_vec)[0]
        if random:
            rad = torch.randn_like(cur_vec.data)
            cur_vec = cur_vec.data - lr * rad / torch.sqrt(rad * rad)
        else:
            cur_vec = cur_vec.data - lr * grad.data / torch.sqrt(grad.data * grad.data)
        cur_vec = cur_vec.clone().detach().requires_grad_(True).to(device)
        visited.append(cur_vec)
        prop_val = property_model.propNN(cur_vec).squeeze()
        visited_prop.append(prop_val)

    hidden_z = torch.cat(visited, dim=0).to(device)
    visited_prop_cat = torch.FloatTensor(visited_prop).to(device)
    adj, x = property_model.reverse(hidden_z)

    # check validity
    val_res = check_cancer_validity(adj, x, visited_prop_cat, atomic_num_list, debug=debug, correct_validity=True)
    valid_mols = val_res["valid_mols"]
    valid_smiles = val_res["valid_smiles"]
    valid_properties = val_res["valid_properties"]

    # removing repetitions
    results = []
    smiles_set = set()
    smiles_set.add(mol_start_smiles)
    for mol_end, mol_end_smiles, mol_end_predicted_prop in zip(valid_mols, valid_smiles, valid_properties):
        if mol_end_smiles in smiles_set:
            continue
        smiles_set.add(mol_end_smiles)
        mol_end_fingerprint = AllChem.GetMorganFingerprint(mol_end, 2)
        sim = DataStructs.TanimotoSimilarity(mol_start_fingerprint, mol_end_fingerprint)
        if sim >= sim_cutoff:
            results.append((mol_end_smiles, mol_end_predicted_prop, mol_start_smiles, sim))
    results.sort(key=lambda tup: tup[1], reverse=False)
    return results


def main():
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./results", required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--data_name", type=str, default="melanoma_skmel28", choices=["melanoma_skmel28"], help="dataset name"
    )
    parser.add_argument("--snapshot_path", "-snapshot", type=str, required=True)
    parser.add_argument("--hyperparams_path", type=str, default="tumflow-params.json", required=True)
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Base learning rate")
    parser.add_argument(
        "-e",
        "--lr_decay",
        type=float,
        default=0.999995,
        help="Learning rate decay, applied every step of the optimization",
    )
    parser.add_argument("-w", "--weight_decay", type=float, default=1e-5, help="L2 norm for the parameters")
    parser.add_argument("--hidden", type=str, default="", help="Hidden dimension list for output regression")
    parser.add_argument("-x", "--max_epochs", type=int, default=5, help="How many epochs to run in total?")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id to use")

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--img_format", type=str, default="svg")
    parser.add_argument(
        "--property_name",
        type=str,
        default="AVERAGE_GI50",
        choices=["AVERAGE_GI50", "AVERAGE_LC50", "AVERAGE_IC50", "AVERAGE_TGI"],
    )
    parser.add_argument(
        "--additive_transformations", type=strtobool, default=False, help="apply only additive coupling layers"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of the gaussian distributions")

    parser.add_argument("--topk", type=int, default=500, help="Top k smiles as seeds")
    parser.add_argument("--debug", type=strtobool, default="true", help="To run optimization with more information")

    parser.add_argument("--sim_cutoff", type=float, default=0.00)
    parser.add_argument("--model_suffix", default="", help="Model name suffix")

    parser.add_argument("--topscore", action="store_true", default=False, help="To find top score")
    parser.add_argument("--eff_energy", action="store_true", default=False, help="To maximize eff and minimize energy")

    args = parser.parse_args()

    # Device configuration
    device = -1
    if args.gpu >= 0:
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))
    property_name = args.property_name
    # chainer.config.train = False
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    model_params = Hyperparameters(path=hyperparams_path)
    torch.manual_seed(model_params.seed)
    model = load_model(snapshot_path, model_params, debug=True)  # Load moflow model

    if args.hidden in ("", ","):
        hidden = []
    else:
        hidden = [int(d) for d in args.hidden.strip(",").split(",")]
    print("Hidden dim for output regression: ", hidden)
    property_model = MoFlowProp(model, hidden)

    atomic_num_list = cancer_atomic_num_list
    transform_fn = transform_cancer.transform_fn_cancer
    valid_idx = transform_cancer.get_val_ids()
    molecule_file = "melanoma_skmel28_relgcn_kekulized_ggnp.npz"

    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, molecule_file), transform=transform_fn)

    print("Load {} done, length: {}".format(os.path.join(args.data_dir, molecule_file), len(dataset)))
    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    n_train = len(train_idx)
    train = torch.utils.data.Subset(dataset, train_idx)
    valid = torch.utils.data.Subset(dataset, valid_idx)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size)

    if args.property_model_path is None:
        print("Training regression model over molecular embedding:")
        prop_list = load_property_csv()
        train_prop = [prop_list[i] for i in train_idx]
        valid_prop = [prop_list[i] for i in valid_idx]
        print("Prepare data done! Time {:.2f} seconds".format(time.time() - start))
        property_model_path = os.path.join(
            args.model_dir, "model_predictor_snapshot_{}{}.pt".format(property_name.lower(), args.model_suffix)
        )
        property_model = fit_model(
            property_model,
            atomic_num_list,
            train_dataloader,
            train_prop,
            valid_dataloader,
            valid_prop,
            device,
            property_name=property_name,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print("saving {} regression model to: {}".format(property_name, property_model_path))
        torch.save(property_model, property_model_path)
        print("Train and save model done! Time {:.2f} seconds".format(time.time() - start))
    else:
        prop_list = load_property_csv(add_clinic=True)
        train_idx = [t for t in range(len(prop_list)) if t not in valid_idx]
        train_prop = [prop_list[i] for i in train_idx]
        valid_prop = [prop_list[i] for i in valid_idx]
        print("Prepare data done! Time {:.2f} seconds".format(time.time() - start))
        if args.eff_energy:
            exit(1)
        else:
            print("Loading trained regression model for optimization")
            property_model_path = os.path.join(args.model_dir, args.property_model_path)
            print("Model path", property_model_path)
            print("loading {} regression model from: {}".format(property_name, property_model_path))
            property_model = MoFlowProp(model, hidden)
            property_model = torch.load(property_model_path, map_location=device)
            # property_model.load_state_dict(torch.load(property_model_path, map_location=device))
            print("Load model done! Time {:.2f} seconds".format(time.time() - start))
            property_model.to(device)
            property_model.eval()
            model.to(device)
            model.eval()

            if args.topscore:
                print("Finding lower score:")
                find_lower_score_smiles(
                    model,
                    property_model,
                    device,
                    args.data_name,
                    property_name,
                    train_prop,
                    args.topk,
                    atomic_num_list,
                    args.debug,
                    args.model_dir,
                    args.model_suffix,
                )

        print("Total Time {:.2f} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()
