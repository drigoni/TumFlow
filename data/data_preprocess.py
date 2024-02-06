import argparse
import os
import time

import pandas as pd
from rdkit import Chem

import mflow.utils.environment as env
from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_name", type=str, default="melanoma_skmel28", choices=["melanoma_skmel28"], help="Dataset to use"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="relgcn",
        choices=["gcn", "relgcn"],
    )
    args = parser.parse_args()
    return args


def cancer_smiles_csv_to_properties(smiles, eff_GI50, eff_LC50, eff_IC50, eff_TGI):
    n = len(smiles)
    f = open("./data/melanoma_skmel28_property.csv", "w")
    f.write("qed,plogp,AVERAGE_GI50,AVERAGE_LC50,AVERAGE_IC50,AVERAGE_TGI,smile\n")
    results = []
    total = 0
    bad_qed = 0
    bad_plogp = 0
    bad_eff = 0
    bad_energy = 0
    invalid = 0
    for i, (smile, score_GI50, score_GLC50, score_IC50, score_TGI) in enumerate(
        zip(smiles, eff_GI50, eff_LC50, eff_IC50, eff_TGI)
    ):
        if i % 10000 == 0:
            print("In {}/{} line".format(i, n))
        total += 1
        mol = Chem.MolFromSmiles(smile)
        if mol == None:
            print("Invalid smile: ", i, smile)
            continue

        try:
            qed = env.qed(mol)
        except ValueError:
            bad_qed += 1
            qed = -1
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), " error in qed")

        try:
            plogp = env.penalized_logp(mol)
        except RuntimeError:
            bad_plogp += 1
            plogp = -999
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), " error in penalized_log")

        results.append((qed, plogp, score_GI50, score_GLC50, score_IC50, score_TGI, smile))
        f.write("{},{},{},{},{},{},{}\n".format(qed, plogp, score_GI50, score_GLC50, score_IC50, score_TGI, smile))
        f.flush()
    f.close()

    print("Dump done!")
    print("Total: {}\t Invalid: {}\t bad_plogp: {} \t bad_qed: {}\n".format(total, invalid, bad_plogp, bad_qed))


def preprocess():
    start_time = time.time()
    args = parse()
    data_name = args.data_name
    data_type = args.data_type
    print("args", vars(args))

    if data_name == "melanoma_skmel28":
        max_atoms = 80
    else:
        raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

    if data_type == "relgcn":
        preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
    else:
        raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

    data_dir = "./data/"
    os.makedirs(data_dir, exist_ok=True)

    if data_name == "melanoma_skmel28":
        print("Preprocessing melanoma_skmel28 data")
        df_cancer = pd.read_csv("./data/melanoma_skmel28.csv", index_col=None)
        labels = ["AVERAGE_GI50", "AVERAGE_LC50", "AVERAGE_IC50", "AVERAGE_TGI"]
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col="SMILES")
        result = parser.parse(df_cancer, return_smiles=True, return_is_successful=True)
        dataset = result["dataset"]
        smiles = result["smiles"]
        is_successful = result["is_successful"]
        eff_GI50 = df_cancer.loc[is_successful, "AVERAGE_GI50"]
        eff_LC50 = df_cancer.loc[is_successful, "AVERAGE_LC50"]
        eff_IC50 = df_cancer.loc[is_successful, "AVERAGE_IC50"]
        eff_TGI = df_cancer.loc[is_successful, "AVERAGE_TGI"]
        # prepare file for property optimization
        cancer_smiles_csv_to_properties(smiles, eff_GI50, eff_LC50, eff_IC50, eff_TGI)
    else:
        raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

    NumpyTupleDataset.save(os.path.join(data_dir, "{}_{}_kekulized_ggnp.npz".format(data_name, data_type)), dataset)
    print("Total time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == "__main__":
    preprocess()
