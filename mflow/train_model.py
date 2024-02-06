import argparse
import functools
import json
import os
import time
from distutils.util import strtobool

import torch
import torch.nn as nn

import wandb
from data import transform_cancer
from data.data_loader import NumpyTupleDataset
from mflow.generate import generate_mols
from mflow.models.hyperparams import Hyperparameters
from mflow.models.model import MoFlow, rescale_adj
from mflow.models.utils import check_validity, save_mol_png
from mflow.utils.timereport import TimeReport

print = functools.partial(print, flush=True)


def get_parser():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument("-i", "--data_dir", type=str, default="./data", help="Location for the dataset")
    parser.add_argument(
        "--data_name", type=str, default="melanoma_skmel28", choices=["melanoma_skmel28"], help="dataset name"
    )
    parser.add_argument(
        "-o",
        "--save_dir",
        type=str,
        default="results/melanoma_skmel28",
        help="Location for parameter checkpoints and samples",
    )
    parser.add_argument(
        "-t", "--save_interval", type=int, default=20, help="Every how many epochs to write checkpoint/samples?"
    )
    parser.add_argument(
        "-r",
        "--load_params",
        type=int,
        default=0,
        help="Restore training from previous model checkpoint? 1 = Yes, 0 = No",
    )
    parser.add_argument("--load_snapshot", type=str, default="", help="load the model from this path")
    # optimization
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Base learning rate")
    parser.add_argument(
        "-e",
        "--lr_decay",
        type=float,
        default=0.999995,
        help="Learning rate decay, applied every step of the optimization",
    )
    parser.add_argument("-x", "--max_epochs", type=int, default=5000, help="How many epochs to run in total?")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id to use")
    parser.add_argument(
        "--save_epochs", type=int, default=1, help="in how many epochs, a snapshot of the model needs to be saved?"
    )
    # data loader
    parser.add_argument("-b", "--batch_size", type=int, default=12, help="Batch size during training per GPU")
    parser.add_argument("--shuffle", type=strtobool, default="false", help="Shuffle the data batch")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in the data loader")
    # For bonds
    parser.add_argument(
        "--b_n_flow", type=int, default=10, help="Number of masked glow coupling layers per block for bond tensor"
    )
    parser.add_argument("--b_n_block", type=int, default=1, help="Number of glow blocks for bond tensor")
    parser.add_argument(
        "--b_hidden_ch", type=str, default="128,128", help="Hidden channel list for bonds tensor, delimited list input "
    )
    parser.add_argument(
        "--b_conv_lu",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0: InvConv2d for 1*1 conv, 1:InvConv2dLU for 1*1 conv, 2: No 1*1 conv, swap updating in the coupling layer",
    )
    # For atoms
    parser.add_argument(
        "--a_n_flow", type=int, default=27, help="Number of masked flow coupling layers per block for atom matrix"
    )
    parser.add_argument("--a_n_block", type=int, default=1, help="Number of flow blocks for atom matrix")
    parser.add_argument(
        "--a_hidden_gnn",
        type=str,
        default="64,",
        help="Hidden dimension list for graph convolution for atoms matrix, delimited list input ",
    )
    parser.add_argument(
        "--a_hidden_lin",
        type=str,
        default="128,64",
        help="Hidden dimension list for linear transformation for atoms, delimited list input ",
    )
    parser.add_argument(
        "--mask_row_size_list", type=str, default="1,", help="Mask row size list for atom matrix, delimited list input "
    )
    parser.add_argument(
        "--mask_row_stride_list",
        type=str,
        default="1,",
        help="Mask row stride list for atom matrix, delimited list input",
    )
    # General
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed to use")
    parser.add_argument("--debug", type=strtobool, default="true", help="To run training with more information")
    parser.add_argument("--learn_dist", type=strtobool, default="true", help="learn the distribution of feature matrix")
    parser.add_argument("--noise_scale", type=float, default=0.6, help="x + torch.rand(x.shape) * noise_scale")

    return parser


def train():
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    parser = get_parser()
    args = parser.parse_args()
    # start wandb
    wandb.init(project="tumflow_melanoma_skmel28")

    # Device configuration
    device = -1
    multigpu = False
    if args.gpu >= 0:
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    elif args.gpu == -1:
        device = torch.device("cpu")
    else:
        # multigpu, can be slower than using just 1 gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multigpu = True

    debug = args.debug
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))  # pretty print args

    # Model configuration
    b_hidden_ch = [int(d) for d in args.b_hidden_ch.strip(",").split(",")]
    a_hidden_gnn = [int(d) for d in args.a_hidden_gnn.strip(",").split(",")]
    a_hidden_lin = [int(d) for d in args.a_hidden_lin.strip(",").split(",")]
    mask_row_size_list = [int(d) for d in args.mask_row_size_list.strip(",").split(",")]
    mask_row_stride_list = [int(d) for d in args.mask_row_stride_list.strip(",").split(",")]
    if args.data_name == "melanoma_skmel28":
        data_file = "melanoma_skmel28_relgcn_kekulized_ggnp.npz"
        transform_fn = transform_cancer.transform_fn_cancer
        atomic_num_list = transform_cancer.cancer_atomic_num_list
        b_n_type = 4
        b_n_squeeze = 16
        a_n_node = 80
        a_n_type = len(atomic_num_list)
        valid_idx = transform_cancer.get_val_ids()
    else:
        raise ValueError(
            "Only support melanoma_skmel28 right now. Parameters need change a little bit for other dataset."
        )

    model_params = Hyperparameters(
        b_n_type=b_n_type,
        b_n_flow=args.b_n_flow,
        b_n_block=args.b_n_block,
        b_n_squeeze=b_n_squeeze,
        b_hidden_ch=b_hidden_ch,
        b_affine=True,
        b_conv_lu=args.b_conv_lu,
        a_n_node=a_n_node,
        a_n_type=a_n_type,
        a_hidden_gnn=a_hidden_gnn,
        a_hidden_lin=a_hidden_lin,
        a_n_flow=args.a_n_flow,
        a_n_block=args.a_n_block,
        mask_row_size_list=mask_row_size_list,
        mask_row_stride_list=mask_row_stride_list,
        a_affine=True,
        learn_dist=args.learn_dist,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    print("Model params:")
    model_params.print()
    model = MoFlow(model_params)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_hyperparams(os.path.join(args.save_dir, "tumflow-params.json"))
    if torch.cuda.device_count() > 1 and multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        multigpu = False
    model = model.to(device)

    # Datasets:
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn)
    if len(valid_idx) > 0:
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
        train = torch.utils.data.Subset(dataset, train_idx)
        test = torch.utils.data.Subset(dataset, valid_idx)
    else:
        torch.manual_seed(args.seed)
        train, test = torch.utils.data.random_split(
            dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
        )

    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers
    )
    valid_dataloader = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers
    )

    print("==========================================")
    print("Load data done! Time {:.2f} seconds".format(time.time() - start))
    print("Data shuffle: {}, Number of data loader workers: {}!".format(args.shuffle, args.num_workers))
    if args.gpu >= 0:
        print("Using GPU device:{}!".format(args.gpu))
    print("Num Train-size: {}".format(len(train)))
    print("Num Minibatch-size: {}".format(args.batch_size))
    print("Num Iter/Epoch: {}".format(len(train_dataloader)))
    print("Num epoch: {}".format(args.max_epochs))
    print("==========================================")

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the models
    iter_train_per_epoch = len(train_dataloader)
    iter_valid_per_epoch = len(valid_dataloader)
    log_step = args.save_interval  # 20 default
    tr = TimeReport(total_iter=args.max_epochs * iter_train_per_epoch)
    for epoch in range(args.max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))

        # starting training
        cumulative_values_train = [0, 0, 0, 0]
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # turn off shuffle to see the order with original code
            x = batch[0].to(device)  # (256,9,5)
            adj = batch[1].to(device)  # (256,4,9, 9)
            adj_normalized = rescale_adj(adj).to(device)

            # Forward, backward and optimize
            z, sum_log_det_jacs = model(adj, x, adj_normalized)
            if multigpu:
                nll = model.module.log_prob(z, sum_log_det_jacs)
            else:
                nll = model.log_prob(z, sum_log_det_jacs)
            loss = nll[0] + nll[1]
            loss.backward()
            optimizer.step()
            tr.update()

            # update cumulative vars
            cumulative_values_train[0] += loss.item() * batch[0].shape[0]
            cumulative_values_train[1] += nll[0].item() * batch[0].shape[0]
            cumulative_values_train[2] += nll[1].item() * batch[0].shape[0]
            cumulative_values_train[3] += int(batch[0].shape[0])

            # Print log info
            if (i + 1) % log_step == 0:  # i % args.log_step == 0:
                print(
                    "Epoch [{}/{}], Train Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f},"
                    " nll_adj: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: ".format(
                        epoch + 1,
                        args.max_epochs,
                        i + 1,
                        iter_train_per_epoch,
                        loss.item(),
                        nll[0].item(),
                        nll[1].item(),
                        tr.get_avg_time_per_iter(),
                        tr.get_avg_iter_per_sec(),
                    )
                )
                tr.print_summary()

        # The same report for each epoch
        print(
            "Epoch [{}/{}], Train Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f},"
            " nll_adj: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: ".format(
                epoch + 1,
                args.max_epochs,
                -1,
                iter_train_per_epoch,
                loss.item(),
                nll[0].item(),
                nll[1].item(),
                tr.get_avg_time_per_iter(),
                tr.get_avg_iter_per_sec(),
            )
        )
        tr.print_summary()

        # Save the model checkpoints
        save_epochs = args.save_epochs
        if save_epochs == -1:
            save_epochs = args.max_epochs
        if (epoch + 1) % save_epochs == 0:
            if multigpu:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.save_dir, "model_snapshot_epoch_{}.pt".format(epoch + 1)),
                )
            else:
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "model_snapshot_epoch_{}.pt".format(epoch + 1))
                )
            tr.end()

        # starting validation
        cumulative_values_valid = [0, 0, 0, 0]
        model.eval()
        for i, batch in enumerate(valid_dataloader):
            optimizer.zero_grad()
            # turn off shuffle to see the order with original code
            x = batch[0].to(device)  # (256,9,5)
            adj = batch[1].to(device)  # (256,4,9, 9)
            adj_normalized = rescale_adj(adj).to(device)

            # Forward, backward and optimize
            z, sum_log_det_jacs = model(adj, x, adj_normalized)
            if multigpu:
                nll = model.module.log_prob(z, sum_log_det_jacs)
            else:
                nll = model.log_prob(z, sum_log_det_jacs)
            loss = nll[0] + nll[1]

            # update cumulative vars
            cumulative_values_valid[0] += loss.item() * batch[0].shape[0]
            cumulative_values_valid[1] += nll[0].item() * batch[0].shape[0]
            cumulative_values_valid[2] += nll[1].item() * batch[0].shape[0]
            cumulative_values_valid[3] += int(batch[0].shape[0])

            # Print log info
            if (i + 1) % log_step == 0:  # i % args.log_step == 0:
                print(
                    "Epoch [{}/{}], Valid Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f}, nll_adj: {:.5f}".format(
                        epoch + 1,
                        args.max_epochs,
                        i + 1,
                        iter_valid_per_epoch,
                        loss.item(),
                        nll[0].item(),
                        nll[1].item(),
                    )
                )

        # The same report for each epoch
        print(
            "Epoch [{}/{}], Valid Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f}, nll_adj: {:.5f}".format(
                epoch + 1, args.max_epochs, -1, iter_valid_per_epoch, loss.item(), nll[0].item(), nll[1].item()
            )
        )

        # start generation
        if debug:

            def print_validity(ith):
                model.eval()
                if multigpu:
                    adj, x = generate_mols(model.module, batch_size=100, device=device)
                else:
                    adj, x = generate_mols(model, batch_size=100, device=device)
                valid_mols = check_validity(adj, x, atomic_num_list)["valid_mols"]
                mol_dir = os.path.join(args.save_dir, "generated_{}".format(ith))

                for ind, mol in enumerate(valid_mols):
                    save_mol_png(mol, os.path.join(mol_dir, "{}.png".format(ind)))
                model.train()

            print_validity(epoch + 1)

        wandb.log(
            {
                "Train loglik": cumulative_values_train[0] / cumulative_values_train[3],
                "Train nll_x": cumulative_values_train[1] / cumulative_values_train[3],
                "Train nll_adj": cumulative_values_train[2] / cumulative_values_train[3],
                "Valid loglik": cumulative_values_valid[0] / cumulative_values_valid[3],
                "Valid nll_x": cumulative_values_valid[1] / cumulative_values_valid[3],
                "Valid nll_adj": cumulative_values_valid[2] / cumulative_values_valid[3],
            }
        )

    print("[Training Ends], Start at {}, End at {}".format(time.ctime(start), time.ctime()))
    wandb.finish()


if __name__ == "__main__":
    train()
