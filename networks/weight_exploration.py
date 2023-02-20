from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from pandas import DataFrame


def compare_u(exp1, target_param, exp2=None):
    # 90 epochs, if no exp2, comparing epoch to previous
    st = 0 if exp2 is not None else 1
    factory = {"dtype": torch.float, "device": "cuda:0" if torch.cuda.is_available() else "cpu"}

    exp1 = Path(exp1)
    # target_param is like module.fc.weight (name of the parameter)
    # "module.fc.weight/epoch/[u, s, vh, weight]"
    target_folder = exp1 / target_param

    if exp2 is None:
        lastweights = torch.load(target_folder / "0" / "p.pt").to(**factory)
        print(lastweights.shape)
        # lastu = torch.load(target_folder / '0' / "u.pt").to(**factory)
        lastu, _, _ = torch.linalg.svd(lastweights, full_matrices=True)
        # lastu = lastu.T
        firstu = lastu
        # firstweights = lastweights
        # last_diff_mins_u = None

    for epoch in range(st, 90):
        # load the new experiment
        currentweights = torch.load(target_folder / str(epoch) / "p.pt").to(**factory)
        # currentu = torch.load(target_folder / str(epoch) / "u.pt").to(**factory)
        currentu, _, _ = torch.linalg.svd(currentweights, full_matrices=True)
        # currentu = currentu.T

        # need '-reduced.pt' for reduced matrices
        # for reduced, need to check on which axis is orthogonal, for full, the whole matrix is
        #   orthogonal, only changes for TS matrix
        #   u ->
        # cossim = torch.nn.CosineSimilarity()
        # weight_diff = currentweights - lastweights
        # # print(f"{weight_diff.mean():.5f}, {weight_diff.min():.5f}, {weight_diff.max():.5f}, "
        # #       f"{weight_diff.std():.5f},")
        # vecdot = torch.linalg.vecdot(currentu, firstu)

        # print_stats(firsts - currents)
        # print_stats(lasts - currents)

        # want to see if the vector exists somewhere else in the matrix, vecdot doesn't really
        # work for this, need to use `cdist`, but that will give a matrix.
        # cdist expects row vectors (both U and V are row vecs
        # cdist = torch.round(torch.cdist(lastu, currentu), decimals=4)

        # llu = lastu @ lastu.T  # need to have sqrt( squared magnitude of each vector)
        # ccu = torch.sqrt(currentu @ currentu.T)  # need to have sqrt( squared magnitude of each vector)
        # print(lastu[0])
        # cdist = torch.cdist(lastu, currentu, p=1)

        # cos_ang = lastu @ currentu.T
        cos_ang = firstu @ currentu.T

        # print(mm)
        # print(cos_ang[0])
        # cdist = torch.cdist(locq.T, qsend.T)
        arange = torch.arange(cos_ang.shape[0], device=cos_ang.device)
        # arange - cdist argmin shows which elements have changed
        #   if the argmin along dim1 is the same as the index, it is closest to the same vector in
        #   the last epoch and in the same epoch (TODO: clarify this...)
        # print(cdist)
        k = 1
        top_angles = torch.topk(cos_ang, k=k, largest=True)[1]
        # top_angles5 = top_angles[:5]
        in_top3 = torch.tensor(
            [a in b for a, b in zip(top_angles, arange)],
            dtype=torch.bool,
            device=top_angles.device,
        )
        # print("top_angles")
        # print(top_angles5)
        # print(in_top3.sum())

        # diff_mins = torch.nonzero(arange - cos_ang.argmin(dim=1))
        # num_different_min = diff_mins.shape[0]
        print(
            epoch,
            f"\tnumber in the top{k} angles:",
            in_top3.sum().item(),
            "\tnumber of elements:",
            top_angles.shape[0],
            "\t%:",
            f"{((in_top3.sum() / top_angles.shape[0]) * 100).item():.4f}",
        )
        # if last_diff_mins_u is not None:
        #     s1 = set(last_diff_mins_u.flatten().tolist())
        #     s2 = set(diff_mins.flatten().tolist())
        #     # print(s1 - s2)
        # print(diff_mins.flatten())
        # print(arange - cdist.argmin(dim=1))

        # lastu = currentu
        # lastweights = currentweights
        # last_diff_mins_u_u = diff_mins
        # print()


def compare_qr(exp1, target_param, exp2=None):
    # 90 epochs, if no exp2, comparing epoch to previous
    st = 0 if exp2 is not None else 1
    factory = {"dtype": torch.float, "device": "cuda:0" if torch.cuda.is_available() else "cpu"}

    exp1 = Path(exp1)
    # target_param is like module.fc.weight (name of the parameter)
    # "module.fc.weight/epoch/[u, s, vh, weight]"
    target_folder = exp1 / target_param

    if exp2 is None:
        lastweights = torch.load(target_folder / "0" / "p.pt").to(**factory)
        print(lastweights.shape)
        # lastu = torch.load(target_folder / '0' / "u.pt").to(**factory)
        lastq, _ = torch.linalg.qr(lastweights.T, mode="complete")
        lastq = lastq.clone().T
        firstq = lastq.clone()
        firstweights = lastweights.clone()
        # last_diff_mins_u = None

    for epoch in range(st, 90):
        # load the new experiment
        currentweights = torch.load(target_folder / str(epoch) / "p.pt").to(**factory)
        # currentu = torch.load(target_folder / str(epoch) / "u.pt").to(**factory)
        currentq, _ = torch.linalg.qr(currentweights.T, mode="complete")
        currentq = currentq.T

        # cossim = torch.nn.CosineSimilarity()
        weight_diff = currentweights - firstweights
        print(
            epoch,
            weight_diff.abs().sum() / weight_diff.numel(),
            (currentweights - lastweights).abs().sum() / weight_diff.numel(),
        )
        # # print(f"{weight_diff.mean():.5f}, {weight_diff.min():.5f}, {weight_diff.max():.5f}, "
        # #       f"{weight_diff.std():.5f},")
        # vecdot = torch.linalg.vecdot(currentu, firstu)

        # cdist = torch.cdist(lastq, currentq, p=2.0)
        # print((cdist.argmin(dim=1) - torch.arange(cdist.shape[0], device=cdist.device)).sum())

        # # cos_ang = lastq @ currentq.T
        cos_ang = firstq @ currentq.T
        arange = torch.arange(cos_ang.shape[0], device=cos_ang.device)
        k = 1
        top_angles = torch.topk(cos_ang, k=k, largest=True)[1]
        # top_angles5 = top_angles[:5]
        # print(top_angles5)
        in_top3 = torch.tensor(
            [a in b for a, b in zip(top_angles, arange)],
            dtype=torch.bool,
            device=top_angles.device,
        )
        print(
            epoch,
            f"\tnumber in the top{k} angles:",
            in_top3.sum().item(),
            "\tnumber of elements:",
            top_angles.shape[0],
            "\t%:",
            f"{((in_top3.sum() / top_angles.shape[0]) * 100).item():.4f}",
        )
        #
        # lastq = currentq
        lastweights = currentweights


def compare_s(exp1, target_param, exp2=None):
    # 90 epochs, if no exp2, comparing epoch to previous
    st = 0 if exp2 is not None else 1
    factory = {"dtype": torch.float, "device": "cuda:0" if torch.cuda.is_available() else "cpu"}

    exp1 = Path(exp1)
    # target_param is like module.fc.weight (name of the parameter)
    # "module.fc.weight/epoch/[u, s, vh, weight]"
    target_folder = exp1 / target_param

    if exp2 is None:
        lasts = torch.load(target_folder / "0" / "s.pt").to(**factory)
        # lastweights = torch.load(target_folder / "0" / "p.pt").to(**factory)
        # firsts = lasts
        # firstweights = lastweights
        # last_diff_mins_u = None

    for epoch in range(st, 45):
        # load the new experiment
        currents = torch.load(target_folder / str(epoch) / "s.pt").to(**factory)
        # currentweights = torch.load(target_folder / str(epoch) / "p.pt").to(**factory)

        # comparins S is based on how much each value changes, not worried about the vectors at
        #   the moment

        change_from_last = currents - lasts
        # change_from_first = firsts - lasts
        perc_from_last = change_from_last * 100
        # perc_from_first = change_from_first * 100
        print(
            f"{epoch}\t"
            # f"change from first: {perc_from_first.cpu()[:5]}\n\t"
            f"change from last: {perc_from_last.cpu()[:5]}\n",
        )

        lasts = currents
        # lastweights = currentweights
        # last_diff_mins_u_u = diff_mins
        # print()


def print_stats(tens):
    print(
        f"{tens.mean():.5f}, {tens.min():.5f}, {tens.max():.5f}, {tens.std():.5f}",
    )


def plot_qr_diffs(arch, datasets, base_dir, out_dir):

    # % matplotlib inline
    # sns.set(rc={'figure.figsize':(8,6)})

    # notebook to explore the q values
    base_path = Path(base_dir) / arch / datasets
    out_path = Path(out_dir) / arch / datasets
    out_path.mkdir(parents=True, exist_ok=True)
    # 4 files, need heatmaps from them
    #
    # first
    for comp in ["first", "1previous", "previous", "2previous"]:
        # need to do the heatmap for each one -> wq, wtq, wqt, wtqt
        for subexp in ["wq", "wtq", "wqt", "wtqt"]:
            title = (
                f"{datasets}, {arch}, Comparing to {comp} epoch: Weight "
                f"{'Transposed' if subexp[1] == 't' else 'normal'}, Q"
                f"{'.T' if subexp[-1] == 't' else ''} "
            )
            try:
                make_n_save_heatmap_from_df(
                    file_name=base_path / f"{subexp}-{comp}.csv",
                    out_loc=out_path / f"{subexp}-{comp}.png",
                    title=title,
                )
            except FileNotFoundError:
                print(f"no file for {comp}, {subexp}")
            print(f"finished {arch}, {datasets}, {comp}, {subexp}")


def make_n_save_heatmap_from_df(file_name, out_loc, title):
    df = pd.read_csv(file_name)
    try:
        df = df.drop(columns=["Unnamed: 0"])
    except KeyError:
        pass
    plt.figure(figsize=(16, 10))
    sns.heatmap(df, annot=False, vmin=0.0, vmax=1.0).set(title=title)
    plt.tight_layout()
    plt.savefig(out_loc)
    plt.close()


if __name__ == "__main__":
    # base_folder = Path(
    #     "/hkfs/work/workspace/scratch/qv2382-dlrt/saved_models/4gpu-svd-tests/normal/resnet18/"
    # )
    files = Path("/home/daniel/horeka-mount/DLRT/networks/tempdata")
    archs = ["resnet18", "vgg16", "ciresan4"]
    datasets = ["mnist", "cifar10"]

    out_dir = Path("/home/daniel/horeka-mount/DLRT/networks/tempanalyzed")
    for a in archs:
        for d in datasets:
            plot_qr_diffs(arch=a, datasets=d, base_dir=files, out_dir=out_dir)

    # modules = ["module.conv1.weight", "module.fc.weight",
    #              "module.layer1.1.conv2.weight", "module.layer3.1.conv2.weight",
    #              "module.layer4.0.downsample.0.weight", ]

    # compare_u(base_folder, target_param=modules[-5])
    # compare_s(base_folder, target_param=modules[-3])
    # compare_qr(base_folder, target_param=modules[0])
