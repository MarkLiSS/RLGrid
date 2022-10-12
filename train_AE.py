import argparse
import os
import sys
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.shapenet import ShapeNet
from model.AutoEncoder import AutoEncoder, ChamferLoss
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from tensorboardX import SummaryWriter

# new apply
import utils.EMD.emd_module as emd
import utils.expansion_penalty.expansion_penalty_module as expansion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint_path = "AE/2022-03-20 23:17:42.556566/models/14_ae_new.pt"
total_epochs = 300
batchsize = 4
LR = 0.0001
momentum = 0.95

ROOT_DIR = "AE/"
now = str(datetime.datetime.now())

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

if not os.path.exists(ROOT_DIR + now):
    os.makedirs(ROOT_DIR + now)

LOG_DIR = ROOT_DIR + now + "/logs/"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

MODEL_DIR = ROOT_DIR + now + "/models/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

summary_writer = SummaryWriter(LOG_DIR)

TRAIN_PLY_VIS = "AE/output(ShapeNet)/train_ply_visual"
VAL_PLY_VIS = "AE/output(ShapeNet)/128_val_ply_visual"


def save_pc(pc, filename, path="AE/output(ShapeNet)"):
    """
    pc: (N, 3) numpy array
    """
    points = pd.DataFrame(pc, columns=["x", "y", "z"])
    cloud = PyntCloud(points)
    cloud.to_file(os.path.join(path, filename))


def train(params):

    torch.backends.cudnn.benchmark = True

    train_dataset = ShapeNet("datasets/PCN", "train", params.category)
    val_dataset = ShapeNet("datasets/PCN", "valid", params.category)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )

    # Create the utils and network
    autoencoder = AutoEncoder(4096).to(device)
    # autoencoder.load_state_dict(torch.load(checkpoint_path))
    chamfer_loss = ChamferLoss(4096).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

    # new apply
    emd_dis = emd.emdModule().to(device)
    expansion_dis = expansion.expansionPenaltyModule().to(device)

    if not os.path.exists(TRAIN_PLY_VIS):
        os.makedirs(TRAIN_PLY_VIS)

    if not os.path.exists(VAL_PLY_VIS):
        os.makedirs(VAL_PLY_VIS)

    print("Autoencoder training...")
    for epoch in range(total_epochs):
        total_loss = 0
        autoencoder.train()
        for i, (p, c) in enumerate(train_dataloader):
            p = p.permute([0, 2, 1]).float().to(device)  # [B x 3 x N]
            c = c.permute([0, 2, 1]).float().to(device)
            optimizer.zero_grad()
            out, _ = autoencoder(p)  # [B x 3 x N]
            p = p.permute([0, 2, 1])  # [B x N x 3]
            out = out.permute([0, 2, 1])
            c = c.permute([0, 2, 1])
            cd_loss = chamfer_loss(out, c)

            # # new apply
            # emd_loss, _ = emd_dis(out, c, eps=0.005, iters=50)
            # emd_loss = torch.sqrt(emd_loss).mean(1).mean()
            # expansion_loss, _, _ = expansion_dis(out, 512, 1.5)
            # loss = emd_loss + 0.1 * (torch.mean(expansion_loss)).mean()
            # total_loss += loss
            loss = cd_loss
            loss.backward()
            optimizer.step()
            print(
                "[%d/%d][%d/%d] CD_Loss: %.4f"
                % (epoch, total_epochs, i, len(train_dataloader), loss)
            )
        # with open("AE/AE_total_loss.txt", "a") as f:
        #     f.write(
        #         "[%d/%d] EMD_expansion_Loss: %.4f \n"
        #         % (epoch, total_epochs, total_loss / len(train_dataloader))
        #     )
        # if i % 500 == 0:
        #     save_pc(p[0].detach().cpu().numpy(), filename=f"p_{epoch}_{i}.ply")
        #     save_pc(c[0].detach().cpu().numpy(), filename=f"c_{epoch}_{i}.ply")
        #     save_pc(out[0].detach().cpu().numpy(), filename=f"out_{epoch}_{i}.ply")

        # print("After, ", epoch, "-th epoch")
        # autoencoder.eval()
        # for j, (p, c) in enumerate(val_dataloader):
        #     p = p.permute([0, 2, 1]).float().to(device)
        #     c = c.permute([0, 2, 1]).float().to(device)
        #     out, _ = autoencoder(p)
        #     p = p.permute([0, 2, 1])
        #     out = out.permute([0, 2, 1])
        #     c = c.permute([0, 2, 1])
        #     # loss = chamfer_loss(out, c)

        #     # new apply
        #     emd_loss = emd_dis(out, c, eps=0.005, iters=50)
        #     expansion_loss, _, _ = expansion_dis(out, 512, 1.5)
        #     loss = emd_loss + 0.1 * torch.mean(expansion_loss)
        #     print("val result:", loss)

        #     for x in range(p.shape[0]):
        #         save_pc(
        #             p[x].detach().cpu().numpy(),
        #             filename=f"p{x}_val{j}_epoch{epoch}.ply",
        #             path=VAL_PLY_VIS,
        #         )
        #         save_pc(
        #             c[x].detach().cpu().numpy(),
        #             filename=f"c{x}_val{j}_epoch{epoch}.ply",
        #             path=VAL_PLY_VIS,
        #         )
        #         save_pc(
        #             out[x].detach().cpu().numpy(),
        #             filename=f"out{x}_val{j}_epoch{epoch}.ply",
        #             path=VAL_PLY_VIS,
        #         )

        if epoch % 5 == 0:
            torch.save(autoencoder.state_dict(), MODEL_DIR + "{}_ae.pt".format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PCN")
    parser.add_argument("--exp_name", type=str, help="Tag of experiment")
    parser.add_argument("--log_dir", type=str, default="log", help="Logger directory")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="The path of pretrained model"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--category", type=str, default="all", help="Category of point clouds"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Epochs of training")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for data loader"
    )
    parser.add_argument(
        "--coarse_loss",
        type=str,
        default="cd",
        help="loss function for coarse point cloud",
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="num_workers for data loader"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device for training"
    )
    parser.add_argument(
        "--log_frequency", type=int, default=10, help="Logger frequency in every epoch"
    )
    parser.add_argument(
        "--save_frequency", type=int, default=10, help="Model saving frequency"
    )
    params = parser.parse_args()

    train(params)
