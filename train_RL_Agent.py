import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
import argparse
import os
import sys
from sklearn.decomposition import PCA
from pyntcloud import PyntCloud

import warnings

warnings.filterwarnings("ignore")


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from MultyEncoderFoldingDecoder.metrics.loss import cd_loss_L1
from MultyEncoderFoldingDecoder.replay_buffer_refer import ReplayBuffer
from MultyEncoderFoldingDecoder.td3_pcn import TD3
from MultyEncoderFoldingDecoder.ddpg_pcn_2dgrid import *
from utils.new_shapenet import ShapeNet
from utils.visual.visualization import plot_pcd_one_view
import utils.EMD.emd_module as emd


ROOT_DIR = "MultyEncoderFoldingDecoder/"
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

TEST_DIR = ROOT_DIR + now + "/test_out"
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

TEST_DIR_GAN_OUT = TEST_DIR + "/gan_out"
if not os.path.exists(TEST_DIR_GAN_OUT):
    os.makedirs(TEST_DIR_GAN_OUT)

TEST_DIR_PLY_OUT = TEST_DIR + "/ply_out"
if not os.path.exists(TEST_DIR_PLY_OUT):
    os.makedirs(TEST_DIR_PLY_OUT)


def compare(partial, pc1, pc2):
    cd1 = cd_loss_L1(partial, pc1)
    cd2 = cd_loss_L1(partial, pc2)
    return pc1 if cd1 <= cd2 else pc2


def save_pc(pc, filename, path):
    """
    pc: (N, 3) numpy array
    """
    points = pd.DataFrame(pc, columns=["x", "y", "z"])
    cloud = PyntCloud(points)
    cloud.to_file(os.path.join(path, filename))


class Arguments:
    def __init__(self, stage="pretrain"):
        self._parser = argparse.ArgumentParser(
            description="Arguments for |RL|pretain|inversion|eval_treegan|eval_completion."
        )

        if stage == "eval_completion":
            self.add_eval_completion_args()
        else:
            self.add_common_args()
            if stage == "pretrain":
                self.add_pretrain_args()
            elif stage == "inversion":
                self.add_inversion_args()
            elif stage == "eval_treegan":
                self.add_eval_treegan_args()

    def add_common_args(self):
        ### data related
        self._parser.add_argument(
            "--class_choice",
            type=str,
            default="chair",
            help="plane|cabinet|car|chair|lamp|couch|table|watercraft",
        )
        self._parser.add_argument(
            "--dataset",
            type=str,
            default="PCN",
            help="CRN|MatterPort|ScanNet|KITTI|PartNet|PFNet",
        )
        self._parser.add_argument("--dataset_path", type=str, help="Dataset path")
        self._parser.add_argument(
            "--split",
            type=str,
            default="test",
            help="NOTE: train if pretrain and generate_fpd_stats; test otherwise",
        )

        ### TreeGAN architecture related
        self._parser.add_argument(
            "--DEGREE",
            type=int,
            default=[1, 2, 2, 2, 2, 2, 64],
            nargs="+",
            help="Upsample degrees for generator.",
        )
        self._parser.add_argument(
            "--G_FEAT",
            type=int,
            default=[96, 256, 256, 256, 128, 128, 128, 3],
            nargs="+",
            help="Features for generator.",
        )
        self._parser.add_argument(
            "--D_FEAT",
            type=int,
            default=[3, 64, 128, 256, 256, 512],
            nargs="+",
            help="Features for discriminator.",
        )
        self._parser.add_argument(
            "--support",
            type=int,
            default=10,
            help="Support value for TreeGCN loop term.",
        )
        self._parser.add_argument(
            "--loop_non_linear",
            default=False,
            type=lambda x: (str(x).lower() == "true"),
        )

        ### others
        self._parser.add_argument(
            "--FPD_path",
            type=str,
            default="./evaluation/pre_statistics_chair.npz",
            help="Statistics file path to evaluate FPD metric. (default:all_class)",
        )
        self._parser.add_argument(
            "--gpu", type=int, default=0, help="GPU number to use."
        )
        self._parser.add_argument(
            "--ckpt_load",
            type=str,
            default="shape_inversion/pretrained_models/chair.pt",
            help="Checkpoint name to load. (default:None)",
        )

        ### RL args
        self._parser.add_argument(
            "--policy", default="DDPG"
        )  # Policy name (TD3, DDPG or OurDDPG)
        self._parser.add_argument(
            "--env", default="My-Design"
        )  # OpenAI gym environment name
        self._parser.add_argument(
            "--seed", default=5, type=int
        )  # Sets Gym, PyTorch and Numpy seeds
        self._parser.add_argument("--category", default="all")
        self._parser.add_argument(
            "--start_timesteps", default=3e3, type=int
        )  # Time steps initial random policy is used
        self._parser.add_argument(
            "--eval_freq", default=10, type=int
        )  # How often (time steps) we evaluate
        self._parser.add_argument(
            "--test_freq", default=10, type=int
        )  # How often (time steps) we test
        self._parser.add_argument(
            "--max_timesteps", default=5e5, type=int
        )  # Max time steps to run environment
        self._parser.add_argument(
            "--expl_noise", default=0.1
        )  # Std of Gaussian exploration noise
        self._parser.add_argument("--batch_size", default=1, type=int)
        self._parser.add_argument(
            "--sample_batch_size", default=256, type=int
        )  # Batch size for both actor and critic
        self._parser.add_argument("--num_workers", default=8, type=int)
        self._parser.add_argument("--discount", default=0.99)  # Discount factor
        self._parser.add_argument("--tau", default=0.005)  # Target network update rate
        self._parser.add_argument(
            "--policy_noise", default=0.2
        )  # Noise added to target policy during critic update
        self._parser.add_argument(
            "--noise_clip", default=0.005
        )  # Range to clip target policy noise
        self._parser.add_argument(
            "--policy_freq", default=2, type=int
        )  # Frequency of delayed policy updates
        self._parser.add_argument(
            "--save_model", action="store_true"
        )  # Save model and optimizer parameters
        self._parser.add_argument(
            "--load_model", default=""
        )  # Model load file name, "" doesn't load, "default" uses file_name
        self._parser.add_argument(
            "--max_episodes_steps", default=5, type=int
        )  # Frequency of delayed policy updates

    def add_inversion_args(self):

        ### loss related
        self._parser.add_argument(
            "--w_nll",
            type=float,
            default=0.001,
            help="Weight for the negative log-likelihood loss (default: %(default)s)",
        )
        self._parser.add_argument(
            "--p2f_chamfer",
            action="store_true",
            default=False,
            help="partial to full chamfer distance",
        )
        self._parser.add_argument(
            "--p2f_feature",
            action="store_true",
            default=False,
            help="partial to full feature distance",
        )
        self._parser.add_argument(
            "--w_D_loss",
            type=float,
            default=[0.1],
            nargs="+",
            help="Discriminator feature loss weight (default: %(default)s)",
        )
        self._parser.add_argument(
            "--directed_hausdorff",
            action="store_true",
            default=False,
            help="directed_hausdorff loss during inversion",
        )
        self._parser.add_argument("--w_directed_hausdorff_loss", type=float, default=1)

        ### mask related
        self._parser.add_argument(
            "--mask_type",
            type=str,
            default="k_mask",
            help="none|knn_hole|ball_hole|voxel_mask|tau_mask|k_mask; for reconstruction, jittering, morphing, use none; the proposed for shape completion is k_mask",
        )
        self._parser.add_argument(
            "--k_mask_k",
            type=int,
            default=[5, 5, 5, 5],
            nargs="+",
            help="the k value for k_mask, i.e., top k to keep",
        )
        self._parser.add_argument(
            "--voxel_bins", type=int, default=32, help="number of bins for voxel mask"
        )
        self._parser.add_argument(
            "--surrounding",
            type=int,
            default=0,
            help="< n surroundings, for the surrounding voxels to be masked as 0, for mask v2",
        )
        self._parser.add_argument(
            "--tau_mask_dist",
            type=float,
            default=[0.01, 0.01, 0.01, 0.01],
            nargs="+",
            help="tau for tau_mask",
        )
        self._parser.add_argument(
            "--hole_radius",
            type=float,
            default=0.35,
            help="radius of the single hole, ball hole",
        )
        self._parser.add_argument(
            "--hole_k", type=int, default=500, help="k of knn ball hole"
        )
        self._parser.add_argument(
            "--hole_n", type=int, default=1, help="n holes for knn hole or ball hole"
        )
        self._parser.add_argument(
            "--masking_option",
            type=str,
            default="element_product",
            help="keep zeros with element_prodcut or remove zero with indexing",
        )

        ### inversion mode related
        self._parser.add_argument(
            "--inversion_mode",
            type=str,
            default="completion",
            help="reconstruction|completion|jittering|morphing|diversity|ball_hole_diversity|simulate_pfnet",
        )

        ### other GAN inversion related
        self._parser.add_argument(
            "--random_G",
            action="store_true",
            default=False,
            help="Use randomly initialized generator? (default: %(default)s)",
        )
        self._parser.add_argument(
            "--select_num",
            type=int,
            default=500,
            help="Number of point clouds pool to select from (default: %(default)s)",
        )
        self._parser.add_argument(
            "--sample_std",
            type=float,
            default=1.0,
            help="Std of the gaussian distribution used for sampling (default: %(default)s)",
        )
        self._parser.add_argument(
            "--iterations",
            type=int,
            default=[200, 200, 200, 200],
            nargs="+",
            help="For bulk structures, i.e., car, couch, cabinet, and plane, each sub-stage consists of 30 iterations; \
            for thin structures, i.e., chair, lamp, table, and boat, each sub-stage consists of 200 iterations.",
        )
        self._parser.add_argument(
            "--G_lrs",
            type=float,
            default=[2e-7, 1e-6, 1e-6, 2e-7],
            nargs="+",
            help="Learning rate steps of Generator",
        )
        self._parser.add_argument(
            "--z_lrs",
            type=float,
            default=[1e-2, 1e-4, 1e-4, 1e-6],
            nargs="+",
            help="Learning rate steps of latent code z",
        )
        self._parser.add_argument(
            "--warm_up",
            type=int,
            default=0,
            help="Number of warmup iterations (default: %(default)s)",
        )
        self._parser.add_argument(
            "--update_G_stages",
            type=str2bool,
            default=[1, 1, 1, 1],
            nargs="+",
            help="update_G, control at stage",
        )
        self._parser.add_argument(
            "--progressive_finetune",
            action="store_true",
            default=False,
            help="progressive finetune at each stage",
        )
        self._parser.add_argument(
            "--init_by_p2f_chamfer",
            action="store_true",
            default=False,
            help="init_by_p2f_chamfer instead of D feature distance",
        )
        self._parser.add_argument(
            "--early_stopping",
            action="store_true",
            default=False,
            help="early stopping",
        )
        self._parser.add_argument(
            "--stop_cd",
            type=float,
            default=0.0005,
            help="CD threshold for stopping training (default: %(default)s)",
        )
        self._parser.add_argument(
            "--target_downsample_method",
            default="",
            type=str,
            help="FPS: can optionally downsample via Farthest Point Sampling",
        )
        self._parser.add_argument(
            "--target_downsample_size",
            default=1024,
            type=int,
            help="downsample target to what number by FPS",
        )

        ### others
        self._parser.add_argument(
            "--save_inversion_path",
            default=f"{TEST_DIR_GAN_OUT}",
            help="directory to save generated point clouds",
        )
        self._parser.add_argument(
            "--dist",
            action="store_true",
            default=False,
            help="Train with distributed implementation (default: %(default)s)",
        )
        self._parser.add_argument(
            "--port",
            type=str,
            default="12345",
            help="Port id for distributed training (default: %(default)s)",
        )
        self._parser.add_argument(
            "--visualize", action="store_true", default=False, help=""
        )

    def add_eval_completion_args(self):
        self._parser.add_argument(
            "--eval_with_GT",
            type=str2bool,
            default=0,
            help="if eval on real scans, choose false",
        )
        self._parser.add_argument(
            "--saved_results_path",
            type=str,
            required=True,
            help="path of saved_results for evaluation",
        )

    def parser(self):
        return self._parser


np.random.seed(5)

emd_dis = emd.emdModule().to(device)

# utils
class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, x):
        #   neglog = - F.log_softmax(x,dim=0)
        # greater the value greater the chance of being real
        # probe = torch.mean(-F.log_softmax(x,dim=0))#F.softmax(x,dim=0)

        #  print(x.cpu().data.numpy())
        # print(-torch.log(x).cpu().data.numpy())
        return torch.mean(x)


class MSE(nn.Module):
    def __init__(self, reduction="mean"):
        super(MSE, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        mse = F.mse_loss(x, y, reduction=self.reduction)
        return mse


class Norm(nn.Module):
    def __init__(self, dims):
        super(Norm, self).__init__()
        self.dims = dims

    def forward(self, x):
        z2 = torch.norm(x, p=2)
        out = z2 - self.dims
        out = out * out
        return out


def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])]
        )
    return pc[idx[:n]]


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(policy, valid_loader, env, eval_episodes=6, render=False):
    avg_reward = 0.0
    env.reset(
        epoch_size=len(valid_loader), figures=8
    )  # reset the visdom and set number of figures

    # for i,(input) in enumerate(valid_loader):
    for i in range(0, eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)
        # data_iter = iter(valid_loader)
        # input = data_iter.next()
        # action_rand = torch.randn(args.batch_size, args.z_dim)

        obs = env.agent_input(input)  # env(input, action_rand)
        done = False
        while not done:
            # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action = torch.tensor(action).cuda().unsqueeze(dim=0)
            _, reward, done, _ = env(input, action, render=render, disp=True)
            avg_reward += reward

        if i + 1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


def test_policy(
    policy,
    test_loader,
    env,
    tree_GAN_out_list,
    tree_GAN_gt_out_list,
    eval_episodes=18,
    render=True,
):
    avg_reward = 0.0
    avg_loss = 0.0
    env.reset(
        epoch_size=len(test_loader), figures=18
    )  # reset the visdom and set number of figures

    test_cd_loss = 0.0

    for i in range(0, eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(test_loader)
            input = next(dataloader_iterator)

        inputtt = []

        if args.inversion_mode == "simulate_pfnet":
            gt = input[1]
            gt = gt.squeeze(0).cuda()
            n_removal = 512
            choice = [
                torch.Tensor([1, 0, 0]),
                torch.Tensor([0, 0, 1]),
                torch.Tensor([1, 0, 1]),
                torch.Tensor([-1, 0, 0]),
                torch.Tensor([-1, 1, 0]),
            ]
            chosen = random.sample(choice, 1)
            dist = gt.add(-chosen[0].cuda())
            dist_val = torch.norm(dist, dim=1)
            top_dist, idx = torch.topk(dist_val, k=2048 - n_removal)
            partial = gt[idx]
            partial = partial.unsqueeze(0)
            gt = gt.unsqueeze(0)

            print(partial.size())
            print(gt.size())

            inputtt.append(partial)
            inputtt.append(gt)

        # data_iter = iter(valid_loader)
        # input = data_iter.next()
        # action_rand = torch.randn(args.batch_size, args.z_dim)
        obs = env.agent_input(inputtt)  # env(input, action_rand)
        done = False

        while not done:
            # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action = torch.tensor(action).cuda().unsqueeze(dim=0)
            _, reward, done, cd_loss = env(
                inputtt,
                action,
                tree_GAN_out_list[i],
                tree_GAN_gt_out_list[i],
                render=render,
                disp=True,
            )
            avg_reward += reward
            avg_loss += cd_loss

        test_cd_loss += cd_loss

        if i + 1 >= eval_episodes:
            break

    avg_reward /= eval_episodes

    avg_loss /= eval_episodes

    print("---------------------------------------")
    print("Test Result:")
    print("---------------------------------------")

    print("---------------------------------------")
    print("AVE_Reward over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    print("---------------------------------------")
    print("AVE_Loss over %d episodes: %f" % (eval_episodes, avg_loss))
    print("---------------------------------------")

    return avg_reward


class envs(nn.Module):
    def __init__(self, ae, epoch_size, tree_GAN):
        super(envs, self).__init__()

        self.nll = NLL()
        self.mse = MSE(reduction="elementwise_mean")
        self.norm = Norm(dims=1024)
        self.i = 0
        self.j = 1
        self.figures = 3

        self.epoch = 0
        self.epoch_size = epoch_size
        self.ae = ae
        self.tree_GAN = tree_GAN
        self.iter = 0

    def reset(self, epoch_size, figures=3):
        self.j = 1
        self.i = 0
        self.figures = figures
        self.epoch_size = epoch_size

    def agent_input(self, input):
        with torch.no_grad():
            inputt = input[0].cuda()
            input_var = Variable(inputt, requires_grad=True)
            gfv = self.ae.our_encode(input_var)
            feature_global, _, _, _ = self.ae.our_decode(gfv)
            gfv_1024 = (
                feature_global.detach().cpu().view(1024, -1).permute([1, 0]).numpy()
            )
            pca = PCA(n_components=128)  # 实例化一个特征矩阵，降到128维
            pca = pca.fit(gfv_1024)
            # # 获取新的矩阵
            x_da = pca.transform(gfv_1024)
            gfv_128 = torch.from_numpy(x_da)
            gfv_128 = gfv_128.squeeze(0)
            gfv_128 = gfv_128[0].view(1, -1)

            a = (
                torch.linspace(-0.05, 0.05, steps=2, dtype=torch.float)
                .view(1, 2)
                .expand(2, 2)
                .reshape(1, -1)
            )
            b = (
                torch.linspace(-0.05, 0.05, steps=2, dtype=torch.float)
                .view(2, 1)
                .expand(2, 2)
                .reshape(1, -1)
            )

            temp_seed = torch.cat([a, b], dim=1)

            out = torch.cat([gfv_128, temp_seed], dim=1)
            out = out.detach().cpu().numpy().squeeze()
        return out

    def forward(
        self, input, action, tree_GAN_out=None, render=False, disp=False,
    ):
        with torch.no_grad():
            # Encoder Input
            inputt0 = input[0].cuda()
            input0_var = Variable(inputt0, requires_grad=True)  #

            inputt1 = input[1].cuda()
            input1_var = Variable(inputt1, requires_grad=True)

            action_var = Variable(action, requires_grad=True).cuda()

            # Encoder  output
            gfv = self.ae.our_encode(input0_var)
            (feature_global, point_feat, coarse_out, dense_out,) = self.ae.our_decode(
                gfv
            )

            # ------------------------------------------------------------------------------------
            # get next_state
            gfv_1920 = (
                feature_global.detach().cpu().view(1920, -1).permute([1, 0]).numpy()
            )
            pca = PCA(n_components=128)
            pca = pca.fit(gfv_1920)
            x_da = pca.transform(gfv_1920)
            gfv_128 = torch.from_numpy(x_da)
            gfv_128 = gfv_128.squeeze(0)
            gfv_128 = gfv_128[0].view(1, -1)

            grid_scale = action_var.cpu().item()
            a = (
                torch.linspace(-grid_scale, grid_scale, steps=2, dtype=torch.float)
                .view(1, 2)
                .expand(2, 2)
                .reshape(1, -1)
            )
            b = (
                torch.linspace(-grid_scale, grid_scale, steps=2, dtype=torch.float)
                .view(2, 1)
                .expand(2, 2)
                .reshape(1, -1)
            )
            temp_new_seed = torch.cat([a, b], dim=1)

            temp = torch.cat([gfv_128, temp_new_seed], dim=1)
            state = temp.detach().cpu().data.numpy().squeeze()
            # ------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------
            # concat to a new feat to send to decoder
            new_seed = torch.cat([a, b], dim=0).view(1, 2, 2 ** 2).cuda()
            new_seed = new_seed.unsqueeze(2).expand(
                1, -1, 1024, -1
            )  # (1, 2, num_coarse, S)
            new_seed = new_seed.reshape(1, -1, 4096)  # (1, 2, num_fine)

            new_feat = torch.cat([feature_global, new_seed, point_feat], dim=1)
            # ------------------------------------------------------------------------------------

            # Decoder Output
            AE_Coarse_out = coarse_out

            # RL Decoder Output
            _, _, _, RL_Dense_out = self.ae.our_decode(gfv, new_feat=new_feat)

        # chamfer_loss
        dense_cd_loss = cd_loss_L1(RL_Dense_out, input1_var)

        # emd_loss
        emd_loss, _ = emd_dis(RL_Dense_out, input1_var, eps=0.005, iters=50)
        emd_loss = torch.sqrt(emd_loss).mean(1).mean()

        # Norm Loss
        # loss_norm = self.norm(action)

        # States Formulation
        state_curr = np.array(
            [dense_cd_loss.cpu().data.numpy(), emd_loss.cpu().data.numpy(),]
        )
        #  state_prev = self.state_prev

        reward_D = -state_curr[0]  # -state_curr[2] + self.state_prev[2]
        reward_EMD = -state_curr[1]

        # Reward Formulation
        reward = reward_D * 50.0 + reward_EMD * 100.0

        cd_loss = 0.0

        if render == True and self.j <= self.figures:
            with torch.no_grad():
                # -------------------------------------------------------------------------------------
                # choose which point_feat to use

                compare_out = compare(input0_var, AE_Coarse_out, tree_GAN_out)
                compare_out = compare_out.unsqueeze(0)
                new_point_feat = compare_out.unsqueeze(2).expand(
                    -1, -1, 2 ** 2, -1
                )  # (B, num_coarse, S, 3)
                new_point_feat = new_point_feat.reshape(-1, 4096, 3).transpose(
                    2, 1
                )  # (B, 3, num_fine)
                new_point_feat = new_point_feat.to(device)
                new_feat2 = torch.cat([feature_global, new_seed, new_point_feat], dim=1)
                # -------------------------------------------------------------------------------------

                _, _, _, RL_Dense_out = self.ae.our_decode(gfv, new_feat=new_feat2)

            cd_loss = cd_loss_L1(
                torch.from_numpy(RL_Dense_out).unsqueeze(0).to(device), input1_var,
            )

            plot_pcd_one_view(
                os.path.join(
                    TEST_DIR, "episode_{:03d}_{:03d}.png".format(self.epoch, self.j)
                ),
                [
                    input0_var[0].detach().cpu().numpy(),
                    AE_Coarse_out[0].detach().cpu().numpy(),
                    tree_GAN_out[0].detach().cpu().numpy(),
                    RL_Dense_out[0].detach().cpu().numpy(),
                    input1_var[0].detach().cpu().numpy(),
                ],
                ["Input", "Coarse_AE", "sc_GAN_Out", "RL_Out", "Ground Truth",],
                xlim=(-0.35, 0.35),
                ylim=(-0.35, 0.35),
                zlim=(-0.35, 0.35),
            )

            self.epoch += 1
            self.j += 1
        if disp:
            print(
                "[{4}][{0}/{1}]\t Reward: {2}\n States: {3}".format(
                    self.i, self.epoch_size, reward, state_curr, self.iter
                )
            )
            self.i += 1
            if self.i >= self.epoch_size:
                self.i = 0
                self.iter += 1

        done = True

        return state, reward, done, cd_loss


if __name__ == "__main__":

    args = Arguments(stage="inversion").parser().parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    with open("MultyEncoderFoldingDecoder/train_ddpg_pcn_gan.txt", "a") as f:
        f.write(
            "---------------------------------------\n"
            f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}\n"
            "---------------------------------------\n"
        )

    train_dataset = ShapeNet("datasets/PCN", "train", args.category)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_dataset = ShapeNet("datasets/PCN", "valid", args.category)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
    )

    test_dataset = ShapeNet("datasets/PCN", "test", args.category)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    checkpoint_path = ""
    ae = Our_AutoEncoder(num_dense=4096, latent_dim=1920, grid_size=2).to(device)
    ae.load_state_dict(torch.load(checkpoint_path))
    ae.eval()

    tree_GAN = TreeGAN_Completeion(args, test_dataloader)
    tree_GAN_out_list = []
    tree_GAN_gt_out_list = []
    tree_GAN_out_list, tree_GAN_gt_out_list = tree_GAN.run()

    epoch_size = len(val_dataloader)
    env = envs(ae, epoch_size=epoch_size, tree_GAN=tree_GAN)

    state_dim = 136
    action_dim = 1
    max_action = 0.1

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": device,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer()

    evaluations = [evaluate_policy(policy, val_dataloader, env)]

    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_test = 0
    episode_num = 0
    done = True
    env.reset(epoch_size=len(train_dataloader))

    while total_timesteps < args.max_timesteps:

        if done:
            inputtt = []
            try:
                input = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_dataloader)
                input = next(dataloader_iterator)

            if args.inversion_mode == "simulate_pfnet":
                gt = input[1]
                gt = gt.squeeze(0).cuda()
                n_removal = 512
                choice = [
                    torch.Tensor([1, 0, 0]),
                    torch.Tensor([0, 0, 1]),
                    torch.Tensor([1, 0, 1]),
                    torch.Tensor([-1, 0, 0]),
                    torch.Tensor([-1, 1, 0]),
                ]
                chosen = random.sample(choice, 1)
                dist = gt.add(-chosen[0].cuda())
                dist_val = torch.norm(dist, dim=1)
                top_dist, idx = torch.topk(dist_val, k=2048 - n_removal)
                partial = gt[idx]
                partial = partial.unsqueeze(0)
                gt = gt.unsqueeze(0)

                inputtt.append(partial)
                inputtt.append(gt)

            if total_timesteps != 0:
                # print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                if args.policy == "TD3":
                    policy.train(
                        replay_buffer,
                        episode_timesteps,
                        args.sample_batch_size,
                        args.discount,
                        args.tau,
                        args.policy_noise,
                        args.noise_clip,
                        args.policy_freq,
                    )
                else:
                    policy.train(
                        replay_buffer,
                        episode_timesteps,
                        args.sample_batch_size,
                        args.discount,
                        args.tau,
                    )

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq

                evaluations.append(
                    evaluate_policy(policy, val_dataloader, env, render=False)
                )

                policy.save(file_name, directory=MODEL_DIR)

                if timesteps_since_test >= args.test_freq:
                    timesteps_since_test %= args.test_freq
                    env.reset(epoch_size=len(test_dataloader))
                    test_policy(
                        policy,
                        test_dataloader,
                        env,
                        tree_GAN_out_list,
                        tree_GAN_gt_out_list,
                        render=True,
                    )
                env.reset(epoch_size=len(train_dataloader))

            # Reset environment
            # obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        obs = env.agent_input(inputtt)

        if total_timesteps < args.start_timesteps:
            action_t = torch.FloatTensor(args.batch_size, action_dim).uniform_(
                0, max_action
            )
            action = action_t.detach().cpu().numpy().squeeze(0)
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (
                    action + np.random.normal(0, args.expl_noise, size=action_dim)
                ).clip(0, max_action * np.ones(action_dim,))
                action = np.float32(action)
            action_t = torch.tensor(action).cuda().unsqueeze(dim=0)

        # Perform action
        new_obs, reward, done, _ = env(inputtt, action_t, disp=True)

        # new_obs, reward, done, _ = env.step(action)
        done_bool = (
            0 if episode_timesteps + 1 == args.max_episodes_steps else float(done)
        )
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_test += 1

