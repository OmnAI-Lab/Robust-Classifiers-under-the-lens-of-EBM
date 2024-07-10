#!/home/briglia/miniconda3/envs/robustness/bin/python3
import torch
import numpy as np

import torchvision.transforms as transforms

from torch.nn import MaxPool2d as MaxPool2d
from torch import nn
import sys, os
import enumeration
import argparse

# get the path of the current file
curr_path = sys.path[0]
print(curr_path)
# %%

os.environ["MPLCONFIGDIR"] = f"/home/briglia/.cache/matplotlib"
os.environ["HF_HOME"] = "/home/briglia/.cache/huggingface/hub"
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = "/home/briglia/.cache"
torch.hub.set_dir("/home/briglia/.cache")

sys.path.append(curr_path + "/robustbenchmaster")
sys.path.append(curr_path + "/utils.py")
sys.path.append(curr_path + "/utils_func.py")

import utils_func
import utils
from robustbenchmaster.robustbench.utils import load_model

# parse arguments
parser = argparse.ArgumentParser(description="Process some parameters.")

parser.add_argument("--wandb_logging", type=bool, default=False, help="log to wandb")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100"],
    help="dataset",
)
parser.add_argument("--step_size", type=float, default=1, help="sampling stepsize")
parser.add_argument(
    "--n_steps", type=int, default=20, help="number of iterations for sampling"
)
parser.add_argument(
    "--n_steps_sampling", type=int, default=5, help="help to  generate 10k images"
)
parser.add_argument("--epsilon", type=int, default=8, help="epsilon")
parser.add_argument("--alpha", type=int, default=2, help="alpha")
parser.add_argument(
    "--sampling_iterations", type=int, default=20, help="sampling iterations"
)
parser.add_argument(
    "--architecture_name", type=str, default="WideResNet", help="architecture name"
)
parser.add_argument(
    "--init",
    type=str,
    default="random",
    choices=["random", "pca", "gaussian", "informative"],
    help="initialization for the generation",
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/briglia/basefolder/LE_PGD/ablation/data",
    help="data directory",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Zhang2019Theoretically",
    help="define the name of the model to be loaded",
)
parser.add_argument(
    "--from_robustbench", type=bool, default=True, help="load model from robustbench"
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="/home/briglia/basefolder/LE_PGD/ablation/models",
)
parser.add_argument(
    "--threat_model",
    type=str,
    default="Linf",
    choices=["Linf", "L2"],
    help="threat model",
)
parser.add_argument(
    "--loss",
    type=str,
    default="energy_xy",
    choices=["energy", "energy_xy", "ce", "reference_energy_xy", "dist_reference"],
    help="loss function",
)
parser.add_argument(
    "--num_samples", type=int, default=10000, help="number of samples to generate"
)
parser.add_argument("--sigma_pca", type=float, default=0.005, help="sigma for pca")
parser.add_argument(
    "--retain_variance", type=float, default=0.95, help="retain variance for PCA"
)
parser.add_argument(
    "--n_components", type=int, default=20, help="number of components for PCA"
)
parser.add_argument(
    "--more_variance", type=bool, default=True, help="more variance for PCA"
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/home/briglia/basefolder/LE_PGD/ablation/logs/eval",
    help="log directory",
)
try:
    args = parser.parse_args()
    print("args are : ", args)
except:
    args = parser.parse_args(
        args=[
            "--wandb_logging",
            "True",
            "--batch_size",
            "200",
            "--dataset",
            "cifar10",
            "--step_size",
            "0.8",
            "--n_steps",
            "50",
            "--n_steps_sampling",
            "5",
            "--epsilon",
            "8/255",
            "--alpha",
            "2/255",
            "--sampling_iterations",
            "20",
            "--architecture_name",
            "WideResNet",
            "--init",
            "pca",
            "--seed",
            "0",
            "--data_dir",
            "/home/briglia/basefolder/LE_PGD/ablation/data",
            "--model_name",
            "Engstrom2019Robustness",
            "--from_robustbench",
            "True",
            "--model_dir",
            "/home/briglia/basefolder/LE_PGD/ablation/models",
            "--loss",
            "energy_xy",
            "--sigma_pca",
            "0.005",
            "--retain_variance",
            "0.95",
            "--log_dir",
            "/home/briglia/basefolder/LE_PGD/ablation/logs",
        ]
    )
    print("command line arguments non found, using default values")
finally:
    import pprint

    pprint.pprint(vars(args))
    wandb = args.wandb_logging
    batch_size = args.batch_size
    dataset = args.dataset
    epsilon = args.epsilon / 255
    alpha = args.alpha / 255
    sampling_iterations = args.sampling_iterations
    architecture_name = args.architecture_name
    init = args.init
    data_dir = args.data_dir
    seed = args.seed
    model_name = args.model_name
    from_robustbench = args.from_robustbench
    model_dir = args.model_dir
    threat_model = args.threat_model
    loss = args.loss
    step_size = args.step_size
    n_steps = args.n_steps
    n_steps_sampling = args.n_steps_sampling
    num_samples = args.num_samples
    retain_variance = args.retain_variance / 100
    n_components = args.n_components
    more_variance = args.more_variance
    sigma_pca = args.sigma_pca
    log_dir = args.log_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################ WANDB LOGGING ####################
if wandb:
    try:

        import wandb, os

        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    except:
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
            print("not able to log on wandb")
            
    config = {
        "batch_size": batch_size,
        "dataset": dataset,
        "epsilon": epsilon,
        "alpha": alpha,
        "sampling_iterations": sampling_iterations,
        "architecture_name": architecture_name,
        "init": init,
        "seed": seed,
        "model_name": model_name,
        "from_robustbench": from_robustbench,
        "model_dir": model_dir,
        "threat_model": threat_model,
        "loss": loss,
        "step_size": step_size,
        "n_steps": n_steps,
        "n_steps_sampling": n_steps_sampling,
        "num_samples": num_samples,
        "retain_variance": retain_variance,
        "n_components": n_components,
        "more_variance": more_variance,
        "sigma_pca": sigma_pca,
    }
    print("logging on wandb")
    run = wandb.init(
        name=f"eval_{architecture_name}_{model_name}_{dataset}_{init}",
        project="LE_PGD",
        config=config,
    )
    
else:
    print("not logging on wandb")

##################### DATA LOADING #####################
label_to_plot = [i for i in range(enumeration.dataset_num_classes[dataset])]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset_train, dataset_test = utils.load_clean_dataset(
    dataset=enumeration.BenchmarkDataset(dataset.lower()),
    n_examples=None,
    data_dir=data_dir,
    transform=transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True
)
# test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# %%
##################### MODEL LOADING #####################

if from_robustbench:
    print("threat model: ", threat_model)
    model = load_model(
        model_name=model_name,
        dataset=dataset,
        threat_model=threat_model,
        model_dir=model_dir,
    )
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

# compute mean and covariance for each class

utils_func.category_mean(
    dload_train=train_dataloader,
    n_classes=enumeration.dataset_num_classes[dataset],
    name_dataset=dataset,
    image_size=enumeration.dataset_image_size[dataset],
    n_channels=3,
    data_dir=data_dir,
)


buffer = utils.center_initialization(
    dataset=dataset, n_classes=enumeration.dataset_num_classes[dataset]
)
replay_buffer = []

# %%
##################### GENERATION #####################
print("generating images")
utils.fix_seeds(seed=seed)
x0, imgs = utils.new_samples(
    model=model,
    label=label_to_plot,
    batch_size=batch_size,
    step_size=step_size,
    n_steps=n_steps,
    inizialization=init,
    n_steps_sampling=n_steps_sampling,
    loss=loss,
    replay_buffer=replay_buffer,
    buffer=buffer,
    variance=retain_variance,
    n_components=n_components,
    more_variance=more_variance,
    sigma_pca=sigma_pca,
)

print("images generated: ", imgs.shape)

##################### EVALUATION #####################

real_dataset = []
labels = []
for batch in train_dataloader:
    real_dataset.append(batch[0])
real_dataset = torch.cat(real_dataset, dim=0)

is_score, is_score_stdv = utils.compute_is(imgs)
print("is score computed")
print(f"IS score: {is_score}")

torchmetrics_fid_score, ganmetrics_fid_score = utils.compute_fid(
    real_images=real_dataset, fake_images=imgs
)
print("fid score computed")
print(
    f"FID score for tochmetris: {torchmetrics_fid_score}\nFID score for ganmetrics: {ganmetrics_fid_score}"
)

kid_score, kid_score_stdv = utils.compute_kid(
    real_images=real_dataset, fake_images=imgs
)
print("kid score computed")
print(f"KID score: {kid_score}")


lpips = utils.compute_lpips(train_loader=train_dataloader, fake_images=imgs)
print("lpips score computed")
print(f"LPIPS score: {lpips}")


################# LOG ON WANDB ####################
if wandb:
    wandb.log(
        {
            "is_score": is_score.item(),
            "torchmetrics_fid_score": torchmetrics_fid_score,
            "ganmetrics_fid_score": ganmetrics_fid_score,
            "kid_score": kid_score.item(),
            "lpips": lpips.item(),
        }
    )
    wandb.log({"images": [wandb.Image(img) for img in imgs[:10]]})

################# CSV LOGGING ####################
logging_dict = {
    "model_name": model_name,
    "init_mode": init,
    "norm": threat_model,
    "explained_variance": retain_variance if init == "pca" else '-',
    "n_steps": n_steps,
    "FID": torchmetrics_fid_score.item(),
    "IS": is_score.item(),
    "LPIPS": lpips.item(),
    "KID": kid_score.item(),
    "num_samples": imgs.shape[0],
}
print("logging on csv", log_dir)
utils.log_on_csv(path=log_dir, log_dict=logging_dict)

if wandb:
    wandb.finish()
