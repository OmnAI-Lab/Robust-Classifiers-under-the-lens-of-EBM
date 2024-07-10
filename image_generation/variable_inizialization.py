# %%
import utils, enumeration, utils_func
import argparse
import torch
import gc
import numpy as np
import random

from torchvision.transforms import transforms


gc.collect()
torch.cuda.empty_cache()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# parse arguments
parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument("--wandb_logging", type=bool, default=False, help="log to wandb")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100"],
    help="dataset",
)
parser.add_argument("--epsilon", type=float, default=8 / 255, help="epsilon")
parser.add_argument("--alpha", type=float, default=2 / 255, help="alpha")
parser.add_argument(
    "--sampling_iterations", type=int, default=20, help="sampling iterations"
)
parser.add_argument(
    "--model_architecture",
    type=str,
    default="wide-resnet-34-10",
    help="architecture chosen for the generation",
)
parser.add_argument("--model_name", type=str, default="WideResNet", help="model name")
parser.add_argument(
    "--from_robustbench", type=bool, default=True, help="load model from robustbench"
)
parser.add_argument(
    "--init",
    type=str,
    default="random",
    choices=["random", "pca", "gaussian"],
    help="initialization for the generation",
)

parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--data_dir", type=str, default="./data", help="data directory")

# %%

try:
    args = parser.parse_args()
    wandb = args.wandb_logging
    batch_size = args.batch_size
    dataset = args.dataset
    epsilon = args.epsilon
    alpha = args.alpha
    sampling_iterations = args.sampling_iterations
    model_architecture = args.model_architecture
    model_name = args.model_name
    from_robustbench = args.from_robustbench
    initialization = args.init
    data_dir = args.data_dir
    seed = args.seed

    print("Using arguments from parser")

except:
    print("Using default arguments")
    wandb = False
    batch_size = 64
    dataset = "cifar10"
    epsilon = 8 / 255
    alpha = 2 / 255
    sampling_iterations = 20
    model_architecture = "wide-resnet-34-10"
    model_name = "WideResNet"
    from_robustbench = True
    init = "random"
    data_dir = "./data"
    seed = 0

# %%
# set seed for the reproducibility of the results
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# %%
#logging on wandb
if wandb:
    import wandb

    config = {
        "dataset": dataset,
        "batch_size": batch_size,
        "epsilon": epsilon,
        "alpha": alpha,
        "sampling_iterations": sampling_iterations,
        "model_architecture": model_architecture,
        "model_name": model_name,
        "from_robustbench": from_robustbench,
        "initialization": init,
    }
    wandb.init(name="generation", project="LE-PGD", entity="le-pgd", config=config)

# %%

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset_train, dataset_test = utils.load_clean_dataset(
    dataset=enumeration.BenchmarkDataset(dataset.lower()),
    n_examples=None,
    data_dir="./data",
    transform=transform,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False
)

# %%
## Define and load the model

# %%
from robustbench.utils import load_model
from torch import nn

from JEMPP.models.jem_models import F, CCF
from JEMPP.models.wideresnet import Wide_ResNet as WideResNet


from_RobustBench = True
from_sadajem = False

# %%

if from_RobustBench:
    model = load_model(
        model_name="Zhang2019Theoretically",
        dataset="cifar10",
        threat_model="Linf",
        model_dir="./models",
    )
    model = nn.DataParallel(model, device_ids=[0])
    model.to(device)

elif from_sadajem:

    model_cls = F
    model = model_cls(28, 10, norm="batch", n_classes=10, model="wrn")
    ckpt_dict = torch.load("./models/cifar10/Linf/sadajem5_95.5_9.4.pt")
    model.load_state_dict(ckpt_dict["model_state_dict"])
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    replay_buffer = ckpt_dict["replay_buffer"]

elif model_name == "local":
    model_path = "./models/wandb_model/" + model_name + ".pt"
    model = WideResNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
# %%
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
label_to_plot = [1,2,3]

x0, imgs = utils.new_samples(
    model=model,
    label=label_to_plot,
    batch_size=20,
    step_size=0.8,
    n_steps=50,
    inizialization="informative",
    n_steps_sampling=1,
    loss="energy_xy",
    replay_buffer=replay_buffer,
    buffer=buffer,
)

# %%
utils_func.plot_imgs_generated(
    x0,
    inizialization="informative",
    rows=10,
    size=(12, 10),
    path="./data/imgs_x0.pdf",
    name="imgs_x0: ",
)

# %%
utils_func.plot_imgs_generated(
    imgs,
    inizialization="informative",
    rows=10,
    size=(12, 10),
    path="./data/imgs.pdf",
    name="imgs: ",
)

# %%
