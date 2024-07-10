import torch
import sys, os, pathlib, yaml
import wandb
import sys
import argparse

from robustbenchmaster.robustbench.model_zoo.architectures.resnet import ResNet18
from robustbenchmaster.robustbench.model_zoo.architectures.wideresnet import WideResNet
from robustbenchmaster.robustbench.utils import (
    clean_accuracy,
    parse_args,
)

from robustbenchmaster.robustbench.utils import (
    parse_args,
)
from eval import *
from robustbenchmaster.robustbench.model_zoo.enums import *

try:
    args = parse_args()

    torch.manual_seed(args.seed)
except:
    torch.manual_seed(0)



parent_dir = str(pathlib.Path(__file__).parent.absolute())


parser = argparse.ArgumentParser(description="PyTorch CW Evaluation")

parser.add_argument(
    "--name",
    required=True,
    type=str,
    help="name of the model to be evaluated",
)
parser.add_argument(
    "--dataset",
    default="svhn",
    type=str,
    choices=["cifar10", "cifar100", "svhn"],
    help="name of the dataset to be evaluated",
)
parser.add_argument(
    "--model_architecture",
    default="resnet18",
    type=str,
    choices=["resnet18", "wideresnet"],
    help="name of the model architecture to be evaluated",
)
parser.add_argument(
    "--wandb-log",
    default=False,
    type = bool,
    help="log to wandb",
    action="store_true"
)

args = parser.parse_args()
name = args.name
dataset_name = args.dataset
model_architecture = args.model_architecture


device = torch.device("cuda:0")
num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}

architectures = {
    "resnet18": ResNet18(num_classes=num_classes[dataset_name]).to(device),
    "wideresnet": WideResNet(depth=28, num_classes=num_classes[dataset_name]).to(
        device
    ),
}


# raed a yaml file with the setup variables defined by the user
with open(str(parent_dir) + "/submit_files" + "/setup.yaml", "r") as stream:
    try:
        setup = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

_model_name = [name.split("/")[-1].split(".")[0]]

_dataset = dataset_name


_batch_size = setup["batch_size"] if setup["batch_size"] else args.batch_size
_n_ex = setup["n_ex"] if setup["n_ex"] else args.n_ex
_to_disk = setup["to_disk"] if setup["to_disk"] else args.to_disk
_eps = setup["eps"] if setup["eps"] else args.eps

if args.wandb_log:

    run = wandb.init(
        # Set the project where this run will be logged
        name="CW-" + str(_model_name[0]),
        project="robust evaluations",
        # Track hyperparameters and run metadata
        config={"clean_accuracy": 0, "adversarial_accuracy": 0},
    )

for s in [name]:

    print("model to be evaluated is: ", s)
    model = architectures[model_architecture]

    if torch.cuda.is_available():
        # Load the model on CPU first
        model.load_state_dict(
            torch.load("./adv_trained_models/" + str(s), map_location="cpu")
        )
        # Move the model to CUDA device
        model.to(device)
    else:
        # Load the model on CPU
        model.load_state_dict()

    model.eval()

    clean_accuracy, adv_accuracy = cw_benchmark(
        model=model,
        n_examples=_n_ex,
        dataset=_dataset,
        threat_model="Linf",
        to_disk=_to_disk,
        model_name=model,
        data_dir="./downloaded_data",
        device=device,
        batch_size=_batch_size,
        eps=_eps / 255,
    )

    # make the dir if it does not exist
    dataset_root_dir = str(parent_dir) + "/submit_files" + f"/{_dataset}"
    print(dataset_root_dir, "being created")
    if not os.path.exists(dataset_root_dir):  # and args.cw_save:
        os.makedirs(dataset_root_dir)
        
if args.wandb_log:
    wandb.log(
        {   
            "dataset": _dataset,
            "attack_type": "CW",
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "model_name": model,
        }
    )
