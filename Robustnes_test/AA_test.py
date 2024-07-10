import torch
import wandb
from eval import benchmark
from robustbenchmaster.robustbench.model_zoo.architectures.resnet import ResNet18
from robustbenchmaster.robustbench.model_zoo.architectures.wideresnet import WideResNet
from ti_resnet import ti_ResNet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser(description="PyTorch Robustness Evaluation")
parser.add_argument(
    "--name",
    default="hearty-lake-494_robust_model_at_epoch_119.pt",
    type=str,
    help="name of the model to be evaluated",
)
parser.add_argument(
    "--dataset",
    default="svhn",
    type=str,
    choices=["cifar10", "cifar100", "svhn", "tinyimagenet"],
    help="name of the dataset to be evaluated",
)
parser.add_argument(
    "--model_architecture",
    default="resnet18",
    type=str,
    choices=["resnet18", "wideresnet", "ti_resnet"],
    help="name of the model architecture to be evaluated",
)
parser.add_argument(
    "--log",
    default=1,
    type=int,
    help="log to file",
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
log = args.log


num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "tinyimagenet": 200}

architectures = {
    "resnet18": ResNet18(num_classes=num_classes[dataset_name]).to(device),
    "wideresnet": WideResNet(depth=34, num_classes=num_classes[dataset_name]).to(
        device
    ),
    "ti_resnet": ti_ResNet18(classes=num_classes[dataset_name]).to(device),
}

if args.wandb_log:
    run = wandb.init(
        # Set the project where this run will be logged
        project="robust evaluations",
        name=dataset_name + "-" + name,
        # Track hyperparameters and run metadata
        config={
            "clean_accuracy": 0,
            "adversarial_accuracy": 0,
            "dataset": dataset_name,
        },
    )
    print("model to be evaluated is: ", run.name)

# %%
for s in [name]:
    print("model to be evaluated is: ", s)
    model = architectures[model_architecture]
    old_state_dict = torch.load("./adv_trained_models/" + str(s), map_location="cpu")
    state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        state_dict[k] = v
    del old_state_dict

    if torch.cuda.is_available():
        model.load_state_dict(state_dict)

        model.to(device)
    else:
        model.load_state_dict(state_dict)

    # model = nn.DataParallel(model)
    model.eval()

    clean_acc, robust_acc = benchmark(
        model,
        dataset=dataset_name,
        threat_model="Linf",
        eps=8 / 255,
        batch_size=128,
        device=device,
        to_disk=True,
        model_name=str(s),
    )

    print(
        f" for model {s} clean accuracy is {clean_acc} and robust accuracy is {robust_acc}"
    )
    if args.wandb_log:
        wandb.log({"clean_accuracy": clean_acc, "adversarial_accuracy": robust_acc})

    if log == 1:
        # if there is no log file create it
        import csv, os

        print(f"logging on ./log/{dataset_name}_{name.split('/')[0]}.txt")
        if not os.path.exists("./log"):
            os.makedirs("./log", exist_ok=True)

        with open(f"./log/{dataset_name}_{name.split('/')[0]}.txt", "w") as f:
            writer = csv.writer(f)
            writer.writerow([clean_acc, robust_acc])
