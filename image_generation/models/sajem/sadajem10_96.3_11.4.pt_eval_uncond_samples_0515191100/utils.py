import torch
import gc
import numpy as np


import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn import functional as Functional
from torch.nn import MaxPool2d as MaxPool2d
from enumeration import BenchmarkDataset, dataset_image_size, dataset_num_classes

from typing import Callable, Dict, Optional, Tuple

import utils_func


def fix_seeds(seed):
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


### Define the dataset preparation ###
def load_cifar10(n_examples: Optional[int], data_dir: str, transform: transforms):
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    if n_examples is not None:
        train_dataset.data = train_dataset.data[:n_examples]
        train_dataset.targets = train_dataset.targets[:n_examples]

    return train_dataset, test_dataset


def load_cifar100(n_examples: Optional[int], data_dir: str, transform: transforms):
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    if n_examples is not None:
        train_dataset.data = train_dataset.data[:n_examples]
        train_dataset.targets = train_dataset.targets[:n_examples]

    return train_dataset, test_dataset


CleanDatasetLoader = Callable[
    [Optional[int], str, Callable], Tuple[torch.Tensor, torch.Tensor]
]

_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100,
}


def load_clean_dataset(
    dataset: BenchmarkDataset,
    n_examples: Optional[int],
    data_dir: str,
    transform: transforms,
):
    return _clean_dataset_loaders[dataset](n_examples, data_dir, transform)


### Utility functions for the generation of adversarial examples ###


###########  INITIALIZATION functions ###########
def fix_seed(seed):
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_loss(y, loss_func, logits, **kwargs):
    import pickle, os

    if loss_func == "ce":
        y = y.long()
        loss = Functional.cross_entropy(logits, y)
    elif loss_func == "energy_x":
        loss = utils_func.compute_energy(logits).sum()
    elif loss_func == "energy_xy":
        loss = utils_func.compute_energy_xy(logits, y).sum()
    elif loss_func == "energy_xy_invertedsign":
        loss = utils_func.compute_energy_xy_energy_x_invertedsign(logits, y).sum()
    elif loss_func == "reference_energy_xy":
        cur_dir = os.getcwd()
        try:
            tm = kwargs["threat_model"]
        except:
            tm = "L2"
        with open(cur_dir + f"/data/{tm}_mean_en_cifar10.pkl", "rb") as f:
            mean_energy_per_class = pickle.load(f)

        loss = utils_func.compute_energy_xy_reference(
            logits, y, mean_energy_per_class
        ).sum()

    elif loss_func == "dist_reference":
        cur_dir = os.getcwd()

        with open(cur_dir + "/data/logits_mean_dict.pkl", "rb") as f:
            mean_logits_per_class = pickle.load(f)
        with open(cur_dir + "/data/logits_cov_dict_inv.pkl", "rb") as f:
            logits_cov_dict = pickle.load(f)

        loss = utils_func.compute_energy_xy_reference_covariance(
            logits, y, mean_logits_per_class, logits_cov_dict
        ).sum()
    else:
        raise ValueError("Invalid loss function")
    # loss should require grad
    loss.requires_grad_()
    return y, loss


def sample_X0_center(
    dataset="cifar10",
    n_classes=10,
    device="cuda",
    batch_size=5,
    y=None,
    dataset_folder="./data",
    dataset_image_size={"cifar10": 32},
):
    buffer = []
    from torch.distributions.multivariate_normal import MultivariateNormal

    size = [3, dataset_image_size[dataset], dataset_image_size[dataset]]

    folder = dataset_folder + "/" + dataset

    if dataset == "cifar10":
        centers = torch.load("%s_mean.pt" % folder)
        covs = torch.load("%s_cov.pt" % folder)

    y_mean = centers[y].to(device)
    y_cov = covs[y].to(device)
    dist = MultivariateNormal(
        y_mean,
        covariance_matrix=y_cov + 1e-4 * torch.eye(int(np.prod(size))).to(device),
    )
    buffer.append(dist)
    buffer = buffer[:10]
    sample = torch.tensor(buffer[0].sample((batch_size,)).view(-1, *size))
    return sample


def label_initialization(
    batch_size=64,
    image_size=32,
    n_classes=10,
    y=None,
    n_ch=3,
    buffer=None,
):
    assert buffer is not None, "buffer is not defined"

    size = [n_ch, image_size, image_size]

    new = torch.zeros(batch_size, n_ch, image_size, image_size)

    if y is None:
        for i in range(batch_size):
            index = np.random.randint(n_classes)
            dist = buffer[index]
            new[i] = dist.sample().view(size)
    else:
        for i in range(batch_size):
            dist = buffer[y[i]]
            new[i] = dist.sample().view(size)

    return torch.clamp(new, 0, 1).cpu()


def label_pca_initialization(
    buffer,
    dataset="cifar10",
    batch_size=64,
    y=None,
    more_variance=True,
    sigma_pca=0.005,
    mean_img=None,
    N_components=30,
    retained_variance=0.95,
    gaussian_blur=0,
    **kwargs,
):
    x0_pca = utils_func.sample_x0_pca(
        img_class=torch.load(f"./data/img_class_{int(y[0])}.pt"),
        more_variance=more_variance,
        sigma_pca=sigma_pca,
        mean_img=mean_img,
        batch_size=batch_size,
        N_components=N_components,
        retained_variance=retained_variance,
        gaussian_blur=gaussian_blur,
        **kwargs,
    )
    x0_info = label_initialization(
        batch_size,
        image_size=dataset_image_size[dataset],
        n_classes=dataset_num_classes[dataset],
        y=y,
        buffer=buffer,
    )
    indices = torch.randint(0, 2, (batch_size,))

    return torch.stack(
        [x0_pca[i] if indices[i] == 0 else x0_info[i] for i in range(batch_size)]
    )


def noise_injection_initialization(
    batch_size=64,
    image_size=32,
    n_classes=10,
    n_ch=3,
    y=None,
    test_set=None,  # test dataset to sample the noise
    buffer=None,
    epsilon=8 / 255,
    alpha=10,
):
    assert buffer is not None, "buffer is not defined"
    assert test_set is not None, "test_set is not defined"

    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, shuffle=False
    )
    el = next(iter(test_dataloader))
    images, labels = el[0], el[1]

    # from the dataset get all the images of a certain class
    class_images = []

    for i in range(n_classes):
        indices = []
        for j, l in enumerate(labels):
            if l.item() == i:
                indices.append(j)

        class_images.append(images[indices])

    new = torch.zeros(batch_size, n_ch, image_size, image_size)

    if y is None:
        y = []
        for i in range(batch_size):
            index = np.random.randint(n_classes)
            y.append(index)

    indexes = [0 for _ in range(n_classes)]

    for i in range(batch_size):
        noise = torch.randn(n_ch, image_size, image_size).normal_(0, alpha * epsilon)
        c = y[i].item()
        ind = indexes[c]

        img = class_images[c][ind]

        indexes[c] = ind + 1
        new[i] = img + noise
    return torch.clamp(new, 0, 1).cpu()


###########    SAMPLERS    ###########
def get_init_images(
    replay_buffer=[],
    batch_size=32,
    y=None,
    inizialization="informative",
    sigma_pca=0.005,
    mean_img=None,
    N_components=30,
    retained_variance=0.95,
    more_variance=True,
    dataset="cifar10",
    buffer=None,
    eps=None,
    **kwargs,
):

    if len(replay_buffer) != 0:
        return [], []

    inizializations = {
        "random": utils_func.random_initialization,
        "informative": label_initialization,
        "pca": utils_func.sample_x0_pca,
        "informative+pca": label_pca_initialization,
        "noise_injection": noise_injection_initialization,
        "center": sample_X0_center,
    }

    if inizialization == "random":
        ret = inizializations[inizialization](
            batch_size, size=dataset_image_size[dataset]
        )
        return ret, []
    elif inizialization == "informative":
        ret = inizializations[inizialization](
            batch_size,
            image_size=dataset_image_size[dataset],
            n_classes=dataset_num_classes[dataset],
            y=y,
            buffer=buffer,
        )
        return ret, []
    elif inizialization == "pca":
        try:
            gaussian_blur = kwargs["gaussian_blur"]
        except:
            gaussian_blur = 0

        try:
            sigma = kwargs["sigma"]
        except:
            sigma = sigma_pca
        ret = inizializations[inizialization](
            img_class=torch.load(f"./data/img_class_{int(y[0])}.pt"),
            n_channels=3,
            img_size=dataset_image_size[dataset],
            more_variance=more_variance,
            sigma_pca=sigma,
            mean_img=mean_img,
            batch_size=batch_size,
            N_components=N_components,
            retained_variance=retained_variance,
            gaussian_blur=gaussian_blur,
        )
        return ret, []

    elif inizialization == "informative+pca":
        try:
            retained_variance = kwargs["variance"]
        except:
            retained_variance = 0.8
        try:
            gaussian_blur = kwargs["gaussian_blur"]
        except:
            gaussian_blur = 0

        ret = inizializations[inizialization](
            buffer=buffer,
            dataset=dataset,
            batch_size=batch_size,
            y=y,
            more_variance=more_variance,
            sigma_pca=sigma_pca,
            mean_img=mean_img,
            N_components=N_components,
            retained_variance=retained_variance,
            gaussian_blur=gaussian_blur,
        )
        return ret, []
    elif inizialization == "noise_injection":
        try:
            test_set = kwargs["test"]
        except:
            raise ValueError("test_set is not defined")
        try:
            alpha = kwargs["alpha"]
        except:
            alpha = 50
        try:
            epsilon = kwargs["epsilon"]
        except:
            epsilon = 8 / 255

        ret = inizializations[inizialization](
            batch_size=batch_size,
            image_size=dataset_image_size[dataset],
            n_classes=dataset_num_classes[dataset],
            y=y,
            test_set=test_set,
            buffer=buffer,
            epsilon=epsilon,
            alpha=alpha,
        )
        return ret, []
    elif inizialization == "center":
        ret = inizializations[inizialization](
            dataset=dataset,
            n_classes=dataset_num_classes[dataset],
            device="cuda",
            batch_size=batch_size,
            dataset_folder="./data",
        )

        return ret, []
    else:
        raise ValueError("Invalid inizialization method")


def sample_q_Heun(
    model,
    replay_buffer=[],
    batch_size=10,
    step_size=1,
    y=None,
    n_steps=20,
    inizialization="informative",
    loss_func="ce",
    sigma_pca=0.005,
    mean_img=None,
    N_components=20,
    retained_variance=0.90,
    more_variance=True,
    device="cuda",  # "cuda" or "cpu"
    dataset="cifar10",
    buffer=None,
    **kwargs,
):
    momentum = kwargs.get("momentum", 0)
    noise_variance = kwargs.get("noise_variance", 0.001)
    alpha = kwargs.get("alpha", 1)
    retained_variance = kwargs.get("variance", 0.95)
    alpha = kwargs.get("alpha", np.sqrt(2 * step_size))

    assert buffer is not None, "buffer is not defined"

    # initialize the starting points
    x0, inds = get_init_images(
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        y=y,
        inizialization=inizialization,
        sigma_pca=sigma_pca,
        mean_img=mean_img,
        N_components=N_components,
        retained_variance=retained_variance,
        more_variance=more_variance,
        dataset=dataset,
        buffer=buffer,
        **kwargs,
    )
    velocity = torch.zeros_like(x0).to(device)

    noise = torch.randn(x0.shape, device=x0.device)
    samples = x0.clone().detach().to(device)
    model.eval()

    for _ in range(n_steps):
        rum = noise.normal_(0, noise_variance).to(device)
        samples.requires_grad = True
        logits = model(samples)

        y, loss = compute_loss(y, loss_func, logits, **kwargs)

        grad1 = torch.autograd.grad(
            loss, samples, retain_graph=False, create_graph=False
        )[0].to(device)

        assert (
            grad1 is not None
        ), "gradient in the first step of the heun sampling is None"

        samples2 = samples.detach() - step_size * grad1.detach()
        samples2.requires_grad = True

        logits = model(samples2)

        y, loss = compute_loss(y, loss_func, logits, **kwargs)

        grad2 = torch.autograd.grad(
            loss, samples2, retain_graph=False, create_graph=False
        )[0].to(device)
        assert (
            grad2 is not None
        ), "gradient in the second step of the heun sampling is None"

        velocity = (
            -0.5 * (grad2 + grad1).detach() * step_size
            if momentum <= 0
            else momentum * velocity - 0.5 * (grad2 + grad1).detach() * step_size
        )
        samples = samples.detach() - velocity + alpha * rum

        samples.data.clamp_(0, 1)

        gc.collect()
        torch.cuda.empty_cache()

    if len(replay_buffer) > 0:
        replay_buffer[inds] = samples.cpu()

    return x0, samples


def sample_q_SGLD(
    model,
    replay_buffer=[],
    batch_size=10,
    step_size=1,
    y=None,
    n_steps=20,
    inizialization="informative",
    loss_func="ce",
    sigma_pca=0.005,
    mean_img=None,
    N_components=20,
    retained_variance=0.90,
    more_variance=True,
    dataset="cifar10",
    buffer=None,
    energy_dict=None,
    eps=None,
    **kwargs,
):
    """
    Sample from the q distribution using Stochastic Gradient Langevin Dynamics (SGLD).

    Args:
        model (nn.Module): The neural network model.
        replay_buffer (list): The replay buffer.
        batch_size (int): The batch size.
        step_size (float): The step size.
        y (Tensor): The target labels.
        n_steps (int): The number of steps.
        inizialization (str): The initialization method.
        loss_func (str): The loss function.
        sigma_pca (float): The standard deviation for PCA.
        mean_img (Tensor): The mean image.
        N_components (int): The number of PCA components.
        retained_variance (float): The retained variance for PCA.
        more_variance (bool): Whether to use more variance for PCA.
        dataset (str): The dataset name.
        buffer (Tensor): The buffer.
        energy_dict (dict): The energy dictionary.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple: A tuple containing the initial images, the sampled images, and the energy dictionary.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retained_variance = kwargs.get("variance", 0.95)
    alpha = kwargs.get("alpha", np.sqrt(2 * step_size))
    momentum = kwargs.get("momentum", 0)
    noise_variance = kwargs.get("noise_variance", 0.001)
    HM = kwargs.get("HM", False)
    compute_energy_model = kwargs.get("compute_energy_model", None)
    adaptive = kwargs.get("adaptive", -1)   
    # init the image X0 based on the MODE
    x0, inds = get_init_images(
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        y=y,
        inizialization=inizialization,
        sigma_pca=sigma_pca,
        mean_img=mean_img,
        N_components=N_components,
        retained_variance=retained_variance,
        more_variance=more_variance,
        dataset=dataset,
        buffer=buffer,
        eps=eps,
        **kwargs,
    )
    # QUI ENTRA
    velocity = torch.zeros_like(x0).to(device)

    samples = x0.clone().detach().to(device)
    noise = torch.randn(x0.shape, device=x0.device)
    flag_noise = torch.randn(x0[0].shape, device=x0.device)
    flag_energy_xy = 10 ^ 5
    eps = 1 if eps is None else eps
    model_name = kwargs.get("model_name", "")
    # print(f"the model name is {model_name}")
    model.eval()
    norm = kwargs.get("norm", "Linf")
    # print(f"the norm is {norm}")
    if adaptive >= 0:
        # read the mean energy of the samples
        import pickle, os

        cur_dir = os.getcwd()

        with open(cur_dir + f"/data/mean_en_cifar10.pkl", "rb") as f:
            mean_energy_per_class = pickle.load(f)

        ref_en = mean_energy_per_class[y[0].item()]

        finished_indexes = []
        flag_noise = torch.randn(x0[0].shape, device=x0.device)
        samples = x0.clone().to(device)
        noise = torch.randn(x0[0].shape, device=x0.device)
        velocity = torch.zeros_like(x0).to(device)

        for _ in range(n_steps):
            rum = []

            for i in range(len(samples)):
                if compute_energy_model is not None:
                    if i in finished_indexes:
                        continue
                    else:
                        label_sample = y[0]
                        logits = model(samples[i])
                        for j in range(10):
                            energy_dict[f"{label_sample}_{j}"].append(
                                utils_func.compute_energy_xy(logits, j).item()
                            )

                en_sample = utils_func.compute_energy_xy(model(samples[i]), y[0])

                if (en_sample.item() - ref_en) ** 2 < adaptive:
                    if i not in finished_indexes:
                        finished_indexes.append(i)
                    # print("not computing")
                    continue

                else:
                    samples.requires_grad = True
                    sample = samples[i].clone().detach().unsqueeze(0)
                    sample.requires_grad = True

                    y, loss = compute_loss(y, loss_func, model(sample), **kwargs)
                    print(f"the loss is {loss}")

                    grad = torch.autograd.grad(
                        loss,
                        sample,
                        retain_graph=False,
                        create_graph=False,
                    )[0].to(device)

                    velocity[i] = (
                        -step_size / 2 * grad
                        if momentum <= 0
                        else momentum * velocity[i] - step_size / 2 * grad
                    )
                    if not HM:
                        samples.requires_grad = False
                        rum = noise.normal_(0, noise_variance).to(device)
                        sample = (
                            sample.detach() + velocity[i].unsqueeze(0) + alpha * rum
                        )
                        sample.data.clamp_(0, 1)
                        samples[i] = sample
                    else:
                        samples.requires_grad = False
                        energy_xy = utils_func.compute_energy_xy(model(sample), y[0])
                        flag_energy_xy = 10 ^ 5
                        counter = 0

                        while flag_energy_xy > energy_xy.item() and counter < 100:
                            flag_rum = noise.normal_(0, noise_variance).to(device)
                            flag_sample = (
                                sample.detach()
                                + velocity[i].unsqueeze(0)
                                + alpha * flag_rum
                            )
                            flag_sample.data.clamp_(0, 1)

                            flag_logit = model(flag_sample)
                            flag_energy_xy = utils_func.compute_energy_xy(
                                flag_logit, y[i]
                            )
                            counter += 1

                        samples[i] = flag_sample

    else:
        for s in range(n_steps):
            
            if compute_energy_model is not None:
                for index, sample in enumerate(samples):
                    sample = sample.unsqueeze(0)
                    for i in range(10):
                        logit = model(sample)
                        print(f"the logits are {logit.shape}")

                        energy_dict[f"{y[index]}_{i}"].append(
                            utils_func.compute_energy_xy(logit, i).item()
                        )
                        print(energy_dict)
            # print(f"the samples are {samples.shape}")
            samples.requires_grad = True
            logits = model(samples) if model_name == "" else model.classify(samples)
            # print(f"the logits are {logits.shape}")
            y, loss = compute_loss(y, loss_func, logits, **kwargs)

            grad = torch.autograd.grad(
                loss, samples, retain_graph=False, create_graph=False
            )[0].to(device)

            velocity = (
                -step_size / 2 * grad
                if momentum <= 0
                else momentum * velocity - step_size / 2 * grad
            )
            
            if not HM:
                rum = noise.normal_(0, noise_variance).to(device)
                noise_added = velocity + alpha * rum
                noise_added.data.clamp_(-eps, eps)
                if norm == "Linf":
                    samples = samples.detach() + noise_added
                    samples.data.clamp_(0, 1)
                else:
                    samples = samples.detach() + noise_added
                    samples = samples.renorm(p=2, dim=0, maxnorm=eps)
                    samples.data.clamp_(0, 1)
            else:
                rum = []
                
                for i, sample in enumerate(samples):
                    energy_xy = utils_func.compute_energy_xy(model(sample), y[i])
                    counter = 0

                    while flag_energy_xy > energy_xy.item() and counter < 100:
                        flag_rum = flag_noise.normal_(0, noise_variance).to(device)
                        flag_noise_added = velocity[i] + alpha * flag_rum
                        flag_noise_added.data.clamp_(-eps, eps)

                        flag_sample = sample.detach() + flag_noise_added
                        flag_sample.data.clamp_(0, 1)

                        flag_logit = model(flag_sample)
                        flag_energy_xy = utils_func.compute_energy_xy(flag_logit, y[i])
                        counter += 1

                    rum.append(flag_rum)

                rum = torch.stack(rum)
                noise_added = velocity + alpha * rum
                if norm == "Linf":
                    samples = samples.detach() + noise_added
                    samples.data.clamp_(0, 1)
                else:
                    samples = samples.detach() + noise_added
                    samples = samples.renorm(p=2, dim=0, maxnorm=eps)
                    samples.data.clamp_(0, 1)
                    

    if len(replay_buffer) > 0:
        replay_buffer[inds] = samples.cpu()

    ret_energy = None if compute_energy_model is None else energy_dict

    return x0, samples, ret_energy


############################################
def new_samples(
    model,
    label=[],
    batch_size=32,
    step_size=1,
    n_steps=100,
    inizialization="informative",
    n_steps_sampling=1,
    loss="ce",
    algo="SGLD",
    replay_buffer=[],
    buffer=None,
    dataset="cifar10",
    sigma_pca=0.005,
    eps=None,
    **kwargs,
):
    """
    generate n_samples"""
    assert buffer is not None, "buffer is not defined"
    if isinstance(label, int):
        label = [label]
    assert len(label) != 0 and isinstance(label, list), "empty label field"
    assert label is not None, "label is None"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        compute_energy_model = kwargs["compute_energy_model"]
        energy_dict = {f"{i}_{j}": [] for j in range(10) for i in range(10)}
    except:
        compute_energy_model = None
        energy_dict = None

    torch.set_grad_enabled(True)

    X0 = []
    samples = []

    for l in label:
        # add elements into tensor
        y = torch.tensor([l] * batch_size).to(device)

        # n_steps_sampling help us to save images
        for _ in range(n_steps_sampling):
            if algo == "SGLD":
                x0, final_samples, energy_dict = sample_q_SGLD(
                    model=model,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    step_size=step_size,
                    y=y,
                    n_steps=n_steps,
                    inizialization=inizialization,
                    loss_func=loss,
                    dataset=dataset,
                    buffer=buffer,
                    sigma_pca=sigma_pca,
                    energy_dict=energy_dict,
                    eps=eps,
                    **kwargs,
                )
            elif algo == "Heun":
                x0, final_samples = sample_q_Heun(
                    model,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    step_size=step_size,
                    y=y,
                    n_steps=n_steps,
                    inizialization=inizialization,
                    loss_func=loss,
                    dataset=dataset,
                    sigma_pca=sigma_pca,
                    device=device,
                    buffer=buffer,
                    **kwargs,
                )
            else:
                raise ValueError("Invalid algorithm")

            X0.append(x0)
            samples.append(final_samples)

            gc.collect()
            torch.cuda.empty_cache()

        # concatante the list of tensors to a single tensor
        tensor_samples = torch.cat(samples, dim=0)
        tensor_x0 = torch.cat(X0, dim=0)

    if compute_energy_model is not None:
        return tensor_x0, tensor_samples, energy_dict
    else:
        return tensor_x0, tensor_samples


############## EVALUATION FUNCTIONS  ##############
# from torchmetrics.image.kid import KernelInceptionDistance
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.image.inception import InceptionScore
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from pytorch_gan_metrics import get_fid


# # compute KID
# def compute_kid(
#     real_images,
#     fake_images,
#     max_subset_size=10,
#     device="cuda",
#     feature="logits_unbiased",
#     normalize=True,
# ):
#     batch_size = 500

#     kid = KernelInceptionDistance(
#         feature=feature, subset_size=max_subset_size, normalize=normalize
#     ).to(device=device)

#     for i in range(0, real_images.size(0), batch_size):
#         batch = real_images[i : i + batch_size].to(device)
#         kid.update(batch, real=True)

#     for i in range(0, fake_images.size(0), batch_size):
#         batch = fake_images[i : i + batch_size].to(device)
#         kid.update(batch, real=False)

#     KID = kid.compute()

#     return KID[0], KID[1]


# def compute_is(imgs, device="cuda"):
#     """
#     Compute the Inception Score
#     Imgs is a tensor of shape (N, C, H, W) with values in [0, 1]
#     """
#     batch_size = 500
#     inception = InceptionScore(feature="logits_unbiased", normalize=True).to(device)

#     for i in range(0, imgs.size(0), batch_size):
#         batch = imgs[i : i + batch_size].to(device)
#         inception.update(batch)
#     IS = inception.compute()
#     return IS[0], IS[1]


# def compute_fid(real_images, fake_images, device="cuda", feature=2048, normalize=True):
#     """
#     All images will be resized to 299 x 299 which is the size of the original training data.
#     The boolian flag real determines if the images should update the statistics of the real
#     distribution or the fake distribution.
#     real_images: tensor of shape (N, C, H, W) with values in [0, 1], IT SHOULD BE CIFAR-10 TRAIN SET or TEST SET
#     fake_images: tensor of shape (M, C, H, W) with values in [0, 1]
#     """
#     ### compute the FID score with torchmetrics

#     batch_size = 500
#     fid = FrechetInceptionDistance(feature=feature, normalize=normalize).to(
#         device=device
#     )

#     for i in range(0, real_images.size(0), batch_size):
#         batch = real_images[i : i + batch_size].to(device)
#         fid.update(batch, real=True)

#     for i in range(0, fake_images.size(0), batch_size):
#         batch = fake_images[i : i + batch_size].to(device)
#         fid.update(batch, real=False)

#     ### compute the FID score with python-gan-metrics
#     FID = get_fid(
#         fake_images,
#         "/home/briglia/basefolder/LE_PGD/ablation/data/cifar10_stats/cifar10.train.npz",
#     )

#     return fid.compute(), FID


# def compute_lpips(train_loader, fake_images, device="cuda", net="alex"):
#     """
#     Compute the LPIPS score
#     fake_images is a tensor of shape (N, C, H, W) with values in [0, 1]
#     train loader is the dataloader of the training set
#     """

#     imgs, labels = [], []
#     for i, (img, label) in enumerate(train_loader):
#         imgs.append(img)
#         labels.append(label)

#     imgs = torch.cat(imgs, dim=0)
#     labels = torch.cat(labels, dim=0)

#     # sort the images for class, to compute the LPIPS
#     imgs = imgs[torch.argsort(labels)]

#     assert imgs.size() == fake_images.size()

#     batch_size = 1000
#     lpips = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(
#         device=device
#     )

#     for i in range(0, imgs.size(0), batch_size):
#         batch_real = imgs[i : i + batch_size].to(device)
#         batch_fake = fake_images[i : i + batch_size].to(device)
#         lpips.update(batch_real, batch_fake)

#     return lpips.compute()


# ############## LOGGING  ##############
# def log_on_csv(path="/home/briglia/basefolder/LE_PGD/ablation/data", **kwargs):
#     import os
#     import csv

#     filename = path + "/logging.csv"
#     logging_dict = kwargs["log_dict"]

#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)
#         with open(filename, "w", newline="\n") as file:
#             writer = csv.writer(file, delimiter=";")
#         writer.writerow(logging_dict.keys())

# with open(filename, "a", newline="\n") as file:
#     writer = csv.writer(file, delimiter=";")
#     writer.writerow(logging_dict.values())
