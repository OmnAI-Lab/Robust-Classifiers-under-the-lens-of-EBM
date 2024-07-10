import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import os
from sklearn.decomposition import PCA

from torch.nn import MaxPool2d as MaxPool2d


def compute_energy(logits):
    # Compute the energy as the negative logarithm of the sum of the exponentials of the logits
    # energy = -torch.log(torch.sum(torch.exp(logits), dim=1))
    energy = -torch.logsumexp(logits, dim=1)
    # log-sum-exp in torch
    return energy


def compute_energy_xy(logits, labels):
    # Get the logit corresponding to the correct label
    correct_logits = logits[torch.arange(logits.size(0)), labels]
    energy = -correct_logits
    return energy


def compute_energy_xy_energy_x_invertedsign(logits, labels):
    e_xy = compute_energy_xy(logits, labels)
    e_x = compute_energy(logits)
    return e_xy - e_x


def compute_energy_xy_reference(logits, labels, mean_energy_per_class):

    import torch.nn.functional as F

    ref_en = []    
    for i in range(len(labels)):
        ref_en.append(mean_energy_per_class[labels[i]])
    en = compute_energy_xy(logits, labels)
    ref_en = torch.tensor(ref_en, dtype=torch.float32, device=en.device)
    loss = F.mse_loss(ref_en, en)
    return  loss


def compute_energy_xy_reference_covariance(logits, labels , mean_energy_per_class, cov_energy):
    losses = []
    # implement the malanobis distance 
    for index in range(len(labels)):
        mean_logit = mean_energy_per_class[labels[index].item()] # numpy array
        mean_logit = torch.tensor(mean_logit, dtype=torch.float32, device=logits.device)
        cov_logit = cov_energy[labels[index].item()] # numpy array
        cov_logit = torch.tensor(cov_logit, dtype=torch.float32, device=logits.device)
        
        x = logits[index]
        
        loss = (x - mean_logit).T @ cov_logit @ (x- mean_logit)
        losses.append(loss)
    loss =  torch.stack(losses)
    return loss
    

def imgpt_to_np(img_py):
    "img_py is 3xHxW and we return is HxWx3 in 0...1"
    return np.transpose(img_py.reshape((3, 32, 32)), (1, 2, 0))


def plot_imgs_generated(
    imgs_generated,
    inizialization,
    rows=20,
    size=(20, 30),
    path="./data/imgs.pdf",
    name="test",
):

    grid = tv.utils.make_grid(
        imgs_generated.cpu(), nrow=imgs_generated.shape[0] // rows, normalize=True, 
    )
    grid = grid.permute(1, 2, 0)
    
    fig_size = (size[0], size[1])
     
    plt.figure(figsize=fig_size)
    
    
    # no border and no indexes on the axes
    plt.axis("off")
    
    plt.imshow(grid)

    plt.savefig(path, format="pdf")
    plt.show()


def random_initialization(
    batch_size=12, n_channels=3, size=32, mean=0, std=1, noise_distribution="uniform"
):
    ret = {
        "uniform": torch.FloatTensor(batch_size, n_channels, size, size).uniform_(
            mean, std
        ),
        "normal": torch.FloatTensor(batch_size, n_channels, size, size).normal_(
            mean, std
        ),
    }
    return ret[noise_distribution]


def category_mean(
    dload_train,
    n_classes=10,
    name_dataset="cifar10",
    n_channels=3,
    image_size=32,
    data_dir="./data",
):
    size = [n_channels, image_size, image_size]

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if os.path.exists(data_dir + "/" + name_dataset + "_mean.pt") and os.path.exists(
        data_dir + "/" + name_dataset + "_cov.pt"
    ):
        print("> Mean and Covariance Already Computed, skipping")
        return

    centers = torch.zeros([n_classes, int(np.prod(size))])
    covs = torch.zeros([n_classes, int(np.prod(size)), int(np.prod(size))])

    im_test, targ_test = [], []
    for im, targ in dload_train:
        im_test.append(im)
        targ_test.append(targ)
    im_test, targ_test = torch.cat(im_test), torch.cat(targ_test)

    for i in range(n_classes):
        imc = im_test[targ_test == i]
        imc = imc.view(len(imc), -1)
        mean = imc.mean(dim=0)
        sub = imc - mean.unsqueeze(dim=0)
        cov = sub.t() @ sub / len(imc)
        centers[i] = mean
        covs[i] = cov
        if not os.path.exists(data_dir + "/" + name_dataset):
            os.makedirs(data_dir + "/" + name_dataset, exist_ok=True)

        torch.save(imc, f"{data_dir}/img_class_{i}.pt")

    # print(time.time() - start)
    folder = data_dir + "/" + name_dataset
    torch.save(centers, "%s_mean.pt" % folder)
    torch.save(covs, "%s_cov.pt" % folder)


def sample_x0_pca(
    img_class,
    more_variance=True,
    img_size=32,
    n_channels=3,
    sigma_pca=0.005,
    mean_img=None,
    batch_size=10,
    N_components=27,
    retained_variance=0.75,
    gaussian_blur=0,
):
    """
    Function that plot x0 sapled with PCA and return x0.
    """

    import numpy as np
    import cv2  # opencv
    from scipy.ndimage import gaussian_filter

    if retained_variance is not None:
        N_components = None  # ask for full components

    if gaussian_blur > 0:
        a = [np.array(el.view(img_size, img_size, n_channels)) for el in img_class]
        list_data = [
            cv2.resize(
                gaussian_filter(np.array(el), sigma=(gaussian_blur, gaussian_blur, 1)),
                dsize=(img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )
            for el in a
        ]
        list_of_tensors = [torch.tensor(ar.reshape(-1)) for ar in list_data]

        img_class = torch.stack(list_of_tensors)

    pca = PCA(n_components=N_components)  # (n_samples, n_features)
    princ = pca.fit(img_class)

    mu_ = princ.mean_ if mean_img is None else img_class[mean_img]

    # else select the 10 less dominant just to see (assume PrinComp are sorted)
    PC = princ.components_ if more_variance else princ.components_[-11:-1, :]
    singular_values = (
        princ.singular_values_ if more_variance else princ.singular_values_[-11:-1]
    )
    N_components = PC.shape[0]  # overwrite how many components we have

    if retained_variance is not None:

        tot_pca_var_energy = princ.explained_variance_.sum()
        cum_energy = np.cumsum(princ.explained_variance_) / tot_pca_var_energy
        # plt.plot(cum_energy)
        # plt.xlabel("n_component")
        # plt.ylabel("retained variance")
        # keep at least 80% of variance
        clip_component = np.argmin(cum_energy < retained_variance)
        PC = princ.components_[:clip_component]  # all till clip_component
        N_components = PC.shape[0]
        singular_values = princ.singular_values_[:clip_component]

    print(
        f"Using number of componets {PC.shape[0]} with  retained_variance {retained_variance}"
    )
    sigma_pca, n_plot = (
        sigma_pca,
        batch_size,
    )  # the noise std. dev is 0.01; we plot 7x7 images
    # fig, axs = plt.subplots(n_plot, n_plot, figsize=(10,)*2)
    # total = n_plot

    ret = []

    for _ in range(batch_size):
        # alpha_i = eigenvalue_i*N(0, sigma_pca)
        alpha = (singular_values * np.random.randn(N_components) * sigma_pca).reshape(
            N_components, 1
        )
        # \mu + sum_i alpha_i * PrincComp_i
        perturb = mu_ + np.dot(alpha.T, PC)
        perturb = np.clip(perturb, 0, 1.0)  # clip so that remains in valid range
        # i, j = idx//n_plot, idx % n_plot  # getting idx for plotting
        # viz = imgpt_to_np(perturb)
        # axs[i, j].imshow(viz)  # viz
        # axs[i, j].axis('off')
        ret.append(perturb)

    return torch.reshape(
        torch.Tensor(np.asarray(ret)), (batch_size, n_channels, img_size, img_size)
    )
