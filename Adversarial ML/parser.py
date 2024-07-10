import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4, metavar='W',
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # CUDA
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')

    # Perturbation parameters
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='perturbation epsilon (default: 8/255)')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='number of perturbation steps (default: 10)')
    parser.add_argument('--step-size', type=float, default=2/255,
                        help='perturbation step size (default: 2/255)')

    # Regularization parameters
    parser.add_argument('--beta', type=float, default=6.0,
                        help='regularization parameter, i.e., 1/lambda in TRADES (default: 6.0)')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--save-model', '-sm', default=90, type=int, metavar='N',
                        help='after which epoch to start saving best model')
    # For PGD attack on val set
    parser.add_argument('--attack-step-size', default=2/255,type=float,
                        help='perturb step size')
    parser.add_argument('--attack-epsilon', default=8/255,type=float,
                        help='perturbation')
    parser.add_argument('--attack-num-steps', default=20,type=int,
                        help='perturb number of steps')

    # Loss and architecture parameters
    parser.add_argument('--adv-loss', choices=[
                            'WEAT_adv','WEAT_nat'], default='WEAT_nat', help='type of adversarial loss to use')
    parser.add_argument('--architecture', choices=[
                            'PreActResNet18', 'Resnet18'
                        ], default='Resnet18', help='model architecture (default: Resnet18)')
    parser.add_argument('--dataset', choices=[
                            'CIFAR-10', 'CIFAR-100', 'SVHN', 'tiny_imagenet'
                        ], default='CIFAR-10', help='dataset to use (default: CIFAR-10)')
    return parser
