import utils
import torch as ch
from autoattack import AutoAttack


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        logits, _ = self.model(x)
        return logits


def get_all_cifar10_data(constants, is_test=True):
    ds = constants.get_dataset()
    if is_test:
        _, data_loader = ds.make_loaders(batch_size=16, only_val=True, workers=8, data_aug=False)
    else:
        data_loader, _ = ds.make_loaders(batch_size=16, workers=8, data_aug=False)
    X, Y = [], []
    for (x, y) in data_loader:
        X.append(x)
        Y.append(y)
    X, Y = ch.cat(X), ch.cat(Y)
    return (X, Y)


if __name__ == "__main__":
    epsilon = 8/255
    constants = utils.CIFAR10()
    ds = constants.get_dataset()
    model = constants.get_model("/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best", "vgg19")
    wrapped_model = ModelWrapper(model)
    images, labels  = get_all_cifar10_data(constants)
    adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, plus=False)
    x_adv = adversary.run_standard_evaluation(images, labels, bs=128)
