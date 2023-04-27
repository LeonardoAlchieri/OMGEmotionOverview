import datetime
import sys

from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def printoneline(*argv):
    s = ""
    for arg in argv:
        s += str(arg) + " "
    s = s[:-1]
    sys.stdout.write("\r" + s)
    sys.stdout.flush()


def dt():
    return datetime.datetime.now().strftime("%H:%M:%S")


def train(
    train_loader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    epoch: int,
    device: str = "cuda",
    reshape_mode: int = 1,
    max_grad: float = 20.0,
    print_freq: int = 3,
):
    model.train()

    train_loss = 0
    batch_idx = 0

    for i, (inputs, targets, _) in tqdm(
        enumerate(train_loader), desc="Training batch", total=len(train_loader)
    ):
        inputs: Tensor
        targets: Tensor

        optimizer.zero_grad()

        inputs, targets = inputs.to(device, non_blocking=False), targets.to(
            device, non_blocking=False
        )

        # NOTE: added for way resnet wants shape
        if reshape_mode == 1:
            inputs = inputs.reshape(
                inputs.shape[0], -1, 3, inputs.shape[-2], inputs.shape[-1]
            )
        elif reshape_mode == 2:
            inputs = inputs.view((-1, 3) + inputs.size()[-2:])
        else:
            raise ValueError("reshape_mode must be 1 or 2. Got %d" % reshape_mode)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # tsn uses clipping gradient
        if max_grad is not None:
            total_norm = clip_grad_norm(model.parameters(), max_grad)
            if total_norm > max_grad:
                print(
                    "clippling gradient: {} with coef {}".format(
                        total_norm, max_grad / total_norm
                    )
                )

        train_loss += loss.data.item()

        if i % print_freq == 0:
            printoneline(
                dt(), "Epoch=%d Loss=%.4f \n" % (epoch, train_loss / (batch_idx + 1))
            )
        batch_idx += 1
