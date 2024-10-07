import argparse
from itertools import cycle

from matplotlib import pyplot as plt

def main(args):
    with open(args.loss_file, "r") as f:
        losses = [ loss_line.rstrip() for loss_line in f ]

    training_epochs, training_losses = [], []
    validation_epochs, validation_losses = [], []

    if args.iters_per_epoch == -1:
        iters_per_epoch = int(losses[0].split(": ")[1])
        losses = losses[1:]
    else:
        iters_per_epoch = args.iters_per_epoch

    curr_epoch = 0
    for i_line, line in enumerate(losses):

        if line.startswith("Epoch: "):
            epoch = int(line.split("Epoch: ")[1].split(",")[0])
            iter = int(line.split("Iter: ")[1].split(",")[0])
            loss_line = losses[i_line + 1]
            tot_loss = float(loss_line.split("total=")[1].split(" ")[0])
            training_epochs.append(epoch + iter / iters_per_epoch)
            training_losses.append(tot_loss)
            curr_epoch = epoch

        if line == "== Validation Loop ==":
            loss_line = losses[i_line + 2]
            tot_loss = float(loss_line.split("total=")[1].split(" ")[0])
            epoch = curr_epoch + 1
            validation_epochs.append(epoch)
            validation_losses.append(tot_loss)

    _, ax = plt.subplots()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    ax.plot(training_epochs, training_losses, label="train - total", c=next(colors))
    ax.plot(validation_epochs, validation_losses, label="valid - total", c=next(colors))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 1.2 * max(validation_losses))
    plt.legend()

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)

    parser.add_argument("--iters_per_epoch", type=int, default=-1)
    parser.add_argument("--legacy", action="store_true")

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

