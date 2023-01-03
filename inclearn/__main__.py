import matplotlib
import sys
sys.path.append("D:\\Codings_piang\\deep_learning\\Piang_inc_222")
import os
from inclearn import parser
from inclearn.train import train
import torch
matplotlib.use('Agg')


def main():
    # torch.cuda.empty_cache()
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.cuda.set_device(0)

    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.

    if args["seed_range"] is not None:
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))

        print("Seed range", args["seed"])

    for _ in train(args):  # `train` is a generator in order to be used with hyperfind.
        pass


if __name__ == "__main__":
    main()
