import argparse
import os
import json
import torch

SMALL_SPLITS = {
    "train": [
        "Uxmj2M2itWa",
        "82sE5b5pLXE",
        "2n8kARJN3HM",
        "1pXnuDYAj8r",
        "VLzqgDo317F",
        "p5wJjkQkbXX",
        "r1Q1Z4BcV1o",
        "HxpKQynjfin",
        "PuKPg4mmafe",
        "cV4RVeZvu5T",
        "PX4nDJXEHrG",
        "VFuaQ6m2Qom",
        "JF19kD82Mey",
        "sT4fr6TAbpF",
        "E9uDoFAP3SH",
        "XcA2TqTSSAj",
        "8WUmhLawc2A",
        "EDJbREhghzL",
        "1LXtFkjw3qL",
        "pRbA3pwrgk9",
        "gZ6f7yhEvPG",
    ],
    "val_seen": [
        "Uxmj2M2itWa",
        "82sE5b5pLXE",
        "2n8kARJN3HM",
        "1pXnuDYAj8r",
        "VLzqgDo317F",
        "p5wJjkQkbXX",
        "r1Q1Z4BcV1o",
        "PuKPg4mmafe",
        "cV4RVeZvu5T",
        "PX4nDJXEHrG",
        "VFuaQ6m2Qom",
        "JF19kD82Mey",
        "sT4fr6TAbpF",
        "E9uDoFAP3SH",
        "XcA2TqTSSAj",
        "8WUmhLawc2A",
        "1LXtFkjw3qL",
        "EDJbREhghzL",
        "pRbA3pwrgk9",
        "gZ6f7yhEvPG",
    ],
    "val_unseen": [
        "2azQ1b91cZZ",
        "QUCTc6BB5sX",
        "zsNo4HB9uLZ",
        "oLBMNvg9in8",
        "8194nk5LbLH",
        "EU6Fwq7SyZv",
        "x8F5xyUWy9e",
        "Z6MFQCViBuw",
        "X7HyMhZNoso",
        "pLe4wQe7qrG",
        "TbHJrupSAjP",
    ],
}


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="", allow_abbrev=False)

        # General
        self.parser.add_argument("--iters", type=int, default=100000)
        self.parser.add_argument("--name", type=str, default="default")
        self.parser.add_argument("--train", type=str, default="speaker")

        # Data preparation
        self.parser.add_argument("--maxInput", type=int, default=80, help="max input instruction")
        self.parser.add_argument("--maxDecode", type=int, default=120, help="max input instruction")
        self.parser.add_argument("--maxAction", type=int, default=20, help="Max Action sequence")
        self.parser.add_argument("--batchSize", type=int, default=64)
        self.parser.add_argument("--ignoreid", type=int, default=-100)
        self.parser.add_argument("--feature_size", type=int, default=2048)
        self.parser.add_argument("--loadOptim", action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # More Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument(
            "--zeroInit",
            dest="zero_init",
            action="store_const",
            default=False,
            const=True,
        )
        self.parser.add_argument("--mlWeight", dest="ml_weight", type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest="teacher_weight", type=float, default=1.0)
        self.parser.add_argument(
            "--accumulateGrad",
            dest="accumulate_grad",
            action="store_const",
            default=False,
            const=True,
        )
        self.parser.add_argument("--features", type=str, default="imagenet")

        # Env Dropout Param
        self.parser.add_argument("--featdropout", type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument(
            "--selfTrain",
            dest="self_train",
            action="store_const",
            default=False,
            const=True,
        )

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument(
            "--paramSearch",
            dest="param_search",
            action="store_const",
            default=False,
            const=True,
        )
        self.parser.add_argument("--submit", action="store_const", default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument("--optim", type=str, default="rms")  # rms, adam
        self.parser.add_argument("--lr", type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument("--decay", dest="weight_decay", type=float, default=0.0)
        self.parser.add_argument("--dropout", type=float, default=0.5)
        self.parser.add_argument(
            "--feedback",
            type=str,
            default="sample",
            help="How to choose next position, one of ``teacher``, ``sample`` and ``argmax``",
        )
        self.parser.add_argument(
            "--teacher",
            type=str,
            default="final",
            help="How to get supervision. one of ``next`` and ``final`` ",
        )
        self.parser.add_argument("--epsilon", type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument("--rnnDim", dest="rnn_dim", type=int, default=512)
        self.parser.add_argument("--wemb", type=int, default=256)
        self.parser.add_argument("--aemb", type=int, default=64)
        self.parser.add_argument("--proj", type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument(
            "--candidate",
            dest="candidate_mask",
            action="store_const",
            default=False,
            const=True,
        )

        self.parser.add_argument("--bidir", type=bool, default=True)  # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")  # soft, mono, shift, dis_shift

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument(
            "--normalize",
            dest="normalize_loss",
            default="total",
            type=str,
            help="batch or total",
        )

        # Obj Args

        self.parser.add_argument("--obj_attn_type", type=str, default="connection")
        self.parser.add_argument("--max_obj_number", type=int, default=20)
        self.parser.add_argument("--obj_aux_task", action="store_true")
        self.parser.add_argument("--obj_aux_task_weight", type=float, default=0.1)
        self.parser.add_argument("--obj_label_task", action="store_true")
        self.parser.add_argument("--obj_label_task_weight", type=float, default=0.1)
        self.parser.add_argument("--dataset", type=str, default="R2R")
        self.parser.add_argument("--include_objs", action="store_true")
        self.parser.add_argument("--include_objs_lstm", action="store_true")
        self.parser.add_argument("--reduced_envs", action="store_true")
        self.parser.add_argument("--buffer_objs", action="store_true")

        self.args, _ = self.parser.parse_known_args()

        if self.args.optim == "rms":
            print("Optimizer: Using RMSProp", flush=True)
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == "adam":
            print("Optimizer: Using Adam", flush=True)
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == "sgd":
            print("Optimizer: sgd", flush=True)
            self.args.optimizer = torch.optim.SGD
        else:
            assert False


param = Param()
args = param.args
args.TRAIN_VOCAB = "tasks/R2R/data/train_vocab.txt"
args.TRAINVAL_VOCAB = "tasks/R2R/data/trainval_vocab.txt"

args.IMAGENET_FEATURES = "img_features/ResNet-152-imagenet.tsv"
args.CANDIDATE_FEATURES = "img_features/ResNet-152-candidate.tsv"

# * ============= My Args ==============
# args.OBJECT_FEATURES = "/home/mrearle/storage/img_features/ResNet-152-imagenet-conv.hdf5"
# args.OBJECT_PROPOSALS = "/home/mrearle/storage/img_features/viewpoint_objects.h5"
args.OBJECT_FEATURES = "/workspace1/mrearle/object_features_filtered.hdf5"
args.OBJECT_CLASS_FILE = "/workspace1/mrearle/object_classes.json"

with open(args.OBJECT_CLASS_FILE, "r") as f:
    objs = json.load(f)
    args.num_obj_classes = len(objs)
args.views = 36
args.logging_vis = False
args.reduced_env_ids = set()

for env_ids in SMALL_SPLITS.values():
    args.reduced_env_ids.update(env_ids)

# args.obj_attn_type = "connection"
# args.max_obj_number = 20
# args.obj_aux_task = False
# args.obj_aux_task_weight = 0.1
# args.dataset = "R2R"

experiment = []
if args.dataset.upper() != "R2R":
    experiment.append(args.dataset)
if args.max_obj_number != 20:
    experiment.append(f"obj({args.max_obj_number})")
if args.obj_aux_task:
    experiment.append(f"aux({args.obj_aux_task_weight})")
if args.reduced_envs:
    experiment.append(f"reduced")

args.experiment = "_".join(experiment) or "default"

args.save_dir = os.path.join(args.name, args.experiment)
# * ====================================

args.features_fast = "img_features/ResNet-152-imagenet-fast.tsv"
args.log_dir = "snap/%s" % args.save_dir

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

result_dir = f"results/{args.save_dir}"
if not os.path.exists(result_dir):
    os.makedirs(f"results/{args.save_dir}")
DEBUG_FILE = open(os.path.join("snap", args.save_dir, "debug.log"), "w")
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {args.device}", flush=True)
print(torch.cuda.get_device_properties(args.device), flush=True)

print(f"\n\n\tTraining model {args.name} in experiment {args.experiment}\n\n", flush=True)
