import os
from collections import OrderedDict, defaultdict

import h5py
import torch
from agent import Seq2SeqAgent
from r2r_src.train import train_val, TRAIN_VOCAB, Evaluation, R2RBatch, Tokenizer, read_vocab, setup
from param import args

log_dir = args.log_dir
FEATURE_FILE = "/home/mrearle/storage/img_features/ResNet-152-imagenet.hdf5"


def load_args(
    base_name: str,
    max_obj_number: int = 20,
    obj_aux_task: bool = True,
    include_objs: bool = True,
    reduced_envs: bool = True,
    logging_vis: bool = False,
    dataset: str = "R2R",
):
    args.accumulate_grad = False
    args.angle_feat_size = 128
    args.attn = "soft"
    args.featdropout = 0.3
    args.feedback = "sample"
    args.gamma = 0.9
    args.ignoreid = -100
    args.maxDecode = 120
    args.maxInput = 80
    args.sub_out = "max"
    args.submit = False
    args.train = "validlistener"
    args.valid = False

    args.obj_attn_type = "connection"
    args.obj_aux_task_weight = 0.1
    args.obj_label_task = False
    args.obj_label_task_weight = 0.1
    args.include_objs_lstm = False
    args.maxAction = 35

    args.include_objs = include_objs
    args.max_obj_number = max_obj_number
    args.obj_aux_task = obj_aux_task
    args.reduced_envs = reduced_envs
    args.dataset = dataset
    args.logging_vis = logging_vis

    args.name = base_name
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
    args.log_dir = f"snap/{args.save_dir}"
    args.load = f"snap/{args.save_dir}/state_dict/best_val_unseen"
    # args.load = "snap/obj/good/craft_obj(32)_aux(0.1)_reduced/state_dict/Iter_100000"

    print(args)


def load_envs():
    setup()

    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_h5 = h5py.File(FEATURE_FILE, "r")
    obj_feat_h5 = h5py.File(args.OBJECT_FEATURES, "r")

    featurized_scans = set(feat_h5.keys())

    train_env = R2RBatch(
        feature_store=feat_h5, object_feat_store=obj_feat_h5, batch_size=args.batchSize, splits=["train"], tokenizer=tok
    )

    val_env_names = ["val_unseen"]

    val_envs = OrderedDict(
        (
            (
                split,
                (
                    R2RBatch(
                        feature_store=feat_h5,
                        object_feat_store=obj_feat_h5,
                        batch_size=args.batchSize,
                        splits=[split],
                        tokenizer=tok,
                    ),
                    Evaluation([split], featurized_scans, tok),
                ),
            )
            for split in val_env_names
        )
    )

    return train_env, val_envs, tok


def setup_agent(attach_hooks=False):
    print("Setup agent")
    agent = train_val()
    # agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    object_attentions = []
    viewpoint_attentions = []
    if attach_hooks:

        def obj_attention_hook(model, input, output):
            if model.traj_info is None:
                return None

            _, attentions, _ = output
            traj_info = model.traj_info
            attns = attentions.detach().cpu()
            object_attentions.append((traj_info, attns))
            model.traj_info = None

        def view_attention_hook(model, input, output):
            _, _, attn, _, _ = output
            viewpoint_attentions.append(attn.detach().cpu())

        agent.decoder.connectionwise_obj_attn.register_forward_hook(obj_attention_hook)
        agent.decoder.register_forward_hook(view_attention_hook)

    return agent, object_attentions, viewpoint_attentions


def run_agent(agent, val_envs):
    print("Run agent")

    with torch.no_grad():
        for env_name, (env, evaluator) in val_envs.items():
            agent.logs = defaultdict(list)
            agent.env = env

            iters = None
            agent.test(use_dropout=False, feedback="argmax", iters=1)
            agent_result = agent.get_results()

            break

    return agent_result


def get_agent_results(agent: Seq2SeqAgent, val_envs, result_path):
    print("Run agent")

    with torch.no_grad():
        for env_name, (env, evaluator) in val_envs.items():
            print(env_name)
            agent.logs = defaultdict(list)
            agent.env = env

            iters = None
            agent.test(use_dropout=False, feedback="argmax", iters=iters, result_path=result_path)

    return agent.get_results()


def setup_and_run_agent(
    base_name: str,
    max_obj_number: int = 20,
    obj_aux_task: bool = True,
    include_objs: bool = True,
    reduced_envs: bool = True,
    dataset: str = "R2R",
    logging_vis: bool = False,
    attach_hooks=False,
    get_results=False,
    result_path=None,
):
    print("Parsing args")
    load_args(
        base_name,
        max_obj_number=max_obj_number,
        obj_aux_task=obj_aux_task,
        include_objs=include_objs,
        reduced_envs=reduced_envs,
        dataset=dataset,
        logging_vis=logging_vis,
    )

    agent, obj_attns, view_attns = setup_agent(attach_hooks=attach_hooks and not get_results)

    print("Loading envs")
    _, val_envs, _ = load_envs()

    if get_results:
        agent_result = get_agent_results(agent, val_envs, result_path)
    else:
        agent_result = run_agent(agent, val_envs)

    return agent_result, obj_attns, view_attns
