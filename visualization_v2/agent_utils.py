from agent import Seq2SeqAgent
from collections import defaultdict
import os
import h5py
from collections import OrderedDict
from train import setup, read_vocab, Tokenizer, TRAIN_VOCAB, read_img_features, R2RBatch, Evaluation
from param import args

log_dir = args.log_dir
FEATURE_FILE = "/home/mrearle/storage/img_features/ResNet-152-imagenet.hdf5"


def load_args(
    agent_name: str,
    max_obj_number: int = 32,
    obj_aux_task: bool = True,
    include_objs: bool = True,
    reduced_envs: bool = True,
    dataset: str = "R2R",
):
    # Model args
    args.name = "py_default"
    args.attn = "soft"
    args.train = "validlistener"
    args.angle_feat_size = 128
    args.accumulateGrad = True
    args.featdropout = 0.4
    args.subout = "max"
    args.optim = "rms"
    args.lr = 1e-4
    args.iters = 10
    args.maxAction = 35

    # Obj attn args
    args.obj_attn_type = "connection"
    args.max_obj_number = max_obj_number
    args.obj_aux_task = obj_aux_task
    args.obj_aux_task_weight = 0.1
    args.include_objs = include_objs
    args.include_objs_lstm = False
    args.reduced_envs = reduced_envs
    args.buffer_objs = False
    args.dataset = dataset

    # Required for visualization
    args.logging_vis = True


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


def setup_agent(train_env, tok, load):
    object_attentions = []

    def obj_attention_hook(model, input, output):
        if model.traj_info is None:
            return None

        _, attentions, _ = output
        traj_info = model.traj_info
        attns = attentions.detach().cpu()
        object_attentions.append((traj_info, attns))
        model.traj_info = None

    viewpoint_attentions = []

    def view_attention_hook(model, input, output):
        _, _, attn, _, _ = output
        viewpoint_attentions.append(attn.detach().cpu())

    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    agent.decoder.connectionwise_obj_attn.register_forward_hook(obj_attention_hook)
    agent.decoder.register_forward_hook(view_attention_hook)

    args.load = load
    print(
        "Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load),
        flush=True,
    )

    return agent, object_attentions, viewpoint_attentions


def run_agent(agent, val_envs):
    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback="argmax", iters=1)
        agent_result = agent.get_results()

        break

    return agent_result


def setup_and_run_agent(
    base_name: str,
    max_obj_number: int = 32,
    obj_aux_task: bool = True,
    include_objs: bool = True,
    reduced_envs: bool = True,
    dataset: str = "R2R",
):
    load_args(
        max_obj_number=max_obj_number,
        obj_aux_task=obj_aux_task,
        include_objs=include_objs,
        reduced_envs=reduced_envs,
        dataset=dataset,
    )


    train_env, val_envs, tok = load_envs()

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
    args.save_dir = os.path.join(base_name, args.experiment)
    load = f"snap/{base_name}/{args.experiment}/state_dict/best_val_unseen"
    
    agent, obj_attns, view_attns = setup_agent(train_env, tok, load)

    agent_result = run_agent(agent, val_envs)

    return agent_result, obj_attns, view_attns
