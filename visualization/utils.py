from copy import deepcopy


def get_result_iterator(object_attentions, viewpoint_attentions, tok):
    for i in range(len(object_attentions)):
        zipped_attentions = zip(object_attentions[i][0], object_attentions[i][1], viewpoint_attentions[i])
        for info, attn, v_attn in zipped_attentions:
            info = deepcopy(info)
            # info["objects"]["names"] = [tok.decode_sentence(x) for x in info["objects"]["names"]]
            # info["objects"]["names"] = [
            #     " ".join([w for w in x.split(" ") if w not in ["<BOS>", "<EOS>", "#"]])
            #     for x in info["objects"]["names"]
            # ]
            attn = attn.squeeze().cpu()
            v_attn = v_attn.cpu().detach()

            candidates = [x["viewpointId"] for x in info["candidate"]]

            print(attn[: len(candidates) + 1].shape, v_attn[: len(candidates) + 1].shape)

            yield info, attn[: len(candidates) + 1], v_attn[: len(candidates) + 1], candidates


def get_traj_result_iterator(object_attentions, viewpoint_attentions, env_idx=0):
    num_envs = len(object_attentions[0][0])
    num_times = len(object_attentions)
    time_idx = 0

    while True:
        if time_idx >= num_times:
            env_idx += 1
        if env_idx >= num_envs:
            raise StopIteration()

        info = object_attentions[time_idx][0][env_idx]
        attn = object_attentions[time_idx][1][env_idx]
        v_attn = viewpoint_attentions[time_idx][env_idx]

        attn = attn.squeeze().cpu()
        v_attn = v_attn.cpu().detach()

        candidates = [x["viewpointId"] for x in info["candidate"]]

        y_res = yield info, attn[: len(candidates) + 1], v_attn[: len(candidates) + 1], candidates

        if y_res is not None and y_res != env_idx:
            env_idx = y_res
            time_idx = 0
        elif y_res == -1:
            time_idx = 0
        else:
            time_idx += 1
