def get_traj_result_iterator(object_attentions, viewpoint_attentions, env_idx=0):
    num_envs = len(object_attentions[0][0])
    num_times = len(object_attentions)
    time_idx = 0

    while True:
        if time_idx >= num_times:
            env_idx += 1
            time_idx = 0
        if env_idx >= num_envs:
            raise StopIteration()

        info = object_attentions[time_idx][0][env_idx]
        attn = object_attentions[time_idx][1][env_idx].squeeze()
        v_attn = viewpoint_attentions[time_idx][env_idx]

        candidates = [x["viewpointId"] for x in info["candidate"]]

        # +1 because of stop action
        y_res = yield info, attn[: len(candidates) + 1], v_attn[: len(candidates) + 1], candidates

        if y_res is not None and y_res != env_idx:
            env_idx = y_res
            time_idx = 0
        elif y_res == -1:
            time_idx = 0
        else:
            time_idx += 1


def map_result(env_i, agent_results, hook_results, get_instruction):
    info, attn, viewpoint_attn, candidates = hook_results
    instruction = get_instruction(info["instr_id"])

    # Get Object Names
    if len(info["objects"]["names"]) == 0:
        obj_names = []
    else:
        obj_names = [info["objects"]["names"][int(i)] for i in info["obj_sample"]]

    # Calculate number of included objects and filter <pad> objects
    num_obj = sum(info["mask"]).cpu().numpy()
    obj_names = obj_names[:num_obj]

    # Get attention over non <pad> objects
    attn = attn[:, :num_obj]

    # Add stop action
    viewpoint_indices = candidates + ["STOP"]

    # Get agent trajectory (not ground truth)
    trajectory = []
    prev = None
    for traj, _, _ in agent_results[env_i]["trajectory"]:
        if traj != prev:
            trajectory.append(traj)
        prev = traj

    return {
        "hook_info": info,
        "object_attn": attn,
        "view_attn": viewpoint_attn,
        "candidate_actions": candidates,
        "object_names": obj_names,
        "num_objs": num_obj,
        "viewpoint_names": viewpoint_indices,
        "agent_trajectory": trajectory,
        "instruction": instruction,
    }
