from copy import deepcopy


def get_result_iterator(object_attentions, viewpoint_attentions, tok):
    for i in range(len(object_attentions)):
        zipped_attentions = zip(object_attentions[i][0], object_attentions[i][1], viewpoint_attentions[i])
        for info, attn, v_attn in zipped_attentions:
            info = deepcopy(info)
            info["objects"]["names"] = [tok.decode_sentence(x) for x in info["objects"]["names"]]
            info["objects"]["names"] = [
                " ".join([w for w in x.split(" ") if w not in ["<BOS>", "<EOS>", "#"]])
                for x in info["objects"]["names"]
            ]
            attn = attn.squeeze().cpu()
            v_attn = v_attn.cpu().detach()

            candidates = [x["viewpointId"] for x in info["candidate"]]

            print(attn[: len(candidates) + 1].shape, v_attn[: len(candidates) + 1].shape)

            yield info, attn[: len(candidates) + 1], v_attn[: len(candidates) + 1], candidates
