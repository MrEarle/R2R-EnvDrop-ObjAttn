import json


def get_id_2_instr(path):
    with open(path) as ar:
        instrs = json.load(ar)

    id_2_instr = {}
    for instr in instrs:
        id_2_instr[instr["path_id"]] = instr

    return id_2_instr


def get_instruction_function(path="./tasks/R2R/data/R2R_val_unseen.json"):
    id_2_instr = get_id_2_instr(path)

    def get_instruction(path_id_with_index: str, include_all=False):
        path_id, index = [int(i) for i in path_id_with_index.split("_")]

        path = id_2_instr[path_id]
        instr = path["instructions"][index]

        if include_all:
            return instr, path
        return instr

    return get_instruction
