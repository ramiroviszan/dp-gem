

import wandb
from common.trials_utils import flat_trial


def get_logger(logger_name, exp_name, parent_info = None, group_name=None):

    end_name = ""
    if parent_info != None: 
        flat = flat_trial(parent_info)
        if len(flat) > 0:
            end_name = f"_{flat}"

    if group_name == None:
        group_name = logger_name

    logger = wandb.init(
        name=f"{logger_name}{end_name}", 
        project=f"{exp_name}",
        group=f"{group_name}",
        reinit=True,
        config = {}
    )

    return logger