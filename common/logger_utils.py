

import wandb
from common.trials_utils import flat_trial


def get_logger(logger_name, exp_name, parent_info = None):

    end_name = ""
    if parent_info != None:
        flat = flat_trial(parent_info)
        end_name = f"_{flat}"
    
    logger = wandb.init(
        name=f"{logger_name}{end_name}", 
        project=f"{exp_name}",
        group=f"{logger_name}",
        reinit=True,
        config = {}
    )

    return logger