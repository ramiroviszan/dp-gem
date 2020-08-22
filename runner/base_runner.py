import sys
import os
import time
import copy
import random
from os import listdir

import wandb

from common.trials_utils import flat_trial

class BaseRunner:

    def __init__(self, module_name, exp_name, exp_path):
        self.module_name = module_name
        self.exp_name = exp_name
        self.exp_path = exp_path

    def run_module(self, module_desc, parent_info=None):
        skip = module_desc.get('skip', 0)
        module_name = module_desc.get('module_name')
        class_name = module_desc.get('class_name')
        use_wandb = module_desc.get('wandb', False)
        build_params = module_desc.get('build_params', {})
        creation_params = [module_name, class_name, build_params] 
        trials_params = module_desc.get('trials_params', [{}])
        mode = module_desc.get('mode', 'all')
        submodules = module_desc.get('submodules', None)
        
        if skip == 0:
            if use_wandb:
                end_name = ""
                if parent_info != None:
                    flat = flat_trial(parent_info)
                    end_name = f"_{flat}"
                
                logger = wandb.init(
                    name=f"{self.exp_name}_{self.module_name}{end_name}", 
                    project="dp-project",
                    group=self.exp_name,
                    reinit=True
                )
                with logger:
                    self._run(creation_params, trials_params, mode, parent_info, submodules, logger)
            else:
                self._run(creation_params, trials_params, mode, parent_info, submodules)
        else:
            print("\n\n\nSkipping", self.module_name)
            


    def _run(self, creation_params, trials, mode, parent_info, submodules, logger=None):
        if mode == "all" or mode == "main_only":
            creation_info = (self.exp_path, {}) if parent_info == None else (self.exp_path, parent_info)
            creation_params_copy = copy.deepcopy(creation_params)
            module = self.hot_new(creation_info, logger, *creation_params_copy)

        for trial_params in trials:
            if module != None:
                trial_copy = copy.deepcopy(trial_params)
                module.run(trial=trial_copy)

            if submodules != None and (mode == "all" or mode == "submodules_only"):
                self.run_submodules(submodules, trial_params)

    def run_submodules(self, submodules, parent_info):
        for submodule_name, submodule in submodules.items():
            sub_runner = BaseRunner(submodule_name, self.exp_name, self.exp_path)    
            sub_runner.run_module(submodule, parent_info)

    def hot_new(self, creation_info, logger, module_name, class_name, params):
        module = __import__(module_name, globals(), locals(), [class_name], 0)
        class_ = getattr(module, class_name)
        instance = class_(creation_info, logger, *params.values())
        return instance
