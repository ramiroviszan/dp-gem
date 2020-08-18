import sys
import os
import time
import copy
import random
from os import listdir

class BaseRunner:

    def __init__(self, exp_name, exp_path):
        self.exp_name = exp_name
        self.exp_path = exp_path

    def run_module(self, module_desc, exp_info = None):
        skipStart = 0
        skipEnd = 0
        
        run_type = 'single'
        if 'run_type' in module_desc.keys():
            skipStart = skipStart + 1
            run_type = module_desc.get('type')

        if 'trials' in module_desc.keys():
            trials = module_desc.get('trials', [])
            skipStart = skipStart + 1
            if len(trials) == 0:
                print("Error")
                return 

        mode = 'all'
        if 'mode' in module_desc.keys():
            skipStart = skipStart + 1
            mode = module_desc.get('mode')
        
        if 'submodules' in module_desc.keys():
            skipEnd = skipEnd - 1
            submodules = module_desc.get('submodules')

        params = list(module_desc.values())[skipStart:SkipEnd]
        
        if run_type == "single":
            self.run_single(params, mode, exp_info, submodules)
        elif run_type == "trials":
            self.run_trials_based(params, mode, exp_info, submodules)
        
    def run_single(self, mode, submodules, params, exp_info):
        
        creation_info = (self.exp_path, {}) if exp_info == None else exp_info
        if mode == "all" or mode == "main_only":
            creation_params = copy.deepcopy(params)
            module = self.hot_new(creation_info, *creation_params)
            module.run()
        
        if submodules != None and (mode == "all" or mode == "submodules_only"):
            self.run_submodules(submodules, creation_info)

    def run_trial_based(self, iterations, trials, mode, submodules, params, exp_info):
        if mode == "all" or mode == "main_only":
            creation_info = (self.exp_path, {}) if exp_info == None else exp_info
            creation_params = copy.deepcopy(params)
            module = self.hot_new(creation_info, *creation_params)

        for trial in trials:
            if module != None:
                module.run(trial)
            
            if submodules != None and (mode == "all" or mode == "submodules_only"):
                parent_info = (self.exp_name, trial)
                self.run_submodules(submodules, parent_info)
    
    def run_submodules(self, submodules, parent_info):
        for submodule in submodules:
            self.run_module(submodule, parent_info)
    
    def hot_new(self, exp_info, module_name, class_name, params):
        module = __import__(module_name, globals(), locals(), [class_name], 0)
        class_ = getattr(module, class_name)
        instance = class_(exp_info, *params.values())
        return instance

