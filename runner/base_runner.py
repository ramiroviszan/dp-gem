class BaseRunner:

    def __init__(self, exp_name, exp_path):
        self.exp_name = exp_name
        self.exp_path = exp_path

    def run_module(self, module_desc, exp_info = None):
        interations = module_desc.get('iterations', 0)
        trials = module_desc.get('trials', [])
        mode = module_desc.get('mode', 'all')
        submodules = module_desc.get('submodules', None)

        if len(trials) == 0:
            if submodules == None:
                params = list(module_desc.values())[2:]
            else:
                params = list(module_desc.values())[2:-1]

            self.run_interation_based(iterations, mode, submodules, params, exp_info)
        else:
            if submodules == None:
                params = list(module_desc.values())[3:]
            else:
                params = list(module_desc.values())[3:-1]
            self.run_trial_based(interations, trials, mode, submodules, params, exp_info)

        
    def run_interation_based(self, interations, mode, submodules, params, exp_info):
        for i in range(0, interations):
            if mode == "all" or mode == "main_only":
                if exp_info == None:
                    run_info = (self.exp_path, {}, i)
                else:
                    run_info = exp_info
                run_params = copy.deepcopy(params)
                module = self.hot_new(run_info, *run_params)
                module.run()
            
            if submodules != None and (mode == "all" or mode == "submodules_only"):
                parent_intertion_desc = (self.exp_name, {}, i)
                self.run_submodules(submodules, parent_intertion_desc)

    def run_trial_based(self, interations, trials, mode, submodules, params, exp_info):
        if mode == "all" or mode == "main_only":
            if exp_info == None:
                run_info = self.exp_path
            else:
                run_info = exp_info
            run_params = copy.deepcopy(params)
            module = self.hot_new(run_info, *run_params)

        for trial in trials:
            if module != None:
                for i in range(0, interations):
                    module.run(trial, i)
            
            if submodules != None and (mode == "all" or mode == "submodules_only"):
                for i in range(0, interations):
                    parent_trial_iteration_desc = (self.exp_name, trial, i)
                    self.run_submodules(submodules, parent_trial_iteration_desc)
    
    def run_submodules(self, submodules, parent_run_desc):
        for submodule in submodules:
            self.run_module(submodule, parent_run_desc)
    
    def hot_new(self, exp_info, module_name, class_name, params):
        module = __import__(module_name, globals(), locals(), [class_name], 0)
        class_ = getattr(module, class_name)
        instance = class_(exp_info, *params.values())
        return instance

