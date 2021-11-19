from torch.optim import Optimizer


class MultiOptimiser(Optimizer):
    """
    Wrapper class for multiple optimisers.
    Runs the optimisers sequentially.
    """
    def __init__(self, use_run_list=True, **optimisers):
        self.optimisers = optimisers
        self.use_run_list = use_run_list
        # List of optimisers to run in this step
        self.run_list = set()

    def add_optimisers(self, **optimisers):
        self.optimisers.update(**optimisers)

    def add_run_opt(self, name):
        self.run_list.add(name)

    def zero_grad(self):
        for name, optimiser in self.optimisers.items():
            optimiser.zero_grad()

        self.run_list = set()

    def step(self):
        if self.use_run_list:
            for name in self.run_list:
                self.optimisers[name].step()
        else:
            for name, optimiser in self.optimisers.items():
                optimiser.step()

    @property
    def param_groups(self):
        # Merge the param_groups lists from all optimisers
        param_groups = []
        for _, opt in self.optimisers.items():
            param_groups = param_groups + opt.param_groups
        return param_groups

    def state_dict(self):
        state_dict = {}
        for name, opt in self.optimisers.items():
            state_dict[name] = opt.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        for name, state in state_dict.items():
            self.optimisers[name].load_state_dict(state)

    @property
    def state(self):
        """
        For Lightning to load state
        """
        states = {}
        for name, opt in self.optimisers.items():
            for s, state in opt.state.items():
                states[f'{name}_{s}'] = state

        return states
