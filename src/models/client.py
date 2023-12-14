import time
from torch.utils.data import DataLoader


class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """
    def __init__(self, cid, group, available_probability, train_data, test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker
        self.available_probability = available_probability

        self.train_data = train_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self.last_activate_round = 0
        self.activate_prob = 0

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def local_train(self, strategy, prev_model, baseline_param, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """
        if strategy == "moon":
            local_solution, worker_stats = self.worker.local_train_moon(self.train_dataloader, prev_model, baseline_param,  **kwargs)
        else: # fedavg or fedprox
            local_solution, worker_stats = self.worker.local_train(self.train_dataloader, baseline_param,  **kwargs)

        stats = {'id': self.cid}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats
    
    def evaluate_train(self, **kwargs):
        return self.worker.evaluate(self.train_dataloader, **kwargs)

    def evaluate_test(self, **kwargs):
        return self.worker.evaluate(self.test_dataloader, **kwargs)

    def getRecentActivity(self, cur_round):     # When available, calculate its own activity
        if cur_round == 0:
            self.activate_prob = 1.0
            return self.activate_prob
        if self.last_activate_round == 0:
            self.last_activate_round = cur_round
            self.activate_prob = 1/cur_round
            return self.activate_prob
        self.activate_prob = (self.last_activate_round * self.activate_prob + 1) / cur_round
        self.last_activate_round = cur_round
        return self.activate_prob

    def getActivity(self):
        return self.activate_prob