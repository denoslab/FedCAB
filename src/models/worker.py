from src.utils.torch_utils import get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch

criterion = nn.CrossEntropyLoss().cuda()  # : added .cuda()
cos_criterion=torch.nn.CosineSimilarity(dim=-1)
KLD = torch.nn.KLDivLoss(reduction='sum')

class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.local_step = options['local_step']
        self.gpu = options['gpu'] if 'gpu' in options else False

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, _ in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def local_train(self, train_dataloader, baseline_param,  **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        global_param = self.model.layer.weight.clone().detach()  # : Fetch global model
        import copy
        global_model = copy.deepcopy(self.model)

        self.model.train()
        train_loss = train_acc = train_total = 0
        # local_counter = 0

        for _ in range(self.local_step): # train K epochs.
            for (x, y) in train_dataloader:

                # if local_counter >= self.local_step:
                #     break
                # local_counter+=1

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                # for MOON
                global_pred = global_model(x)
                posi = cos_criterion(pred, global_pred)
                logits = posi.reshape(-1, 1)

                loss = criterion(pred, y)

                # Add FedProx term
                lambda_reg = baseline_param['lambda_reg']
                local_model = self.model.layer.weight.clone().detach()
                local_model = local_model.ge(0).long() * local_model  # : Add FedCAB term
                global_param = global_param.ge(0).long() * global_param  # : Add FedCAB term
                loss_kld = KLD(local_model.softmax(dim=-1).log(), global_param.softmax(dim=-1))  # : FedCAB loss
                reg_term = sum((p - g).norm() ** 2 for p, g in zip(local_model, global_param))

                loss += 0.5 * lambda_reg * reg_term

                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                        "max": local_solution.max().item(),
                        "min": local_solution.min().item()}
            return_dict = {"loss": train_loss/train_total,
                        "acc": train_acc/train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict

    def local_train_moon(self, train_dataloader, prev_model, baseline_param,  **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        global_param = self.model.layer.weight.clone().detach()  # : Fetch global model
        import copy
        global_model = copy.deepcopy(self.model)
        temperature = baseline_param['temperature']
        mu = baseline_param['mu']


        self.model.train()
        train_loss = train_acc = train_total = 0
        # local_counter = 0

        for _ in range(self.local_step): # train K epochs.
            for (x, y) in train_dataloader:

                # if local_counter >= self.local_step:
                #     break
                # local_counter+=1

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                # : for MOON
                global_pred = global_model(x)
                posi = cos_criterion(pred, global_pred)
                logits = posi.reshape(-1, 1)

                set_flat_params_to(global_model, prev_model)
                prev_model_pred = global_model(x)
                nega = cos_criterion(pred, prev_model_pred)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = mu * criterion(logits, labels)

                loss = criterion(pred, y)

                loss += loss2

                # : Add FedProx term
                # lambda_reg = 0.0
                # local_model = self.model.layer.weight.clone().detach()
                # reg_term = sum((p - g).norm() ** 2 for p, g in zip(local_model, global_param))
                # loss += 0.5 * lambda_reg * reg_term


                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                        "max": local_solution.max().item(),
                        "min": local_solution.min().item()}
            return_dict = {"loss": train_loss/train_total,
                        "acc": train_acc/train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict

    def evaluate(self, dataloader, **kwargs):
        """
            Evaluate model on a dataset
        Args:
            dataloader: DataLoader class in Pytorch

        Returns
            Test loss and Test acc
        """
        self.model.eval()

        test_loss = test_acc = test_total = 0

        with torch.no_grad():
            for (x, y) in dataloader:
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                test_loss += loss.item() * y.size(0)
                test_acc += correct
                test_total += target_size

        return_dict = {"num_samples": test_total,
                        "total_loss": test_loss,
                       "total_correct": test_acc}
        return return_dict