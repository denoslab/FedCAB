import random
import matplotlib.pyplot as plt

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
import torch

class FedAvgTrainer(BaseTrainer):
    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.cnn = (options['model'] =='cnn')
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedAvgTrainer, self).__init__(options, dataset, model, self.optimizer, result_dir)
        self.clients_per_round = min(options['clients_per_round'], len(self.clients))

        self.budget = []
        self.cache_model = [torch.zeros(size=(878538 if self.cnn else 7850,)) for idx in range(self.num_round)]
        self.cache_model_valid = [False for idx in range(self.num_round)]

        self.recent_activeness = {}
        self.recent_active_size = 20
        self.recent_active_window = [[-1 for i in range(self.recent_active_size)] for i in range(len(self.clients))]
        self.recent_active_pointer = [0 for i in range(len(self.clients))]

        self.using_dif_model = False


    def train(self):
        KLD = torch.nn.KLDivLoss(reduction='sum')
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        
        true_round = 0

        strategy = "fedavg"  # "moon" or "fedavg"
        lambda_reg = 0.9  # for FedProx. Set 0 for FedAvg
        temperature = 0.6  # for MOON
        mu = 0.7  # for MOON
        baseline_param = {'lambda_reg': lambda_reg, "temperature": temperature, "mu": mu}


        using_budget = True  # Budget
        basic_budget = 20
        self.budget = [basic_budget for i in range(len(self.clients))]    # Budget

        using_client_holdoff = True  # Hold-off
        holdoff_round = 50  # Hold-off
        held_clients = [i for i in range(20)]  # Hold-off

        decay_reset_flag = False  # Reset Decay
        using_hard_decay = True
        using_soft_decay = False

        cache_solns = [torch.zeros(size=(878538 if self.cnn else 7850,)) for idx in range(len(self.clients))]
        is_cached = [False for idx in range(len(self.clients))]

        using_caching_the_change = False  # X
        cache_threshold = 7.35e-05
        loss_min = 0x3f3f3f00
        loss_max = 0
        loss_total = 0
        loss_count = 0

        using_ranked_loss = False
        kl_loss_rank_mtpl_basic = 100.0
        kl_loss_rank_decay = 1.6

        using_random_init_samp = False
        random_init_samp_round = 5
        random_samp_model = torch.zeros(size=(878538 if self.cnn else 7850,))

        using_cache_loss = False

        update_counter = [0 for i in range(len(self.clients))]
        update_modifier_max = 2.0
        update_booster_base = 2.1
        update_booster_decay = 0.055
        update_booster = [update_booster_base for i in range(len(self.clients))]

        using_cached_aggregation = False  # X
        using_cached_aggregation_with_lr = False  # X

        client_activity = {}
        recent_activeness = {}
        for k in range(len(self.clients)):
            client_activity[k] = 0
            recent_activeness[k] = 0

        plot_x_parti_cnt = [0 for i in range(self.num_round)]
        plot_y_round = [i for i in range(self.num_round)]
        plot_x_stat_parti = [[0 for i in range(60)] for j in range(10)]
        plot_y_stat_round = [i*10 for i in range(60)]

        avail_clients = set()
        selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False).tolist()
        set_selected_clients = set([c.cid for c in selected_clients])

        for round_i in range(self.num_round):
            if not self.budget_check():
                break
            print("round", round_i)
            selected_clients = []
            parti = []  # list of participating clients this round
            participating_client = []
            rand_boost = 0
            finish_flag = True

            while len(selected_clients) == 0:
                selected_clients = []
                print("looping warning")
                new_clients = self.get_avail_clients(seed=(round_i+rand_boost))
                selected_clients = new_clients

                # Using Controlled Client Updating
                if using_client_holdoff:
                    final_selection = []
                    for cli in selected_clients:
                        if (cli.cid in held_clients) and round_i < holdoff_round:
                            pass
                        else:
                            final_selection.append(cli)
                    selected_clients = final_selection

                # Early-stop: budget check
                if using_budget:
                    rand_samp_flag = False
                    rand_pool = []
                    for _, val in enumerate(self.budget):
                        if val > 0:
                            finish_flag = False
                            if rand_boost >= 5:
                                rand_samp_flag = True
                                rand_pool.append(_)
                            # break
                    if rand_samp_flag:
                        np.random.seed((round_i + rand_boost) * (self.options['seed'] + 1))
                        coin = random.randint(0, len(rand_pool)-1)
                        for c in self.clients:
                            if c.cid == rand_pool[coin]:
                                selected_clients.append(c)

                    if finish_flag:
                        break

                # Using Budget
                if using_budget:
                    final_selection = []
                    for cli in selected_clients:
                        if self.budget[cli.cid] <= 0:
                            pass
                        else:
                            final_selection.append(cli)
                            self.budget[cli.cid] -= 1

                    selected_clients = final_selection

                for cli in selected_clients:
                    parti.append((cli.cid, self.budget[cli.cid]))
                participating_client = selected_clients

                if round_i == 0:
                    finish_flag = False
                    break
                rand_boost += 1

            # repeated query each device until devices in the selected subset are all available
            if not finish_flag:
                # fetch diffed model
                if self.using_dif_model:
                    if round_i < 10:
                        pass
                    else:
                        self.diffed_model = self.latest_model
                        for i in range(2, 10):
                            for j in range(round_i, 0, -1):
                                if self.dif_model_valid[i][j]:
                                    self.diffed_model = self.diffed_model - self.dif_model[i][j] / 8
                                    break
                                else:
                                    pass

                participated_unavail = []  # They use their cache because they're unavailable

                # Solve minimization locally
                if len(selected_clients):
                    print("Participation: ", parti)
                    solns, stats = self.local_train(true_round, participating_client, strategy, baseline_param)
                    # Caching each client's latest update
                    if using_caching_the_change:
                        new_solns = []
                        for idx, cli_stat in enumerate(stats):
                            old_cache = cache_solns[cli_stat['id']].clone().detach()
                            old_cache = old_cache.ge(0).long() * old_cache

                            new_cache = solns[idx][1].clone().detach()
                            new_cache - new_cache.ge(0).long() * new_cache

                            global_cache = self.latest_model.clone().detach()
                            global_cache = global_cache.ge(0).long() * global_cache

                            loss = KLD(new_cache.softmax(dim=-1).log(), old_cache.softmax(dim=-1))
                            global_loss = KLD(new_cache.softmax(dim=-1).log(), global_cache.softmax(dim=-1))
                            ranking_loss[cli_stat['id']] = loss
                            if loss > cache_threshold and is_cached[cli_stat['id']]:
                                self.budget[cli_stat['id']] += 1
                                final_selection = []
                                for pti in participating_client:
                                    if pti.cid == cli_stat['id']:
                                        print("Threshold Failed Remove: ", pti.cid)
                                    else:
                                        final_selection.append(pti)
                                participating_client = final_selection
                            else:
                                cache_solns[cli_stat['id']] = solns[idx][1]
                                is_cached[cli_stat['id']] = True
                                new_solns.append(solns[idx])
                            if self.using_dif_model:
                                self.dif_model[cli_stat['id']][round_i] = solns[idx][1] - self.latest_model
                                self.dif_model_valid[cli_stat['id']][round_i] = True
                        solns = new_solns
                        parti = []
                        for pti in participating_client:
                            parti.append((pti.cid, self.budget[pti.cid]))
                        print("Real Participation: ", parti)

                    elif using_ranked_loss:
                        ranked_solns = []
                        ranking_loss = {}  # used for ranking
                        activity_rank = {}
                        ranking_weight = {}  # used for ranking
                        participate_counter = {}
                        new_solns = []
                        unavailable_clients = []


                        if using_cache_loss:
                            unavailable_clients = [i for i in range(len(self.clients))]

                        for idx, cli_stat in enumerate(stats):
                            if using_cache_loss:
                                unavailable_clients.remove(cli_stat['id'])
                            ranking_weight[cli_stat['id']] = 1.0
                            old_cache = cache_solns[cli_stat['id']].cuda().clone().detach()
                            old_cache = old_cache.ge(0).long() * old_cache

                            new_cache = solns[idx][1].cuda().clone().detach()
                            new_cache = new_cache.ge(0).long() * new_cache

                            global_cache = self.latest_model.cuda().clone().detach()
                            global_cache = global_cache.ge(0).long() * global_cache

                            loss = KLD(new_cache.softmax(dim=-1).log(), old_cache.softmax(dim=-1))
                            cache_loss = KLD(global_cache.softmax(dim=-1).log(), old_cache.softmax(dim=-1))
                            global_loss = KLD(global_cache.softmax(dim=-1).log(), new_cache.softmax(dim=-1))
                            # global_loss = KLD(new_cache.softmax(dim=-1).log(), global_cache.softmax(dim=-1))

                            # Can be loss, cache_loss or global_loss
                            if using_cache_loss:
                                ranking_loss[cli_stat['id']] = global_loss if cache_loss >= global_loss else cache_loss
                            else:
                                ranking_loss[cli_stat['id']] = global_loss

                            participate_counter[cli_stat['id']] = self.budget[cli_stat['id']]
                            # for clac in client_activity.items():   # global activeness calculation
                            for clac in recent_activeness.items():   # recent activeness calculation
                                if clac[0] == cli_stat['id']:
                                    activity_rank[clac[0]] = clac[1]

                        print("UNAVAIL LIST:", unavailable_clients)
                        if using_cache_loss:
                            for unavail_cli in unavailable_clients:
                                if is_cached[unavail_cli]:
                                    ranking_weight[unavail_cli] = 1.0
                                    old_cache = cache_solns[unavail_cli].clone().detach()
                                    global_cache = self.latest_model.clone().detach()
                                    global_cache = global_cache.ge(0).long() * global_cache
                                    cache_loss = KLD(global_cache.softmax(dim=-1).log(), old_cache.softmax(dim=-1))
                                    activity_rank[unavail_cli] = client_activity[unavail_cli]
                                    ranking_loss[unavail_cli] = cache_loss

                        # activity_rank = sorted(activity_rank.items(), key=lambda x: x[1], reverse=True)  # global activeness calculation
                        activity_rank = sorted(activity_rank.items(), key=lambda x: x[1])  # recent activeness calculation
                        ranking_loss = sorted(ranking_loss.items(), key=lambda x: x[1], reverse=True)

                        for idx, aval in enumerate(activity_rank):
                            ranking_weight[aval[0]] *= (99 if self.budget[aval[0]] >= (basic_budget - 1) else ((idx + 1) / len(activity_rank)))

                        for idx, aval in enumerate(activity_rank):
                            ranking_weight[aval[0]] *= update_booster[aval[0]]


                        update_mean = float(sum(update_counter))/float(len(update_counter))
                        max_sub = max(update_counter) - min(update_counter)
                        if not max_sub == 0 and not (update_mean == max_sub) and not (update_mean == 0):
                            xxa, xxb, xxc = self.special_function_solver_2(update_modifier_max, max_sub, update_mean)
                            for idx, aval in enumerate(activity_rank):
                                bval = update_counter[aval[0]]
                                # ranking_weight[aval[0]] *= (xxa*bval*bval + xxb*bval + xxc)
                                if update_counter[aval[0]] <= update_mean:
                                    # ranking_weight[aval[0]] *= (1.0 + (update_mean - update_counter[aval[0]]) / update_mean)
                                    ranking_weight[aval[0]] *= 1.3

                        if kl_loss_rank_mtpl_basic > 1.0:
                            xxa, xxb, xxc = self.special_function_solver(kl_loss_rank_mtpl_basic, len(ranking_loss))
                            for idx, aval in enumerate(ranking_loss):
                                ranking_weight[aval[0]] *= (xxa*(idx+1)*(idx+1) + xxb*(idx+1) + xxc)
                        else:
                            for idx, aval in enumerate(ranking_loss):
                                ranking_weight[aval[0]] *= ((idx + 1) / len(ranking_loss))

                        if using_random_init_samp and round_i <= random_init_samp_round:
                            participate_counter = sorted(participate_counter.items(), key=lambda x: x[1], reverse=True)
                            for idx, aval in enumerate(participate_counter):
                                ranking_weight[aval[0]] *= (9999 if aval[1] >= (basic_budget-1) else 1)

                        ranking_weight = sorted(ranking_weight.items(), key=lambda x: x[1], reverse=True)

                        for idx, loss in enumerate(ranking_weight):
                            if idx < self.clients_per_round:
                                if using_cache_loss and (loss[0] in unavailable_clients):
                                    if is_cached[loss[0]]:
                                        update_counter[loss[0]] += 1
                                        new_solns.append((108, cache_solns[loss[0]]))
                                        participated_unavail.append(loss[0])
                                else:
                                    for idx_solns, cli_stat in enumerate(stats):
                                        if cli_stat['id'] == loss[0]:
                                            update_counter[loss[0]] += 1
                                            new_solns.append(solns[idx_solns])
                                            if self.using_dif_model:
                                                self.dif_model[cli_stat['id']][round_i] = solns[idx_solns][1] - self.latest_model
                                                self.dif_model_valid[cli_stat['id']][round_i] = True
                            else:
                                if using_cache_loss and (loss[0] in unavailable_clients):
                                    print("(Cache) Sampling Failed Remove: ", loss[0])
                                else:
                                    self.budget[loss[0]] += 1
                                    final_selection = []
                                    for pti in participating_client:
                                        if pti.cid == loss[0]:
                                            pass
                                        else:
                                            final_selection.append(pti)
                                    participating_client = final_selection
                        parti = []
                        for pti in participating_client:
                            parti.append((pti.cid, self.budget[pti.cid]))
                            self.recent_active_window[pti.cid][self.recent_active_pointer[pti.cid]] = round_i
                            self.recent_active_pointer[pti.cid] += 1
                            if self.recent_active_pointer[pti.cid] >= self.recent_active_size:
                                self.recent_active_pointer[pti.cid] = 0
                            recent_cnt = 0
                            for i in self.recent_active_window[pti.cid]:
                                if i in range(max(round_i - self.recent_active_size + 1, 0), round_i + 1):
                                    recent_cnt += 1
                            self.recent_activeness[pti.cid] = recent_cnt / self.recent_active_size
                            update_booster[pti.cid] -= update_booster_decay
                            if update_booster[pti.cid] < 1:
                                update_booster[pti.cid] = 1
                        print("Recent Activeness:")
                        print(self.recent_activeness)
                        print("Real Participation: ", parti)
                        print("Unavailable Participant: ", participated_unavail)
                        plot_x_parti_cnt[round_i] = len(parti)
                        solns = new_solns
                        kl_loss_rank_mtpl_basic -= kl_loss_rank_decay
                        if kl_loss_rank_mtpl_basic < 1.0:
                            kl_loss_rank_mtpl_basic = 1.0

                    else:
                        new_solns = []
                        selected_ = []
                        selected_cid = []
                        if len(stats) <= self.clients_per_round:
                            for idx, cli_stat in enumerate(stats):
                                selected_cid.append(cli_stat['id'])
                                cache_solns[cli_stat['id']] = solns[idx][1]
                                is_cached[cli_stat['id']] = True
                                if self.using_dif_model:
                                    self.dif_model[cli_stat['id']][round_i] = solns[idx][1] - self.latest_model
                                    self.dif_model_valid[cli_stat['id']][round_i] = True
                        else:
                            np.random.seed((round_i + rand_boost) * (self.options['seed'] + 1))
                            for idx in range(self.clients_per_round):
                                ridx = random.randint(0, len(stats)-1)
                                while ridx in selected_:
                                    ridx = random.randint(0, len(stats) - 1)
                                selected_.append(ridx)
                                selected_cid.append(stats[ridx]['id'])
                                cache_solns[stats[ridx]['id']] = solns[ridx][1]
                                is_cached[stats[ridx]['id']] = True
                                new_solns.append(solns[ridx])
                                if self.using_dif_model:
                                    self.dif_model[stats[ridx]['id']][round_i] = solns[ridx][1] - self.latest_model
                                    self.dif_model_valid[stats[ridx]['id']][round_i] = True
                            solns = new_solns

                        for _, cli_stat in enumerate(stats):
                            if cli_stat['id'] not in selected_cid:
                                self.budget[cli_stat['id']] += 1
                                final_selection = []
                                for pti in participating_client:
                                    if pti.cid == cli_stat['id']:
                                        pass
                                    else:
                                        final_selection.append(pti)
                                participating_client = final_selection

                        parti = []
                        for pti in participating_client:
                            parti.append((pti.cid, self.budget[pti.cid]))
                        print("Real Participation: ", parti)
                        plot_x_parti_cnt[round_i] = len(parti)

                    for pti, _ in parti:
                        plot_x_stat_parti[int(pti/10)][int(round_i/10)] += 1

                for cli in participating_client:
                    client_activity[cli.cid] = cli.getRecentActivity(round_i)
                    recent_activeness[cli.cid] = -1
                for pu in participated_unavail:
                    for cli in self.clients:
                        if cli.cid == pu:
                            client_activity[cli.cid] = cli.getRecentActivity(round_i)
                    recent_activeness[pu] = -1
                for idx in range(len(recent_activeness)):
                    recent_activeness[idx] += 1

                # Update latest model
                if len(participating_client):
                    if using_cached_aggregation:  # Cache aggregate
                        self.latest_model = self.aggregate_using_cache(cache_solns, is_cached,
                                                                       using_cached_aggregation_with_lr)
                    elif using_random_init_samp and round_i <= random_init_samp_round:
                        if round_i == random_init_samp_round:
                            rmodel = self.aggregate(solns, stats)
                            random_samp_model += rmodel
                            rmodel = random_samp_model / random_init_samp_round
                            self.latest_model = rmodel
                        else:
                            rmodel = self.aggregate(solns, stats)
                            random_samp_model += rmodel
                    else:
                        self.latest_model = self.aggregate(solns, stats)  # FedAvg aggregate

                    if using_hard_decay:
                        self.optimizer.inverse_prop_decay_learning_rate(true_round + 1)
                    elif using_soft_decay:
                        self.optimizer.soft_decay_learning_rate()
                    self.cache_model[round_i] = self.latest_model
                    self.cache_model_valid[round_i] = True

                # Reset the learning rate
                if round_i >= holdoff_round and decay_reset_flag:
                    self.optimizer.set_learning_rate_to_001(true_round + 1)
                    decay_reset_flag = False
                
                train_loss, train_acc = self.evaluate_train()
                test_loss, test_acc = self.evaluate_test()
                out_dict = {'train_loss': train_loss, 'train_acc':train_acc,'test_loss': test_loss, 'test_acc':test_acc}
                print("training loss & acc",train_loss, train_acc )
                print("test loss & acc", test_loss, test_acc)
                self.logger.log(round_i ,out_dict)
                self.logger.dump()
                if (round_i > random_init_samp_round) or (not using_random_init_samp):
                    true_round += 1

        print(self.budget)
        train_loss, train_acc = self.evaluate_train()
        test_loss, test_acc = self.evaluate_test()
        out_dict = {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc}
        print("final training loss & acc", train_loss, train_acc)
        print("final test loss & acc", test_loss, test_acc)
        x_bar = [i*5 for i in range(int((self.num_round/5)))]

        # plt.plot(plot_y_round, plot_x_parti_cnt, '-', color=(0, 0, 1))

        for idx in range(10):
            plt.plot(plot_y_stat_round, plot_x_stat_parti[idx], '-', color=(1-0.1*idx, (0.5-0.1*idx) if idx<5 else (1.5-0.1*idx), 0.1*idx))
        label = ["Group %d" % i for i in range(10)]
        plt.legend(label, loc='best')
        plt.xticks(x_bar, fontsize=8)
        plt.show()

    def aggregate(self, solns, stats, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """
        averaged_solution = torch.zeros_like(self.latest_model)
        for _, local_solution in solns:
            averaged_solution += local_solution
        averaged_solution /= len(solns)

        participant = []

        for idx, cli_stat in enumerate(stats):
            participant.append(cli_stat['id'])

        use_weight = False

        if use_weight:
            averaged_solution = 0.5 * self.latest_model + 0.5 * averaged_solution
            print("Use Weighted Aggregation")
        else:
            pass

        return averaged_solution.detach()

    def aggregate_using_cache(self, cache, cache_flag, using_lr):
        update_table = []
        for idx, slv in enumerate(cache):
            if cache_flag[idx]:
                update_table.append(slv)

        if using_lr:
            return self.optimizer.get_current_lr() * sum(update_table) / len(update_table)
        else:
            return sum(update_table) / len(update_table)

    def special_function_solver(self, alpha, num_client):
        # x1, y1 = dot_zero
        # x2, y2 = dot_numClient
        # x3, y3 = dot_symmetricZero

        y1 = alpha
        x2 = num_client
        y2 = 1

        c = y1
        a = (c - y2) / (x2 * x2)
        b = -2 * x2 * a

        # print("y = %fxx + %fx + %f" % (a, b, c))
        return a, b, c

    def special_function_solver_2(self, alpha, max_sub, mean):  # problem
        #  passes (max_sub, 1/alpha), (mean, 1)
        c = alpha
        a = (1.0 - alpha - (1.0/alpha - alpha)*mean / max_sub) / (mean*mean - max_sub*mean)
        b = (1.0/alpha - alpha - a*max_sub*max_sub) / max_sub

        # print("y = %fxx + %fx + %f" % (a, b, c))
        return a, b, c

    def budget_check(self):
        for ite in self.budget:
            if ite > 0:
                return True
        return False
