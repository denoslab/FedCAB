# FedCAB

This repository is the source code implementation of [Federated Learning with Client Availability Budgets](https://edas.info/showManuscript.php?m=1570915829&ext=pdf&random=2013857912&type=stamped). 

We present FedCAB, an algorithm applying our theoretical model for the probabilistic rankings of the available clients to select in each round of FL model aggregation.

Our code is adapted from the [code](https://github.com/hmgxr128/MIFA_code) for paper [Fast Federated Learning in the Presence of Arbitrary Device Unavailability](https://arxiv.org/abs/2106.04159).

## Data Preparation

Modify the function called in line 166 in ```data/generate_equal.py``` to specify the heterogenous data allocation method.
Non-iid data can be generated with each device holding samples of only two classes with function `choose_two_digit(split_this_traindata)`, or you can specify labels contained for each group with function `choose_arbtr_digit(user)`.

## Training

To start an example FL training, simply run this command:

```shell
python main.py --result_dir results_mnist_logit_p1 --dataset mnist_all.pkl --algo fedavg --num_user 100 --participation_level 1 --participation_pattern adversarial --clients_per_round 10 --model logistic --num_round 250 --local_step 1 --batch_size 100 --lr 0.01 --wd 0.001
```

You can modify the shell script to change the experiment setup. The meaning of each variable is listed as follows: 

- num_user: the total number of clients
- clients_per_round: is the number of participating devices each round for FedAvg with device sampling.  
- num_round: the total number of communication rounds
- local_step: the number of local epochs
- device: GPU id
- model: the model, should be set as "logistic" or "cnn"
- wd: weight decay
- algo: the algorithm should be set to fedavg as we implemented all of FedCAB algorithm in fedavg.py
- result_dir: the directory for experiment logs

During training, the logs will be saved under the directory specified by the user. For each run, the folder is named as the hash of the starting time. Each folder contains two files, i.e. ```options.json``` and ```log.json```. The former records the experiment setup and the latter records the training loss, training accuracy, test loss and test accuracy.

To train with FedCAB, set the variable `using_ranked_loss = True` in line 69 in```src/trainers/fedavg.py```. You can also adjust the hyperparameters given for this algorithm in this file. 
$\alpha$ is indicated as `kl_loss_rank_mtpl_basic` in line 70 and its decay value is initialized in line 71. 
$\beta$ is indicated as `update_booster_base` in line 81 and its decay value is initialized in line 82.
Line 47 to 49 initialize the budgets allocated for the clients.
Line 51 to 53 initialize the late-join clients.

To train with other FL algorithms, modify the parameter in line 40 in ```src/trainers/fedavg.py``` to specify the algorithm. Use "moon" to run [MOON](https://arxiv.org/abs/2103.16257); use "fedavg" to run FedAvg or [FedProx](https://arxiv.org/abs/1812.06127) based on the parameters given. You can specify the hyperparameters used for MOON or FedProx by modifying parameters from line 41 to 43.

## Visualization

To visualize the training curves, run this command: 

```shell
python plot.py $LOG_DIR $PLOT_OPTION $DATASET
```

The usage of variables is listed as follows: 

- \$LOG_DIR: the directory for experiment logs
- \$PLOT_OPTION: should be in $\{0, 1, 2, 3\}$, corresponding to training loss, training accuracy, test loss and test_accuracy.
- $DATASET: should be 'cifar' or 'mnist'.

Example:

```
python plot.py results_mnist_logit_p1 3 mnist
```
