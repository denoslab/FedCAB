import torch
import numpy as np
import pickle
import os
import torchvision

cpath = os.path.dirname(os.path.abspath(__file__))

NUM_USER = 100
NONIID_Q = 0.5
SAVE = True

split_num = 21
np.random.seed(16)

class ImageDataset(object):
    def __init__(self, images, labels):
        if isinstance(images, torch.Tensor):
            self.data = images.numpy()
        else:
            self.data = images
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta = len(data) // num_split
    data_lst = []
    i = 0
    while i < delta * num_split:
        data_lst.append(data[i:i+delta])
        i += delta
    return data_lst


def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst

# Split the clients into 10 groups, and allocate the data
def choose_arbtr_digit(user):
    lst = [[] for i in range(NUM_USER)]
    lst[0] = [0, 1]
    lst[1] = [1, 2]
    lst[2] = [2, 3]
    lst[3] = [3, 4]
    lst[4] = [4, 5]
    lst[5] = [5, 6]
    lst[6] = [6, 7]
    lst[7] = [7, 8]
    lst[8] = [8, 9]
    lst[9] = [9, 2]

    ret = [[] for i in range(NUM_USER)]

    for idx in range(10):
        for j in range(10):
            ret[idx * 10 + j] = lst[idx]
    return ret[user]

def choose_degree_digit(user):
    lst = [[] for i in range(10)]
    np.random.seed(7)
    for loopidx in range(5):
        for idx in range(10):
            for group in range(10):
                prob = np.random.rand()
                if prob < NONIID_Q and idx == group:
                    lst[group].append(idx)
                elif prob < ((1.0-NONIID_Q)/9.0) and not idx == group:
                    lst[group].append(idx)

    ret = [[] for i in range(NUM_USER)]

    for idx in range(10):
        for j in range(10):
            ret[idx * 10 + j] = lst[idx]
    return ret[user]

def main(dataset):

    assert dataset in ['cifar', 'mnist', 'emnist'], 'dataset not supported!'
    dataset_dir = os.path.join(cpath, dataset)
    if dataset == 'mnist':
        # Get MNIST data, normalize, and divide by level
        print('>>> Get MNIST data.')
        trainset = torchvision.datasets.MNIST(dataset_dir, download=True, train=True)
        testset = torchvision.datasets.MNIST(dataset_dir, download=True, train=False)

        train_d = ImageDataset(trainset.data, trainset.targets)
        test_d = ImageDataset(testset.data, testset.targets)
    if dataset == 'cifar':
        # Get CIFAR data, normalize, and divide by level
        print('>>> Get CIFAR10 data.')
        trainset = torchvision.datasets.CIFAR10(dataset_dir, download=True, train=True)
        testset = torchvision.datasets.CIFAR10(dataset_dir, download=True, train=False)

        train_d = ImageDataset(trainset.data, trainset.targets)
        test_d = ImageDataset(testset.data, testset.targets)
    if dataset == 'emnist':
        # Get Extended MNIST data, normalize, and divide by level
        print('>>> Get EMNIST data.')
        trainset = torchvision.datasets.EMNIST(dataset_dir, download=True, split='digits', train=True)
        testset = torchvision.datasets.EMNIST(dataset_dir, download=True, split='digits', train=False)
        # trainset = torchvision.datasets.FashionMNIST(dataset_dir, download=True, train=True)
        # testset = torchvision.datasets.FashionMNIST(dataset_dir, download=True, train=False)

        train_d = ImageDataset(trainset.data, trainset.targets)
        test_d = ImageDataset(testset.data, testset.targets)

    this_traindata = []
    for number in range(10):
        idx = train_d.target == number
        this_traindata.append(train_d.data[idx])
    min_number = min([len(dig) for dig in this_traindata])
    for number in range(10):
        this_traindata[number] = this_traindata[number][:min_number-1]

    split_this_traindata = []
    for digit in this_traindata:
        split_this_traindata.append(data_split(digit, split_num))

    this_testdata = []
    for number in range(10):
        idx = test_d.target == number
        this_testdata.append(test_d.data[idx])
    split_this_testdata = []
    for digit in this_testdata:
        split_this_testdata.append(data_split(digit, split_num))

    data_distribution = np.array([len(v) for v in this_traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_this_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_this_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]
    classes_info = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(NUM_USER):
        print(user, np.array([len(v) for v in split_this_traindata]))

        # for d in choose_two_digit(split_this_traindata):
        for d in choose_arbtr_digit(user):
        # for d in choose_degree_digit(user):
            classes_info[user].append(d)
            l = len(split_this_traindata[d][-1])
            train_X[user] += split_this_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()
        for d in range(10):
            l = len(split_this_testdata[d][-1])
            test_X[user] += split_this_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    print("Classes Info:", classes_info)
    # Setup directory for train/test data
    print('>>> Set data path for {}.'.format(dataset))
    train_path = '{}/{}/data/train/all.pkl'.format(cpath,dataset)
    test_path = '{}/{}/data/test/all.pkl'.format(cpath,dataset)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 100 users
    for i in range(NUM_USER):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')

    return classes_info

def initAvailClientData():# copy of __main__ part for main.py
    from data.gen_avail_prob import gen_avail_prob_random, gen_avail_prob_adversarial, gen_avail_prob_equal, \
        gen_avail_prob_varied, gen_avail_prob_reverse
    for dataset in ('mnist', 'cifar'):
        class_info = main(dataset)
        for i in range(1, 10):  # 9_adversarial is for varying availability
            gen_avail_prob_equal(list(range(NUM_USER)), 0.1 * i, 1,
                                 '{}/{}/data/avail_prob_{}_random.pkl'.format(cpath, dataset, i), class_info)
            # gen_avail_prob_adversarial(list(range(NUM_USER)), 0.1 * i, 1, '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, i), class_info)
            gen_avail_prob_varied(list(range(NUM_USER)), 0.5, 1,
                                  '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, i), class_info)
    for dataset in ('emnist'):
        class_info = main(dataset)
        for i in range(1, 10):  # 9_adversarial is for varying availability
            gen_avail_prob_equal(list(range(NUM_USER)), 0.1 * i, 1,
                                 '{}/{}/data/avail_prob_{}_random.pkl'.format(cpath, dataset, i), class_info)

def initMifaClientData(num_clients):# copy of __main__ part for main.py
    from data.gen_avail_prob import gen_avail_prob_random, gen_avail_prob_adversarial, gen_avail_prob_equal, \
        gen_avail_prob_varied, gen_avail_prob_reverse
    global NUM_USER
    global split_num
    NUM_USER = num_clients
    split_num = num_clients

    for dataset in ('mnist', 'cifar'):
        class_info = main(dataset)

        for i in range(1, 10):
            gen_avail_prob_varied(list(range(NUM_USER)), 0.5, 1,
                                  '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, i), class_info)
        gen_avail_prob_varied(list(range(NUM_USER)), 0.5, 1,
                              '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, 1), class_info)
        gen_avail_prob_equal(list(range(NUM_USER)), 0.5, 1,
                              '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, 2), class_info)
        gen_avail_prob_reverse(list(range(NUM_USER)), 0.5, 1,
                              '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, 3), class_info)
        F = open(r'{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, 2),'rb')
        content = pickle.load(F)
    class_info = main('emnist')
    gen_avail_prob_equal(list(range(NUM_USER)), 0.5, 1,
                         '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, 2), class_info)

def callfunc():
    from data.gen_avail_prob import gen_avail_prob_random, gen_avail_prob_adversarial, gen_avail_prob_equal, \
        gen_avail_prob_varied, gen_avail_prob_reverse
    for dataset in ('mnist','cifar'):
        class_info = main(dataset)
        for i in range(1,10):   # 9_adversarial is for varying availability
            gen_avail_prob_reverse(list(range(NUM_USER)), 0.5, 1, '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath,dataset,i), class_info)

if __name__ == '__main__':
    from gen_avail_prob import gen_avail_prob_random, gen_avail_prob_adversarial, gen_avail_prob_equal, \
        gen_avail_prob_varied, gen_avail_prob_reverse
    for dataset in ('mnist','cifar'):
        class_info = main(dataset)
        for i in range(1,10):
            gen_avail_prob_equal(list(range(NUM_USER)), 0.1 * i, 1, '{}/{}/data/avail_prob_{}_random.pkl'.format(cpath,dataset,i), class_info)
            gen_avail_prob_varied(list(range(NUM_USER)), 0.5, 1, '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath,dataset,i), class_info)
    for dataset in ('emnist'):
        class_info = main(dataset)
        gen_avail_prob_equal(list(range(NUM_USER)), 0.5, 1,
                             '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath, dataset, 2), class_info)

