import argparse
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from module import TGAN
from tgopt import NeighborFinder

parser = argparse.ArgumentParser(Path(__file__).name)
parser.add_argument('-d',
                    '--data',
                    type=str,
                    required=True,
                    help='dataset to use (e.g. snap-msg or jodie-wiki)')
parser.add_argument('--model',
                    type=str,
                    required=True,
                    help='prefix to name the saved model')
parser.add_argument('--dir',
                    type=str,
                    default='data',
                    help='directory to load data files (default: data)')
parser.add_argument('--bs',
                    type=int,
                    default=200,
                    help='batch size (default: 200)')
parser.add_argument('--n-epoch', type=int, default=5)
parser.add_argument('--n-degree',
                    type=int,
                    default=20,
                    help='number of neighbors to sample (default: 20)')
parser.add_argument('--n-layer', type=int, default=2)
parser.add_argument(
    '--n-head',
    type=int,
    default=2,
    help='number of heads used in attention layer (default: 2)')
parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--drop-out',
                    type=float,
                    default=0.1,
                    help='dropout probability (default: 0.1)')
parser.add_argument('--patience',
                    type=int,
                    default=5,
                    help='early stop patience (default: 5)')
parser.add_argument('--gpu',
                    type=int,
                    default=-1,
                    help='idx for the gpu to use (default: -1 for cpu)')
args = parser.parse_args()

DATA = args.data
BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
NUM_NEIGHBORS = args.n_degree
NUM_LAYER = args.n_layer
NUM_HEADS = args.n_head
LEARNING_RATE = args.lr
DROP_OUT = args.drop_out
PATIENCE = args.patience
GPU = args.gpu

# ENABLE_OPTS = (args.opt_all or args.opt_dedup or args.opt_cache or args.opt_time)

Path('./logs').mkdir(parents=True, exist_ok=True)
Path('./saved_models').mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.model}-{args.data}.pth'
data_dir = Path(args.dir)

### Set up logger
log_time = int(time.time())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'logs/{args.model}-{args.data}-{str(log_time)}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


# why need this ?
class RandEdgeSampler(object):

    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        t_total = 0
        for k in range(num_test_batch):
            t_start = time.perf_counter()
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut,
                                               dst_l_fake, ts_l_cut,
                                               NUM_NEIGHBORS)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(),
                                         (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            t_total += (time.perf_counter() - t_start)
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
        logger.info(f'inference total elapsed: {t_total} secs')
        logger.info(
            f'average inference time per batch: {t_total / num_test_batch} secs'
        )
    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)


g_df = pd.read_csv(data_dir / f'ml_{DATA}.csv')
e_feat = np.load(data_dir / f'ml_{DATA}.npy')
n_feat = np.load(data_dir / f'ml_{DATA}_node.npy')

val_time, test_time = list(np.quantile(g_df.ts,
                                       [0.70, 0.85]))  # why need this?

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

random.seed(2022)

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

mask_node_set = set(
    random.sample(
        tuple(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))),
        int(0.1 * num_total_unique_nodes)))
mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)
assert (len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set

# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

is_new_node_edge = np.array([(a in new_node_set or b in new_node_set)
                             for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge
nn_test_flag = valid_test_flag * is_new_node_edge

# validation and test with all edges
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]

# validation and test with edges that at least has one new node (not in training set)
nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_e_idx_l = e_idx_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]

nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l,
                              train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list)
del adj_list

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list)
del full_adj_list

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

### Model load
device = torch.device(f'cuda:{GPU}' if GPU >= 0 else 'cpu')
tgan = TGAN(train_ngh_finder,
            n_feat,
            e_feat,
            num_layers=NUM_LAYER,
            num_heads=NUM_HEADS,
            drop_out=DROP_OUT)
state = torch.load(MODEL_SAVE_PATH, map_location=device)
state['n_feat_th'] = tgan.n_feat_th
state['e_feat_th'] = tgan.e_feat_th
state['node_raw_embed.weight'] = tgan.n_feat_th
state['edge_raw_embed.weight'] = tgan.e_feat_th
tgan.load_state_dict(state)
del state

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

saved_n_feat_th = tgan.n_feat_th
saved_e_feat_th = tgan.e_feat_th

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.info(f'num of training instances: {num_instance}')
logger.info(f'num of batches per epoch: {num_batch}')
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

for epoch in range(NUM_EPOCH):
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start epoch {}'.format(epoch))

    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance, s_idx + BATCH_SIZE)

        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[
            s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake,
                                           ts_l_cut, NUM_NEIGHBORS)
        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(),
                                         (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            # f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))

    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes',
                                                      tgan, val_rand_sampler,
                                                      val_src_l, val_dst_l,
                                                      val_ts_l, val_label_l)
    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(
        'val for new nodes', tgan, nn_val_rand_sampler, nn_val_src_l,
        nn_val_dst_l, nn_val_ts_l, nn_val_label_l)

    logger.info(f'epoch mean loss: {np.mean(m_loss)}')
    logger.info(
        f'train acc: {np.mean(acc)}, val acc: {val_acc}, new node val acc: {nn_val_acc}'
    )

# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes',
                                                      tgan, test_rand_sampler,
                                                      test_src_l, test_dst_l,
                                                      test_ts_l, test_label_l)
nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(
    'test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l,
    nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(
    test_acc, test_auc, test_ap))
logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(
    nn_test_acc, nn_test_auc, nn_test_ap))
