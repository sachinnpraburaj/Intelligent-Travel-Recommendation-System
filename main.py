import argparse
import numpy as np
from utils import Util
from rbm import RBM

parser = argparse.ArgumentParser()
parser.add_argument('--num_hid', type=int, default=64,
                    help='Number of hidden layer units (latent factors)')
parser.add_argument('--user', type=int, default=22,
                    help='user id to recommend books \
                    to (not all ids might be present)')
parser.add_argument('--data_dir', type=str, default='data', required=True,
                    help='path to dataset')
parser.add_argument('--rows', type=int, default=200000,
                    help='number of rows to be used for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--free_energy', type=bool, default=False,
                    help='Export free energy plot')
parser.add_argument('--verbose', type=bool, default=False,
                    help='Display info after each epoch')
args = parser.parse_args()

def main():
    util = Util()
    dir = args.data_dir
    rows = args.rows
    ratings, to_read, books = util.read_data(dir)
    ratings = util.clean_subset(ratings, rows)
    num_vis = len(ratings)
    free_energy = args.free_energy
    train = util.preprocess(ratings)
    valid = None
    if free_energy:
        train, valid = util.split_data(train)
    H = args.num_hid
    user = args.user
    alpha = args.alpha
    w = np.random.normal(loc=0, scale=0.01, size=[num_vis, H])
    rbm = RBM(alpha, H, num_vis)
    epochs = args.epochs
    batch_size = args.batch_size

    v = args.verbose
    reco, prv_w, prv_vb, prv_hb = rbm.training(train, valid, user,
                                                epochs, batch_size,
                                                free_energy, v)
    unread, read = rbm.calculate_scores(ratings, books,
                                        to_read, reco, user)
    rbm.export(unread, read)

if __name__ == "__main__":
    main()
