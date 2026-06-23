# flake8: noqa
import os.path as osp
from qualiteacher.train_pipeline import train_pipeline

import qualiteacher.archs
import qualiteacher.data
import qualiteacher.models
import qualiteacher.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
