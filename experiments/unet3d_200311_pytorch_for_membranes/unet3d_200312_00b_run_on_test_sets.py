
from pytorch.pytorch_tools.data_generation import parallel_data_generator
import os
from h5py import File
from pytorch.pytorch_tools.piled_unets import PiledUnet
from pytorch.pytorch_tools.losses import WeightMatrixWeightedBCE, CombinedLosses
from pytorch.pytorch_tools.training import train_model_with_generators, cb_save_model, cb_run_model_on_data
import torch as t

from torchsummary import summary
import numpy as np
from pytorch.pytorch_tools.run_models import predict_model_from_h5_parallel_generator
from glob import glob


experiment_name = 'unet3d_200312_00_membranes_epochs300'
net_folder = os.path.join(
    '/g/schwab/hennies/phd_project/image_analysis/autoseg/cnn_3d_devel',
    'unet3d_200311_pytorch_for_membranes',
    experiment_name
)
results_folder = os.path.join(
    net_folder,
    'results_0036'
)

model = PiledUnet(
    n_nets=3,
    in_channels=1,
    out_channels=[1, 1, 1],
    filter_sizes_down=(
        ((16, 32), (32, 64), (64, 128)),
        ((16, 32), (32, 64), (64, 128)),
        ((16, 32), (32, 64), (64, 128))
    ),
    filter_sizes_bottleneck=(
        (128, 256),
        (128, 256),
        (128, 256)
    ),
    filter_sizes_up=(
        ((256, 128, 128), (128, 64, 64), (64, 32, 32)),
        ((256, 128, 128), (128, 64, 64), (64, 32, 32)),
        ((256, 128, 128), (128, 64, 64), (64, 32, 32))
    ),
    batch_norm=True,
    output_activation='sigmoid',
    predict=True
)
model.cuda()
summary(model, (1, 64, 64, 64))

model.load_state_dict(t.load(os.path.join(net_folder, 'model_0036.h5')))

if not os.path.exists(results_folder):
    os.mkdir(results_folder)

aug_dict_preprocessing = dict(
    smooth_output_sigma=0.5
)

im_list = sorted(glob(os.path.join(
    '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/all_datasets_test_samples',
    '*.h5'
)))

with t.no_grad():
    for filepath in im_list:
        with File(filepath, mode='r') as f:
            area_size = list(f['data'].shape)
            if area_size[0] > 256:
                area_size[0] = 256
            if area_size[1] > 256:
                area_size[1] = 256
            if area_size[2] > 256:
                area_size[2] = 256
            channels = [[f['data'][:]]]

        predict_model_from_h5_parallel_generator(
            model=model,
            results_filepath=os.path.join(results_folder, os.path.split(filepath)[1]),
            raw_channels=channels,
            spacing=(32, 32, 32),
            area_size=area_size,
            target_shape=(64, 64, 64),
            num_result_channels=1,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            n_workers=16,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            full_dataset_shape=None
        )
