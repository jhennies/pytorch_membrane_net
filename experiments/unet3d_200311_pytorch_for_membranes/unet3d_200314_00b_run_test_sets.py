
import os
from h5py import File
from pytorch_tools.piled_unets import MembraneNet
import torch as t

from torchsummary import summary
import numpy as np
from pytorch_tools.run_models import predict_model_from_h5_parallel_generator
from glob import glob


experiment_name = 'unet3d_200314_00b_membranes_no_gauss'

# Where the Weights are stored (the folder)
net_folder = os.path.join(
    '/g/schwab/hennies/phd_project/image_analysis/autoseg/cnn_3d_devel',
    'unet3d_200311_pytorch_for_membranes',
    experiment_name
)

# Where the results will be written to
results_folder = os.path.join(
    net_folder,
    'results_best_no_overlap_write_in_and_out'
)

# Define the input here as list (Batch processing possible)
im_list = sorted(glob(os.path.join(
    '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/all_datasets_test_samples',
    '*.h5'
)))

model = MembraneNet(predict=True)
model.cuda()
summary(model, (1, 64, 64, 64))

model.load_state_dict(t.load(os.path.join(net_folder, 'best_model.pth')))

if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# Does not do anything if set to zero
aug_dict_preprocessing = dict(
    smooth_output_sigma=0
)

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
            # spacing=(32, 32, 32),
            spacing=(64, 64, 64),
            area_size=area_size,
            target_shape=(64, 64, 64),
            num_result_channels=1,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            n_workers=16,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            full_dataset_shape=None,
            write_in_and_out=True
        )

