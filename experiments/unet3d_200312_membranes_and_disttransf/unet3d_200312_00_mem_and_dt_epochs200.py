
from pytorch.pytorch_tools.data_generation import parallel_data_generator
import os
from h5py import File
from pytorch.pytorch_tools.piled_unets import PiledUnet
from pytorch.pytorch_tools.losses import WeightMatrixWeightedBCE, WeightMatrixMSELoss, CombinedLosses
from pytorch.pytorch_tools.training import train_model_with_generators, cb_save_model, cb_run_model_on_data

from torchsummary import summary
import numpy as np

from torch.nn import MSELoss


experiment_name = 'unet3d_200312_00_mem_and_dt_epochs200'
results_folder = os.path.join(
    '/g/schwab/hennies/phd_project/image_analysis/autoseg/cnn_3d_devel',
    'unet3d_200312_membranes_and_disttransf',
    experiment_name
)

if True:

    raw_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    raw_filepaths = [
        [
            os.path.join(raw_path, 'raw.h5'),
        ],
    ]
    gt_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    gt_filepaths = [
        [
            os.path.join(gt_path, 'gt_mem.h5'),
            os.path.join(gt_path, 'gt_dt.h5'),
            os.path.join(gt_path, 'gt_mask_organelle_insides.h5')
        ],
    ]
    raw_channels = []
    for volumes in raw_filepaths:
        raws_data = []
        for chid, channel in enumerate(volumes):
            if chid == 1:
                # Specifically only load last channel of the membrane prediction
                raws_data.append(File(channel, 'r')['data'][..., -1])
            else:
                raws_data.append(File(channel, 'r')['data'][:])
        raw_channels.append(raws_data)
    gt_channels = [[File(channel, 'r')['data'][:] for channel in volumes] for volumes in gt_filepaths]

    val_raw_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    val_raw_filepaths = [
        [
            os.path.join(val_raw_path, 'val_raw_512.h5'),
        ],
        [
            os.path.join(
                '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/psp_200107_00_ds_20141002_hela_wt_xyz8nm_as_multiple_scales/step0_datasets/psp0_200108_02_select_test_and_val_cubes',
                'val0_x1390_y742_z345_pad.h5'

            )
        ]
    ]
    val_gt_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    val_gt_filepaths = [
        [
            os.path.join(val_gt_path, 'val_gt_mem.h5'),
            os.path.join(val_gt_path, 'val_gt_dt.h5'),
            os.path.join(val_gt_path, 'val_gt_mask_organelle_insides.h5')
        ]
    ]
    val_raw_channels = []
    for volumes in val_raw_filepaths:
        val_raws_data = []
        for chid, channel in enumerate(volumes):
            if chid == 1:
                # Specifically only load last channel of the membrane prediction
                val_raws_data.append(File(channel, 'r')['data'][..., -1])
            else:
                val_raws_data.append(File(channel, 'r')['data'][:])
        val_raw_channels.append(val_raws_data)
    val_gt_channels = [[File(channel, 'r')['data'][:] for channel in volumes] for volumes in val_gt_filepaths]

if True:
    data_gen_args = dict(
        rotation_range=180,  # Angle in degrees
        shear_range=20,  # Angle in degrees
        zoom_range=[0.8, 1.2],  # [0.75, 1.5]
        horizontal_flip=True,
        vertical_flip=True,
        noise_var_range=1e-1,
        random_smooth_range=[0.6, 1.5],
        smooth_output_sigma=0.5,
        displace_slices_range=2,
        fill_mode='reflect',
        cval=0,
        brightness_range=92,
        contrast_range=(0.5, 2)
    )

    aug_dict_preprocessing = dict(
        smooth_output_sigma=0.5
    )

    train_gen = parallel_data_generator(
        raw_channels=raw_channels,
        gt_channels=gt_channels,
        spacing=(32, 32, 32),
        area_size=(32, 512, 512),
        # area_size=(32, 128, 128),
        target_shape=(64, 64, 64),
        gt_target_shape=(64, 64, 64),
        stop_after_epoch=False,
        aug_dict=data_gen_args,
        transform_ratio=0.9,
        batch_size=2,
        shuffle=True,
        add_pad_mask=False,
        n_workers=16,
        noise_load_dict=None,
        gt_target_channels=None,
        areas_and_spacings=None,
        n_workers_noise=16,
        noise_on_channels=None,
        yield_epoch_info=True
    )

    val_gen = parallel_data_generator(
        raw_channels=val_raw_channels[:1],
        gt_channels=val_gt_channels,
        spacing=(64, 64, 64),
        area_size=(256, 256, 256),
        target_shape=(64, 64, 64),
        gt_target_shape=(64, 64, 64),
        stop_after_epoch=False,
        aug_dict=aug_dict_preprocessing,
        transform_ratio=0.,
        batch_size=1,
        shuffle=False,
        add_pad_mask=False,
        n_workers=16,
        gt_target_channels=None,
        yield_epoch_info=True
    )

model = PiledUnet(
    n_nets=3,
    in_channels=1,
    out_channels=[1, 1, 2],
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
    output_activation='sigmoid'
)
model.cuda()
summary(model, (1, 64, 64, 64))

if not os.path.exists(results_folder):
    os.mkdir(results_folder)

train_model_with_generators(
    model,
    train_gen,
    val_gen,
    n_epochs=300,
    loss_func=CombinedLosses(
        losses=(
            WeightMatrixWeightedBCE(((0.1, 0.9),)),
            WeightMatrixWeightedBCE(((0.2, 0.8),)),
            WeightMatrixWeightedBCE(((0.3, 0.7),)),
            WeightMatrixMSELoss()
        ),
        y_pred_channels=(np.s_[:1], np.s_[1:2], np.s_[2:3], np.s_[3:4]),
        y_true_channels=((0, 2), (0, 2), (0, 2), (1, 2)),
        weigh_losses=np.array([0.15, 0.25, 0.3, 0.3])
    ),
    l2_reg_param=1e-5,
    callbacks=[
        cb_run_model_on_data(
            results_filepath=os.path.join(results_folder, 'improved_result1_{epoch:04d}.h5'),
            raw_channels=val_raw_channels[:1],
            spacing=(32, 32, 32),
            area_size=(64, 256, 256),
            target_shape=(64, 64, 64),
            num_result_channels=4,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            n_workers=16,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            full_dataset_shape=None,
            min_epoch=5
        ),
        cb_run_model_on_data(
            results_filepath=os.path.join(results_folder, 'improved_result2_{epoch:04d}.h5'),
            raw_channels=val_raw_channels[1:],
            spacing=(32, 32, 32),
            area_size=(64, 256, 256),
            target_shape=(64, 64, 64),
            num_result_channels=4,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            n_workers=16,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            full_dataset_shape=None,
            min_epoch=5
        ),
        cb_save_model(
            filepath=os.path.join(results_folder, 'model_{epoch:04d}.h5'),
            min_epoch=5
        )
    ],
    writer_path=os.path.join(results_folder, 'tb')
)
