
from torchsummary import summary
import torch as t
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from pytorch_tools.run_models import predict_model_from_h5_parallel_generator

from matplotlib import pyplot as plt


def cb_save_model(
        filepath,
        min_epoch=0
):

    def run(model, epoch):
        if epoch >= min_epoch:
            print('Saving model ...')
            t.save(model.state_dict(), filepath.format(epoch=epoch))
        else:
            print('Not saving model: Minimum number of epochs not reached.')

    return run


def cb_run_model_on_data(
        results_filepath,
        raw_channels,
        spacing,
        area_size,
        target_shape,
        num_result_channels,
        smooth_output_sigma,
        n_workers,
        compute_empty_volumes,
        thresh,
        write_at_area,
        offset,
        full_dataset_shape,
        min_epoch=0
):

    def run(model, epoch):
        if epoch >= min_epoch:
            print('Running model on data ...')
            predict_model_from_h5_parallel_generator(
                model=model,
                results_filepath=results_filepath.format(epoch=epoch),
                raw_channels=raw_channels,
                spacing=spacing,
                area_size=area_size,
                target_shape=target_shape,
                num_result_channels=num_result_channels,
                smooth_output_sigma=smooth_output_sigma,
                n_workers=n_workers,
                compute_empty_volumes=compute_empty_volumes,
                thresh=thresh,
                write_at_area=write_at_area,
                offset=offset,
                full_dataset_shape=full_dataset_shape
            )
        else:
            print('Not running model on data: Minimum number of epochs not reached.')

    return run


def train_model_with_generators(
        model,
        train_generator,
        val_generator,
        n_epochs,
        loss_func,
        optimizer=None,
        l2_reg_param=1e-3,
        callbacks=None,
        writer_path=None
):

    if writer_path is not None:
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        writer_train = SummaryWriter(os.path.join(writer_path, 'train'))
        writer_val = SummaryWriter(os.path.join(writer_path, 'val'))
    else:
        writer_train = None
        writer_val = None

    def _on_epoch_end(model, best_val_loss):
        print('------------------------------------------')
        print('Epoch ended, evaluating model ...')

        # Evaluation at the end of an epoch
        with t.no_grad():
            model.eval()

            sum_loss = 0.
            eval_not_done = True
            it_idx = 0.

            while eval_not_done:

                valx, valy, val_epoch, valn, loe = next(val_generator)

                # Forward pass
                tensx = t.tensor(np.moveaxis(valx, 4, 1), dtype=t.float32).cuda()
                predy = model(tensx)

                # Compute loss
                tensy = t.tensor(np.moveaxis(valy, 4, 1), dtype=t.float32).cuda()
                loss = loss_func(predy, tensy)
                sum_loss += loss

                it_idx += 1

                if loe:
                    break

            val_loss = sum_loss / (it_idx + 1)

            if writer_val:
                writer_val.add_scalar('losses/loss', val_loss, epoch)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                improvement = True
            else:
                improvement = False

        print('Epoch: {} | Train loss: {} | Val loss: {} | Best val loss: {}'.format(
            epoch, train_loss, val_loss, best_val_loss)
        )

        print('------------------------------------------')

        return best_val_loss, improvement

    if optimizer is None:
        # optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #                          lr=1e-4, betas=(0.9, 0.999), weight_decay=l2_reg_param)
        optimizer = t.optim.Adam(model.parameters(),
                                 lr=1e-4, betas=(0.9, 0.999), weight_decay=l2_reg_param)
        
    print(optimizer)
    
    best_val_loss = None
    print('__________________________________________')
    print('Epoch = 0')
    sum_loss = 0.

    # The train loop
    model.train()
    for it_idx, (x, y, epoch, n, loe) in enumerate(train_generator):

        optimizer.zero_grad()
        
        # Forward pass
        tensx = t.tensor(np.moveaxis(x, 4, 1), dtype=t.float32).cuda()
        predy = model(tensx)

        # Compute loss
        tensy = t.tensor(np.moveaxis(y, 4, 1), dtype=t.float32).cuda()
        loss = loss_func(predy, tensy)

        # Back propagation
        loss.backward()

        optimizer.step()
        sum_loss += loss.item()

        train_loss = sum_loss / (n + 1)
        print('Iteration = {} | Loss = {} | Average loss = {}'.format(n + 1, loss.item(), train_loss))

        if loe:

            if writer_train:
                writer_train.add_scalar('losses/loss', train_loss, epoch)

            # Evaluate previous epoch
            best_val_loss, improvement = _on_epoch_end(model, best_val_loss)

            # Callbacks when the model improved
            if callbacks is not None and improvement:
                with t.no_grad():
                    model.eval()
                    print('Validation loss improved! Computing callbacks ...')
                    with t.no_grad():
                        for callback in callbacks:
                            callback(model, epoch)

            # Break if specified number of epochs is reached
            if epoch + 1 == n_epochs:
                break

            # Initialize new epoch
            print('__________________________________________')
            print('Epoch = {}'.format(epoch + 1))

            sum_loss = 0.
            model.train()


if __name__ == '__main__':

    from pytorch.pytorch_tools.data_generation import parallel_data_generator
    import os
    from h5py import File
    from pytorch.pytorch_tools.piled_unets import PiledUnet
    from pytorch.pytorch_tools.losses import WeightMatrixWeightedBCE, CombinedLosses

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
            area_size=(32, 128, 128),  # (32, 512, 512),
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
            output_activation='sigmoid'
        )
    model.cuda()
    summary(model, (1, 64, 64, 64))

    results_folder = '/g/schwab/hennies/tmp/pytorch_test2'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    
    train_model_with_generators(
        model,
        train_gen,
        val_gen,
        n_epochs=100,
        loss_func=CombinedLosses(
            losses=(
                WeightMatrixWeightedBCE(((0.1, 0.9),)),
                WeightMatrixWeightedBCE(((0.2, 0.8),)),
                WeightMatrixWeightedBCE(((0.3, 0.7),))),
            y_pred_channels=(np.s_[:1], np.s_[1:2], np.s_[2:3]),
            y_true_channels=(np.s_[:], np.s_[:], np.s_[:]),
            weigh_losses=np.array([0.2, 0.3, 0.5])
        ),
        l2_reg_param=1e-5,
        callbacks=[
            cb_run_model_on_data(
                results_filepath=os.path.join(results_folder, 'improved_result2_{epoch:04d}.h5'),
                raw_channels=val_raw_channels[:1],
                spacing=(32, 32, 32),
                area_size=(64, 256, 256),
                target_shape=(64, 64, 64),
                num_result_channels=3,
                smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
                n_workers=16,
                compute_empty_volumes=True,
                thresh=None,
                write_at_area=False,
                offset=None,
                full_dataset_shape=None
            ),
            cb_save_model(
                filepath=os.path.join(results_folder, 'model_{epoch:04d}.h5')
            )
        ],
        writer_path=os.path.join(results_folder, 'run')
    )
