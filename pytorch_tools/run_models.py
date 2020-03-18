import h5py
import os
import numpy as np
import sys
import torch as t
from pytorch_tools.data_generation import parallel_test_data_generator


def predict_model_from_h5_parallel_generator(
        model,
        results_filepath,
        raw_channels,
        spacing,
        area_size,
        target_shape,
        num_result_channels,
        smooth_output_sigma=0.5,
        n_workers=16,
        compute_empty_volumes=True,
        thresh=0,
        write_at_area=False,
        offset=None,
        full_dataset_shape=None,
        write_in_and_out=False  # For debugging
):
    model.eval()

    print('offset = {}'.format(offset))
    print('write_at_area = {}'.format(write_at_area))
    print('full_dataset_shape = {}'.format(full_dataset_shape))

    def _write_result(dataset, result, position, spacing):

        spacing = np.array(spacing)
        spacing_half = (spacing / 2).astype(int)
        shape = np.array(dataset.shape[:3])
        shape_half = (shape / 2).astype(int)
        result_shape = np.array(result.shape[:3])
        result_shape_half = (result_shape / 2).astype(int)

        # Pre-crop the result
        start_crop = result_shape_half - spacing_half
        stop_crop = result_shape_half + spacing_half
        s_pre_crop = np.s_[
                     start_crop[0]: stop_crop[0],
                     start_crop[1]: stop_crop[1],
                     start_crop[2]: stop_crop[2]
                     ]
        result_cropped = result[s_pre_crop]

        # All the shapes and positions
        result_shape = np.array(result_cropped.shape[:3])
        result_shape_half = (result_shape / 2).astype(int)
        position = np.array(position)

        start_pos = position + shape_half - result_shape_half
        stop_pos = start_pos + spacing
        # print('')
        # print('Before correction ...')
        # print('start_pos = {}'.format(start_pos))
        # print('stop_pos = {}'.format(stop_pos))
        start_out_of_bounds = np.zeros(start_pos.shape, dtype=start_pos.dtype)
        start_out_of_bounds[start_pos < 0] = start_pos[start_pos < 0]
        stop_out_of_bounds = stop_pos - shape
        stop_out_of_bounds[stop_out_of_bounds < 0] = 0
        start_pos[start_pos < 0] = 0
        stop_pos[stop_out_of_bounds > 0] = shape[stop_out_of_bounds > 0]
        # print('After correction ...')
        # print('start_pos = {}'.format(start_pos))
        # print('stop_pos = {}'.format(stop_pos))

        # For the results volume
        s_source = np.s_[
                   -start_out_of_bounds[0]:stop_pos[0] - start_pos[0] - start_out_of_bounds[0],
                   -start_out_of_bounds[1]:stop_pos[1] - start_pos[1] - start_out_of_bounds[1],
                   -start_out_of_bounds[2]:stop_pos[2] - start_pos[2] - start_out_of_bounds[2],
                   :
                   ]
        # For the target dataset
        s_target = np.s_[
                   start_pos[0]:stop_pos[0],
                   start_pos[1]:stop_pos[1],
                   start_pos[2]:stop_pos[2],
                   :
                   ]

        dataset[s_target] = (result_cropped * 255).astype('uint8')[s_source]

    if offset is None:
        offset = (0, 0, 0)

    # Generate results file
    if not write_at_area:
        with h5py.File(results_filepath, 'w') as f:
            f.create_dataset('data', shape=tuple(area_size) + (num_result_channels,), dtype='uint8', compression='gzip',
                             chunks=(32, 32, 32, 1))
    else:
        if not os.path.exists(results_filepath):
            with h5py.File(results_filepath, 'w') as f:
                f.create_dataset('data', shape=tuple(full_dataset_shape) + (num_result_channels,), dtype='uint8',
                                 compression='gzip')

    # Generate debug input and results files
    if write_in_and_out:
        f_ins = h5py.File(os.path.splitext(results_filepath)[0] + '_ins.h5', 'w')
        f_outs = h5py.File(os.path.splitext(results_filepath)[0] + '_outs.h5', 'w')
        folder_ins = os.path.splitext(results_filepath)[0] + '_ins'
        folder_outs = os.path.splitext(results_filepath)[0] + '_outs'
        if not os.path.exists(folder_ins):
            os.mkdir(folder_ins)
        if not os.path.exists(folder_outs):
            os.mkdir(folder_outs)

    for idx, element in enumerate(parallel_test_data_generator(
            raw_channels=raw_channels,
            spacing=spacing,
            area_size=area_size,
            target_shape=target_shape,
            smooth_output_sigma=smooth_output_sigma,
            n_workers=n_workers
    )):
        im = element[0]
        xyz = element[1][0] + np.array(offset)

        # xyz = np.array(xyz) + (np.array(source_size) / 2).astype(int) - (np.array(spacing) / 2).astype(int)
        x = xyz[2]
        y = xyz[1]
        z = xyz[0]

        sys.stdout.write('\r' + 'x = {}; y = {}, z = {}'.format(x, y, z))

        if compute_empty_volumes or (im < thresh).sum():

            im = np.moveaxis(im, 4, 1)

            if write_in_and_out:
                f_ins.create_dataset('{}_{}_{}'.format(z, y, x), data=im)
                np.savez(os.path.join(folder_ins, '{}_{}_{}.npz'.format(z, y, x)), im)

            imx = t.tensor(im, dtype=t.float32).cuda()

            result = model(imx)
            result = result.cpu().numpy()

            if write_in_and_out:
                f_outs.create_dataset('{}_{}_{}'.format(z, y, x), data=result)
                np.savez(os.path.join(folder_outs, '{}_{}_{}.npz'.format(z, y, x)), result)

            result = np.moveaxis(result, 1, 4)

            # overlap = np.array(result.shape[1:4]) - np.array(spacing)
            #
            with h5py.File(results_filepath, 'a') as f:
                # write_test_h5_generator_result(f['data'], result, x, y, z, overlap, ndim=ndim)
                _write_result(f['data'], result[0, :], xyz, spacing)

        else:

            print(' skipped...')
