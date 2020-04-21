import os
from h5py import File
import numpy as np
import scipy.ndimage as ndi
from skimage.util import random_noise
from skimage.transform import downscale_local_mean
from scipy.ndimage import zoom
import sys

from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool


def _build_equally_spaced_volume_list(
        spacing,
        area_size,
        n_volumes,
        transform_ratio,
        set_volume=None
):
    # Components
    spacing = spacing
    half_area_size = (np.array(area_size) / 2).astype(int)

    # This generates the list of all positions, equally spaced and centered around zero
    mg = np.mgrid[
         -half_area_size[0]: half_area_size[0]: spacing[0],
         -half_area_size[1]: half_area_size[1]: spacing[1],
         -half_area_size[2]: half_area_size[2]: spacing[2]
         ].squeeze()
    mg[0] -= int((mg[0].max() + mg[0].min()) / 2)
    mg[1] -= int((mg[1].max() + mg[1].min()) / 2)
    mg[2] -= int((mg[2].max() + mg[2].min()) / 2)
    mg = mg.reshape(3, np.prod(np.array(mg.shape)[1:]))
    positions = mg.swapaxes(0, 1)

    n_transform = int(n_volumes * len(positions) * transform_ratio)
    transform = [True] * n_transform + [False] * (n_volumes * len(positions) - n_transform)
    np.random.shuffle(transform)

    index_array = []

    idx = 0
    for volume in range(n_volumes):
        for position in positions:

            if set_volume:
                index_array.append(
                    [
                        position,
                        set_volume,  # Always volume 0
                        transform[idx]
                    ]
                )
            else:
                index_array.append(
                    [
                        position,
                        volume,  # Always volume 0
                        transform[idx]
                    ]
                )
            idx += 1

    print('Equally spaced volumes:')
    print('    Total samples:       {}'.format(len(positions) * n_volumes))
    print('    Volumes:             {}'.format(n_volumes))
    print('    Transformed samples: {}'.format(n_transform))
    print('Actual size of index_array: {}'.format(len(index_array)))

    return index_array


def _find_bounds(position, crop_shape, full_shape):
    position = np.array(position)
    crop_shape = np.array(crop_shape)
    full_shape = np.array(full_shape)

    # Start and stop in full volume (not considering volume boundaries)
    start = (position - crop_shape / 2 + full_shape / 2).astype('int16')
    stop = start + crop_shape

    # Checking for out of bounds
    start_corrected = start.copy()
    start_corrected[start < 0] = 0
    start_oob = start_corrected - start
    stop_corrected = stop.copy()
    stop_corrected[stop > full_shape] = full_shape[stop > full_shape]
    stop_oob = stop - stop_corrected

    # Making slicings ...
    # ... where to take the data from in the full shape ...
    s_source = np.s_[
               start_corrected[0]: stop_corrected[0],
               start_corrected[1]: stop_corrected[1],
               start_corrected[2]: stop_corrected[2]
               ]
    # ... and where to put it into the crop
    s_target = np.s_[
               start_oob[0]: crop_shape[0] - stop_oob[0],
               start_oob[1]: crop_shape[1] - stop_oob[1],
               start_oob[2]: crop_shape[2] - stop_oob[2]
               ]

    return s_source, s_target


def _load_data_with_padding(
        channels,
        position,
        target_shape,
        auto_pad=False,
        return_pad_mask=False,
        return_shape_only=False,
        auto_pad_z=False
):
    source_shape = np.array(channels[0].shape)

    shape = np.array(target_shape)
    if auto_pad:
        shape[1:] = np.ceil(np.array(target_shape[1:]) * np.sqrt(2) / 2).astype(int) * 2
    if auto_pad_z:
        shape[0] = np.ceil(np.array(target_shape[0]) * np.sqrt(2) / 2).astype(int) * 2

    if return_shape_only:
        return shape.tolist() + [len(channels)]

    s_source, s_target = _find_bounds(position, shape, source_shape)

    # Defines the position of actual target data within the padded data
    pos_in_pad = ((shape - target_shape) / 2).astype(int)
    s_pos_in_pad = np.s_[pos_in_pad[0]: pos_in_pad[0] + target_shape[0],
                   pos_in_pad[1]: pos_in_pad[1] + target_shape[1],
                   pos_in_pad[2]: pos_in_pad[2] + target_shape[2]]

    x = []
    for cid, channel in enumerate(channels):
        # Load the data according to the definitions above

        vol_pad = np.zeros(shape, dtype=channel.dtype)

        vol_pad[s_target] = channel[s_source]
        x.append(vol_pad[..., None])

    if return_pad_mask:
        pad_mask = np.zeros(x[0].shape, dtype=channels[0].dtype)
        pad_mask[s_target] = 255
        x.append(pad_mask)

    x = np.concatenate(x, axis=3)

    return x, s_pos_in_pad


def _load_data_with_padding_old(
        channels,
        position,
        target_shape,
        auto_pad=False,
        return_pad_mask=False,
        return_shape_only=False,
        downsample_output=1
):
    source_shape = np.array(channels[0].shape)

    shape = np.array(target_shape) * downsample_output
    if auto_pad:
        shape[1:] = np.ceil(np.array(target_shape[1:]) * np.sqrt(2) / 2).astype(int) * 2 * downsample_output

    # These are used to load the data with zero padding if necessary
    start_pos = (position - (shape / 2) + (source_shape / 2)).astype('int16')
    stop_pos = start_pos + shape
    start_out_of_bounds = np.zeros(start_pos.shape, dtype='int16')
    start_out_of_bounds[start_pos < 0] = start_pos[start_pos < 0]
    stop_out_of_bounds = stop_pos - source_shape
    stop_out_of_bounds[stop_out_of_bounds < 0] = 0
    start_pos[start_pos < 0] = 0
    stop_pos[stop_out_of_bounds > 0] = source_shape[stop_out_of_bounds > 0]

    if return_shape_only:
        return (stop_pos[0] + stop_out_of_bounds[0] - start_pos[0] - start_out_of_bounds[0],
                stop_pos[1] + stop_out_of_bounds[1] - start_pos[1] - start_out_of_bounds[1],
                stop_pos[2] + stop_out_of_bounds[2] - start_pos[2] - start_out_of_bounds[2], len(channels))

    s_source = np.s_[
               start_pos[0]:stop_pos[0],
               start_pos[1]:stop_pos[1],
               start_pos[2]:stop_pos[2]
               ]
    s_target = np.s_[
               stop_out_of_bounds[0]:stop_pos[0] + stop_out_of_bounds[0] - start_pos[0],
               stop_out_of_bounds[1]:stop_pos[1] + stop_out_of_bounds[1] - start_pos[1],
               stop_out_of_bounds[2]:stop_pos[2] + stop_out_of_bounds[2] - start_pos[2],
               ]

    # Defines the position of actual target data within the padded data
    pos_in_pad = ((shape / downsample_output - target_shape) / 2).astype(int)
    s_pos_in_pad = np.s_[pos_in_pad[0]: pos_in_pad[0] + target_shape[0],
                   pos_in_pad[1]: pos_in_pad[1] + target_shape[1],
                   pos_in_pad[2]: pos_in_pad[2] + target_shape[2]]

    x = []
    for cid, channel in enumerate(channels):
        # Load the data according to the definitions above

        vol_pad = np.zeros(shape, dtype=channel.dtype)

        vol_pad[s_target] = channel[s_source]
        x.append(vol_pad[..., None])

    if return_pad_mask:
        pad_mask = np.zeros(x[0].shape, dtype=channels[0].dtype)
        pad_mask[s_target] = 255
        x.append(pad_mask)

    x = np.concatenate(x, axis=3)
    # if merge_output is not None and x.shape[3] > 1:
    #     if merge_output == 'max':
    #         x = np.max(x, axis=3)[..., None]
    #     else:
    #         raise NotImplementedError

    # if downsample_output > 1:
    #     # x = downscale_local_mean(x, (downsample_output,) * 3 + (1,)).astype(dtype=channels[0].dtype)
    #     x = zoom(x, (1 / downsample_output,) * 3 + (1,))
    # # x = np.zeros(shape, dtype='uint8')

    return x, s_pos_in_pad


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    ndim=3):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:ndim, :ndim]
    final_offset = transform_matrix[:ndim, ndim]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def smooth_output(x, smooth_output_sigma):
    for ch in range(x.shape[3]):
        x[..., ch] = ndi.gaussian_filter(x[..., ch], smooth_output_sigma)

    return x


def preprocessing(x, smooth_output_sigma):
    if smooth_output_sigma > 0:
        x = smooth_output(x, smooth_output_sigma)

    return x


def random_displace_slices(x, displace_slices, fill_mode, cval):
    tx = displace_slices[0]
    ty = displace_slices[1]

    if len(tx) > 0 and len(ty) > 0:

        img_channel_axis = 2

        new_x = []

        for slidx, slc in enumerate(x):
            ctx = tx[slidx]
            cty = ty[slidx]

            shift_matrix = np.array([[1, 0, ctx],
                                     [0, 1, cty],
                                     [0, 0, 1]])

            new_x.append(apply_transform(slc, shift_matrix, img_channel_axis,
                                         fill_mode=fill_mode, cval=cval, ndim=2))

        x = np.array(new_x)

    return x


def add_random_noise(x, noise):
    if noise is not None:
        noisy = x.astype('float32') + (noise.astype('float32') - 128)
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0

        # # Does not multithread, so make sure not to add noise to large data samples
        # noisy = random_noise(
        #     x[crop].astype('float32') / 255,
        #     mode='gaussian',
        #     seed=seed,
        #     clip=True,
        #     mean=0,
        #     var=var
        # )

        x = noisy.astype('uint8')

    return x


def random_brightness_contrast(x, brightness, contrast):
    if brightness > 0 or contrast > 0:
        x = (x.astype('float32') - 128) * contrast + 128
        x += brightness

        x[x > 255] = 255
        x[x < 0] = 0

        x = x.astype('uint8')

    return x


def random_smooth(x, random_smooth):
    """

    a: angle
    s_0: sigma
    s_1: sigma

    exp (-( ((x * sin(a) - y * cos(a))^2/(2*s_0^2)) + ((x * cos(a) + y * sin(a))^2/(2*s_1^2)) ))

    :param x:
    :param random_smooth_range: (s_0, s_1)
    :param seed:
    :return:
    """

    # self.random_smooth_s0_range = 0.3
    # self.random_smooth_s1_range = 1.5
    a = random_smooth[0]
    s_0 = random_smooth[1]
    s_1 = random_smooth[2]

    if s_0 > 0 or s_1 > 0:
        mx, my = np.mgrid[-4:5, -4:5]

        kernel = np.exp(-(((mx * np.sin(a) - my * np.cos(a)) ** 2 / (2 * s_0 ** 2)) + (
                    (mx * np.cos(a) + my * np.sin(a)) ** 2 / (2 * s_1 ** 2))))
        # Like this, the kernel will only work on the x-y planes
        kernel = kernel[None, :, :, None]
        kernel /= kernel.sum()

        x = ndi.filters.convolve(x, kernel)

    return x


def random_transform(
        x,
        rotation,
        shear,
        zoom,
        horizontal_flip,
        vertical_flip,
        depth_flip,
        transpose,
        fill_mode,
        cval
):
    """Randomly augment a single image tensor.

    # Arguments
        x: 3D tensor, single image.
        seed: random seed.

    # Returns
        A randomly transformed version of the input (same shape).
    """

    def flip_axis(x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, o_x],
                                  [0, 0, 1, o_y],
                                  [0, 0, 0, 1]])
        reset_matrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, -o_x],
                                 [0, 0, 1, -o_y],
                                 [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    # x is a single image, so it doesn't have image number at index 0
    img_x_axis = 2
    img_y_axis = 1
    img_z_axis = 0
    img_channel_axis = 3

    # use composition of homographies
    # to generate final transform that needs to be applied
    theta = rotation

    zx = zoom[0]
    zy = zoom[1]

    # Building the transform matrix
    transform_matrix = None
    # if theta != 0:
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, np.cos(theta), -np.sin(theta), 0],
                                [0, np.sin(theta), np.cos(theta), 0],
                                [0, 0, 0, 1]])
    transform_matrix = rotation_matrix

    # if shear != 0:
    shear_matrix = np.array([[1, 0, 0, 0],
                             [0, 1, -np.sin(shear), 0],
                             [0, 0, np.cos(shear), 0],
                             [0, 0, 0, 1]])
    transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

    # if zx != 1 or zy != 1:
    zoom_matrix = np.array([[1, 0, 0, 0],
                            [0, zy, 0, 0],
                            [0, 0, zx, 0],
                            [0, 0, 0, 1]])
    transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[img_x_axis], x.shape[img_y_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=fill_mode, cval=cval)

    if horizontal_flip:
        x = flip_axis(x, img_x_axis)

    if vertical_flip:
        x = flip_axis(x, img_y_axis)

    if depth_flip:
        x = flip_axis(x, img_z_axis)

    if transpose:
        x = x.transpose(*transpose)

    return x


default_aug_dict = dict(
    noise_var_range=0.,
    random_smooth_range=(0, 0),
    displace_slices_range=0,
    smooth_output_sigma=0.,
    rotation_range=0.,
    shear_range=0.,
    zoom_range=(0., 0.),
    horizontal_flip=False,
    vertical_flip=False,
    depth_flip=False,
    fill_mode='nearest',
    cval=0.,
    brightness_range=0.,
    contrast_range=None
)


def _prep_channel(ch, channel_definitions):
    dtype = ch[0].dtype

    new_new_channels = []

    print('Preparing channels ...')

    for gtc in channel_definitions:
        # print('Preparing channels {}'.format(gtc['channel']))

        if type(gtc['channel'][0]) == str:
            if gtc['channel'][0] == 'inv':
                data = [ch[idx] for idx in gtc['channel'][1:]]
                data = 255 - np.sum(data, axis=0)
            else:
                raise ValueError
        else:
            data = [ch[idx] for idx in gtc['channel']]
            data = np.max(data, axis=0)
        data = downscale_local_mean(data, (gtc['downsample'],) * 3).astype(dtype)
        if dtype == 'float32':
            data[data < -0.5] = -1
            data[np.logical_and(data >= -0.5, data < 0.5)] = 0
            data[data >= 0.5] = 1
        elif dtype == 'uint8':
            data[data < 128] = 0
            data[data >= 128] = 255
        else:
            raise NotImplementedError

        new_new_channels.append(data)

    return new_new_channels


def _prepare_channels(channels, channel_definitions, n_workers=1):
    if n_workers == 1:

        new_channels = []

        for ch in channels:
            new_new_channels = _prep_channel(ch, channel_definitions)

            new_channels.append(new_new_channels)

    else:

        with ThreadPoolExecutor(max_workers=n_workers) as tpe:
            tasks = [
                tpe.submit(_prep_channel, ch, channel_definitions)
                for ch in channels
            ]

            new_channels = [task.result() for task in tasks]

    return new_channels


def _pre_load_data(
        index_array,
        raw_channels,
        gt_channels,
        target_size,
        gt_target_size,
        add_pad_mask
):
    xs = []
    ys = []
    s_pads_x = []
    s_pads_y = []

    # Iterate over the positions
    for idx, (position, volume, transform) in enumerate(index_array):

        # Load data
        x, s_pad_x = _load_data_with_padding(raw_channels[volume], position, target_size, auto_pad=transform)

        if gt_channels is not None:
            # Load data
            y, s_pad_y = _load_data_with_padding(gt_channels[volume], position, gt_target_size,
                                                 auto_pad=transform,
                                                 return_pad_mask=add_pad_mask)

            ys.append(y)
            s_pads_y.append(s_pad_y)
        xs.append(x)
        s_pads_x.append(s_pad_x)

    if not ys:
        ys = None
        s_pads_y = None

    return xs, ys, s_pads_x, s_pads_y


import cv2


def _get_random_args(aug_dict, shape, noise_load_dict=None, noise_on_channel=None):
    """
    :param aug_dict:
    :param shape:
    :param noise_load_dict:
        dict(
            filepath='/path/to/noise_file',
            size=number_of_elements
        )
    :return:
    """

    def _load_noise_from_data():
        # Randomly select chunk position
        pos = int(np.random.uniform(0, noise_load_dict['size'] - np.prod(shape)))
        # Load the data
        noise = noise_load_dict['data'][pos: pos + np.prod(shape)]
        # Reshape to match the images
        noise = np.reshape(noise, shape)
        # Get the proper standard variation
        var = np.random.uniform(0, aug_dict['noise_var_range'])
        noise *= (var ** 0.5)
        return noise

    # FIXME this is still the major bottleneck
    if aug_dict['noise_var_range'] > 0:
        if noise_load_dict is not None:
            if 'data' not in noise_load_dict or noise_load_dict['data'] is None:
                print('Trying to load some noise ...')
                if os.path.exists(noise_load_dict['filepath']):
                    with File(noise_load_dict['filepath'], mode='r') as f:
                        noise_load_dict['data'] = f['data'][:]
                else:
                    print('Noise file does not exist, creating it now ... This may take a while ...')
                    print('Generating a lot of noise ...')
                    noise_load_dict['data'] = np.random.normal(0, 1, (noise_load_dict['size'],))
                    print('Make some noise!!!')
                    with File(noise_load_dict['filepath'], mode='w') as f:
                        f.create_dataset('data', data=noise_load_dict['data'])

            noise = _load_noise_from_data()

        else:
            var = np.random.uniform(0, aug_dict['noise_var_range'])
            # noise = np.random.normal(0, var ** 0.5, shape)
            if noise_on_channel is None:
                im = np.zeros((np.prod(shape),))
                noise = cv2.randn(im, 0, var ** 0.5)
                noise = (np.reshape(noise, shape) * 127 + 128).astype('uint8')
            else:
                noise = np.ones(shape, dtype='uint8') * 128
                for ch in noise_on_channel:
                    n_im = np.zeros((int(np.prod(shape) / shape[3]),))
                    n_im = cv2.randn(n_im, 0, var ** 0.5)
                    n_im = np.reshape(n_im, shape[:3])
                    noise[..., ch] = (n_im * 127 + 128).astype('uint8')
    else:
        noise = None

    # print('Noise.shape = {}'.format(noise.shape))

    random_smoothing = [0, 0, 0]
    random_smoothing[0] = np.random.uniform(0, 1) * np.pi
    if aug_dict['random_smooth_range'][0] > 0:
        random_smoothing[1] = np.random.uniform(0, aug_dict['random_smooth_range'][0])
    if aug_dict['random_smooth_range'][1] > 0:
        random_smoothing[2] = np.random.uniform(0, aug_dict['random_smooth_range'][1])

    displace_slices = [[], []]
    if aug_dict['displace_slices_range'] > 0:
        displace_slices[0] = [
            np.random.uniform(-aug_dict['displace_slices_range'], aug_dict['displace_slices_range'])
            for idx in range(shape[2])
        ]
        displace_slices[1] = [
            np.random.uniform(-aug_dict['displace_slices_range'], aug_dict['displace_slices_range'])
            for idx in range(shape[2])
        ]

    brightness = 0
    if aug_dict['brightness_range'] > 0:
        brightness = np.random.uniform(-aug_dict['brightness_range'], aug_dict['brightness_range'])
    contrast = 0
    if aug_dict['contrast_range']:
        if type(aug_dict['contrast_range']) == tuple:
            contrast = np.random.uniform(aug_dict['contrast_range'][0], aug_dict['contrast_range'][1])
        elif type(aug_dict['contrast_range']) == float:
            divide = np.random.random() < 0.5
            contrast = np.random.uniform(1, aug_dict['contrast_range'])
            if divide:
                contrast = 1 / contrast
        elif type(aug_dict['contrast_range']) == dict:
            ctr_settings = aug_dict['contrast_range']
            divide = np.random.random() < ctr_settings['increase_ratio']
            if divide:
                contrast = np.random.uniform(1, ctr_settings['increase'])
            else:
                contrast = 1 / np.random.uniform(1, ctr_settings['decrease'])
        else:
            raise NotImplementedError

    rotation = 0
    if aug_dict['rotation_range']:
        rotation = np.deg2rad(np.random.uniform(-aug_dict['rotation_range'], aug_dict['rotation_range']))

    shear = 0
    if aug_dict['shear_range'] > 0:
        shear = np.deg2rad(np.random.uniform(-aug_dict['shear_range'], aug_dict['shear_range']))

    zoom = [1, 1]
    if aug_dict['zoom_range'][0] != 1 and aug_dict['zoom_range'][1] != 1:
        zoom = list(np.random.uniform(aug_dict['zoom_range'][0], aug_dict['zoom_range'][1], 2))

    horizontal_flip = False
    if aug_dict['horizontal_flip']:
        horizontal_flip = np.random.random() < 0.5

    vertical_flip = False
    if aug_dict['vertical_flip']:
        vertical_flip = np.random.random() < 0.5

    depth_flip = False
    if aug_dict['depth_flip']:
        depth_flip = np.random.random() < 0.5

    transpose = None
    if aug_dict['transpose']:
        transpose = list(range(0, len(shape) - 1))  # The last dim is the channel dim which should not be touched
        np.random.shuffle(transpose)

    return dict(
        noise=noise,
        random_smooth=random_smoothing,
        displace_slices=displace_slices,
        rotation=rotation,
        shear=shear,
        zoom=zoom,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        depth_flip=depth_flip,
        transpose=transpose,
        brightness=brightness,
        contrast=contrast
    )


def _pre_generate_random_values(raw_channels, aug_dict, transform, volume, position, target_shape, noise_load_dict,
                                noise_on_channel, idx):
    # print('Noise for {}'.format(idx))

    if transform:
        shape = _load_data_with_padding(
            channels=raw_channels[volume],
            position=position,
            target_shape=target_shape,
            auto_pad=transform,
            return_shape_only=True,
            auto_pad_z=transform and bool(aug_dict['transpose'])
        )
        random_args = _get_random_args(aug_dict, shape, noise_load_dict=noise_load_dict,
                                       noise_on_channel=noise_on_channel)
    else:
        random_args = None

    return random_args


def _get_batches_of_transformed_samples(
        index_array,
        raw_channels,
        gt_channels,
        target_size,
        gt_target_size,
        gt_target_channels=None,
        yield_xyz=False,
        aug_dict=default_aug_dict,
        aug_args=None,
        add_pad_mask=False,
        batch_no=None,
        batches=None
):
    """
    The steps are:
        1. Simulate bad imaging quality (raw channels, transformed only)
            a) random noise
            b) random smooth
        2. Smooth the input (raw channels, all samples)
        3. Random transformations (all channels, transformed only)

    :param gt_target_channels:

        example

        gt_target_channels=(
            dict(channel=0, downsample=4),
            dict(channel=0, downsample=2),
            dict(channel=1, downsample=2),
            dict(channel=0, downsample=1),
            dict(channel=1, downsample=1),
            dict(channel=2, downsample=1)
        )
    """

    if batch_no is not None:
        assert batches is not None
        sys.stdout.write('\r' + 'Started data generation for batch {}/{}'.format(batch_no + 1, batches))

    if gt_target_channels is not None:
        assert not add_pad_mask

    # Initialize the data volumes for raw and groundtruth
    batch_x = []
    batch_y = []
    positions = []

    # # Iterate over the positions
    for idx, (position, volume, transform) in enumerate(index_array):
        # position, volume, transform = index_array

        aug = aug_args[idx]

        position = position.astype('int16')
        positions.append(position)

        # __________________________________
        # Load and transform the raw volumes

        # # Load data
        x, s_pad_x = _load_data_with_padding(raw_channels[volume],
                                             position, target_size,
                                             auto_pad=transform,
                                             auto_pad_z=transform and bool(aug['transpose']))

        # Simulation of bad imaging quality on samples that are supposed to be transformed
        if transform:

            if aug['transpose']:
                x = x.transpose(*aug['transpose'] + [3])

            x = add_random_noise(x, aug['noise'])
            x = random_smooth(x, aug['random_smooth'])
            x = random_displace_slices(x,
                                       aug['displace_slices'],
                                       aug_dict['fill_mode'],
                                       aug_dict['cval'])
            x = random_brightness_contrast(x, aug['brightness'], aug['contrast'])

        # Smoothing is a general pre-processing of the data and is performed on all samples
        x = preprocessing(x, aug_dict['smooth_output_sigma'])

        # Save the random state here to be able to perform exactly the same transformations on the GT
        rdm_state = np.random.get_state()

        # Random transformations on the raw samples that are set to be transformed
        if transform:
            x = random_transform(x,
                                 aug['rotation'],
                                 aug['shear'],
                                 aug['zoom'],
                                 aug['horizontal_flip'],
                                 aug['vertical_flip'],
                                 aug['depth_flip'],
                                 None,
                                 aug_dict['fill_mode'],
                                 aug_dict['cval'])

        # Crop x to the target size
        batch_x.append(x[s_pad_x])
        # batch_x[idx] = x[s_pad_x]

        # _______________________________
        # Now the same for the GT volumes
        if gt_channels is not None:

            # # Load data
            if gt_target_channels is None:
                y, s_pad_y = _load_data_with_padding(gt_channels[volume], position, gt_target_size,
                                                     auto_pad=transform,
                                                     return_pad_mask=add_pad_mask,
                                                     auto_pad_z=transform and bool(aug['transpose']))

            else:
                data_s_pad_y = np.array([
                    _load_data_with_padding(
                        [gt_channels[volume][gidx]],
                        (position / gtc['downsample']).astype(int),
                        gt_target_size,
                        auto_pad=transform,
                        return_pad_mask=add_pad_mask
                    )
                    for gidx, gtc in enumerate(gt_target_channels)
                ])
                y = data_s_pad_y[:, 0]
                s_pad_y = data_s_pad_y[0, 1]

                y = np.concatenate(y, axis=3)

            # Set the random state which was saved initially to transforming the raw images
            np.random.set_state(rdm_state)

            # Random transformations on the samples that are set to be transformed
            if transform:
                if aug['transpose']:
                    y = y.transpose(*aug['transpose'] + [3])
                y = random_transform(y,
                                     aug['rotation'],
                                     aug['shear'],
                                     aug['zoom'],
                                     aug['horizontal_flip'],
                                     aug['vertical_flip'],
                                     aug['depth_flip'],
                                     None,
                                     aug_dict['fill_mode'],
                                     aug_dict['cval'])

            # Crop y to the target size
            batch_y.append(y[s_pad_y])
            # batch_y[idx] = y[s_pad_y]

    batch_x = np.array(batch_x)
    if gt_channels is not None:
        batch_y = np.array(batch_y)

        # Convert gt to bool
        if batch_y.dtype == 'float32':
            # Let's assume this only happens if there are the three states (-1, 0, 1)
            # now, the first item is true if y = -1, 1 and the second if y = -1, 0
            # That means      [0] and     [1] = -1
            #                 [0] and not [1] =  1
            #             not [0] and     [1] =  0
            batch_y = [batch_y.astype(bool), (batch_y - 1).astype(bool)]
        elif batch_y.dtype == 'uint8':
            batch_y[batch_y < 128] = 0
            batch_y[batch_y >= 128] = 1
            batch_y = batch_y.astype('bool')
        else:
            raise ValueError

    if yield_xyz:
        if gt_channels is not None:
            return batch_x, batch_y, positions
        else:
            return batch_x, positions
    else:
        if gt_channels is not None:
            return batch_x, batch_y
        else:
            return batch_x


def _initialize(
        raw_channels,
        gt_channels,
        spacing=None,
        area_size=None,
        areas_and_spacings=None,
        target_shape=(64, 64, 64),
        gt_target_shape=(64, 64, 64),
        gt_target_channels=None,
        aug_dict=default_aug_dict,
        transform_ratio=0.,
        batch_size=2,
        shuffle=False,
        add_pad_mask=False,
        n_workers=1,
        epoch_idx=0,
        q=None,
        noise_load_dict=None,
        yield_xyz=False,
        n_workers_noise=1,
        noise_on_channels=None
):
    assert ((spacing is not None and area_size is not None)
            or areas_and_spacings is not None), 'The areas and spacings have to be specified either with the ' \
                                                'parameters area_size and spacing, or with areas_and_spacings.'
    if areas_and_spacings is not None:
        assert spacing is None and area_size is None

    n_volumes = len(raw_channels)

    if areas_and_spacings is None:
        # Generate array of transformations
        if type(area_size[0]) == int:
            transformation_array = _build_equally_spaced_volume_list(
                spacing,
                area_size,
                n_volumes,
                transform_ratio
            )
        elif type(area_size[0]) == tuple:
            assert n_volumes == len(area_size)
            transformation_array = []
            for idx, asize in enumerate(area_size):
                transformation_array += _build_equally_spaced_volume_list(
                    spacing,
                    asize,
                    1,
                    transform_ratio,
                    set_volume=idx
                )
        else:
            raise ValueError
    else:
        transformation_array = []
        for aas in areas_and_spacings:
            transformation_array += _build_equally_spaced_volume_list(
                aas['spacing'],
                aas['area_size'],
                1,
                transform_ratio,
                set_volume=aas['vol']
            )

    steps_per_epoch = int(len(transformation_array) / batch_size)
    # steps_per_epoch = len(transformation_array)

    # Shuffle array
    if shuffle:
        np.random.shuffle(transformation_array)

    # Pre-generate random values
    print('Pre-generating random values...')

    if n_workers_noise == 1:
        random_args = []
        for tidx, (position, volume, transform) in enumerate(transformation_array):
            random_args.append(
                _pre_generate_random_values(raw_channels, aug_dict, transform, volume, position, target_shape,
                                            noise_load_dict, noise_on_channels, tidx))
    else:
        with ThreadPoolExecutor(max_workers=n_workers_noise) as tpe:
            tasks = [
                tpe.submit(_pre_generate_random_values,
                           raw_channels, aug_dict, transform, volume, position, target_shape, noise_load_dict,
                           noise_on_channels, tidx)
                for tidx, (position, volume, transform) in enumerate(transformation_array)
            ]
        random_args = [task.result() for task in tasks]

    # Reshape for batch size
    random_args = np.reshape(random_args, (steps_per_epoch, batch_size))
    transformation_array = np.reshape(transformation_array, (steps_per_epoch, batch_size, 3))

    # Generate data for epoch
    if n_workers == 1:

        print('Fetching data with one worker...')
        print(' ')

        results = [
            _get_batches_of_transformed_samples(
                index_array,
                raw_channels,
                gt_channels,
                target_shape,
                gt_target_shape,
                gt_target_channels=gt_target_channels,
                yield_xyz=yield_xyz,
                aug_dict=aug_dict,
                aug_args=random_args[idx],
                add_pad_mask=add_pad_mask,
                batch_no=idx,
                batches=len(transformation_array)
            )
            for idx, index_array in enumerate(transformation_array)
        ]

        print(' ')

    else:

        print('Fetching data with {} workers...'.format(n_workers))
        print(' ')

        # with Pool(processes=n_workers) as p:
        with ThreadPoolExecutor(max_workers=n_workers) as p:

            print('Submitting tasks')
            tasks = [
                p.submit(
                    _get_batches_of_transformed_samples,
                    index_array,
                    raw_channels,
                    gt_channels,
                    target_shape,
                    gt_target_shape,
                    gt_target_channels,
                    yield_xyz,
                    aug_dict,
                    random_args[idx],
                    add_pad_mask,
                    idx,
                    len(transformation_array)
                )
                for idx, index_array in enumerate(transformation_array)
            ]
            results = [task.result() for task in tasks]

        print(' ')

    return results, steps_per_epoch


def parallel_data_generator(
        raw_channels,
        gt_channels,
        spacing=None,
        area_size=None,  # Can now be a tuple of a shape for each input volume
        areas_and_spacings=None,
        target_shape=(64, 64, 64),
        gt_target_shape=(64, 64, 64),
        gt_target_channels=None,
        stop_after_epoch=True,
        aug_dict=default_aug_dict,
        transform_ratio=0.,
        batch_size=2,
        shuffle=False,
        add_pad_mask=False,
        noise_load_dict=None,
        n_workers=1,
        n_workers_noise=1,
        noise_on_channels=None,
        yield_epoch_info=False
):
    """

    :param raw_channels:
    :param gt_channels:
    :param spacing:
    :param area_size:
    :param areas_and_spacings: The new definition of areas and spacings, individually for each volume. Also one volume
        can be defined multiple times to enable different area/spacing combinations.

        example:

        area_and_spacings=(
            dict(vol=0, area_size=(256, 256, 256), spacing=(32, 32, 32)),
            dict(vol=0, area_size=(512, 512, 512), spacing=(128, 128, 128)),
            dict(vol=1, area_size=(512, 512, 512), spacing=(128, 128, 128))
        )

    :param target_shape:
    :param gt_target_shape:
    :param gt_target_channels:

        example:

        gt_target_channels=(
            dict(chanenl=0, downsample=2),
            dict(channel=0, downsample=1),
            dict(channel=1, downsample=1)
        )

    :param stop_after_epoch:
    :param aug_dict:
    :param transform_ratio:
    :param batch_size:
    :param shuffle:
    :param add_pad_mask:
    :param noise_load_dict:
    :param n_workers:
    :return:
    """

    assert ((spacing is not None and area_size is not None)
            or areas_and_spacings is not None), 'The areas and spacings have to be specified either with the ' \
                                                'parameters area_size and spacing, or with areas_and_spacings.'
    if areas_and_spacings is not None:
        assert spacing is None and area_size is None

    # Prepare the gt channels if necessary
    if gt_target_channels is not None:
        gt_channels = _prepare_channels(gt_channels, gt_target_channels, n_workers)

    # Start the generator
    n = 0
    results, steps_per_epoch = _initialize(
        raw_channels=raw_channels,
        gt_channels=gt_channels,
        spacing=spacing,
        area_size=area_size,
        areas_and_spacings=areas_and_spacings,
        target_shape=target_shape,
        gt_target_shape=gt_target_shape,
        gt_target_channels=gt_target_channels,
        aug_dict=aug_dict,
        transform_ratio=transform_ratio,
        batch_size=batch_size,
        shuffle=shuffle,
        add_pad_mask=add_pad_mask,
        n_workers=n_workers,
        noise_load_dict=noise_load_dict,
        n_workers_noise=n_workers_noise,
        noise_on_channels=noise_on_channels
    )

    epoch = 0

    while True:

        if n == 0 and not stop_after_epoch:
            print('Submitting new job')

            p = ThreadPool(processes=1)
            res = p.apply_async(_initialize, (
                raw_channels,
                gt_channels,
                spacing,
                area_size,
                areas_and_spacings,
                target_shape,
                gt_target_shape,
                gt_target_channels,
                aug_dict,
                transform_ratio,
                batch_size,
                shuffle,
                add_pad_mask,
                n_workers - 1,
                epoch,
                None,
                noise_load_dict,
                False,
                n_workers_noise,
                noise_on_channels
            ))

        last_of_epoch = False
        if n == steps_per_epoch - 1:
            last_of_epoch = True
        if stop_after_epoch and n == steps_per_epoch:
            break
        if n == steps_per_epoch:

            n = 0
            print('Fetching results')
            results, _ = res.get()
            print('Joining job')
            epoch += 1

        else:

            # Convert to float
            batch_x = results[n][0]
            batch_y = results[n][1]
            batch_x = batch_x.astype('float32') / 255

            if type(batch_y) is list:
                # Let's assume this only happens if there are the three states (-1, 0, 1)
                # now, the first item is true if y = -1, 1 and the second if y = -1, 0
                # That means      [0] and     [1] = -1
                #                 [0] and not [1] =  1
                #             not [0] and     [1] =  0
                t_batch_y = np.zeros(batch_y[0].shape, dtype='float32')
                t_batch_y[np.logical_and(batch_y[0], batch_y[1])] = -1
                t_batch_y[np.logical_and(batch_y[0], np.logical_not(batch_y[1]))] = 1
                # t_batch_y[not batch_y[0] and batch_y[1]] = 0  <- not needed, right?
                batch_y = t_batch_y
            else:
                batch_y = batch_y.astype('float32')

            if yield_epoch_info:
                yield batch_x, batch_y, epoch, n, last_of_epoch
            else:
                yield batch_x, batch_y
            n += 1


def parallel_test_data_generator(
        raw_channels,
        spacing=(32, 32, 32),
        area_size=(256, 256, 256),
        target_shape=(64, 64, 64),
        smooth_output_sigma=0.5,
        n_workers=1):
    # Start the generator
    n = 0
    results, steps_per_epoch = _initialize(
        raw_channels=raw_channels,
        gt_channels=None,
        spacing=spacing,
        area_size=area_size,
        target_shape=target_shape,
        gt_target_shape=None,
        aug_dict=dict(
            smooth_output_sigma=smooth_output_sigma
        ),
        transform_ratio=0,
        batch_size=1,
        shuffle=False,
        add_pad_mask=False,
        n_workers=n_workers,
        noise_load_dict=None,
        yield_xyz=True
    )

    epoch = 0

    while True:

        if n == steps_per_epoch:
            break

        else:

            # Convert to float
            batch_x = results[n][0]
            batch_x = batch_x.astype('float32') / 255
            # print('min = {}; max = {}'.format(batch_x.min(), batch_x.max()))
            xyz = results[n][1]

            yield batch_x, xyz
            n += 1
