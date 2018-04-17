"""
Video augmentation for deep learning
The framework is general and can be used for any framework such as pytorch, tensorflow or keras
Written by: Quang Vinh Tran
Date created: 04/03/2018
Date modified: 04/17/2018
The code is partially snatched from torch vision transformations
"""

import cv2
import numpy as np

_str_to_cv2_interp = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC
}


class Read(object):
    """Read video clip in color format"""
    def __init__(self, size=None, mode='RGB', interp='bilinear', data_format='channels_last'):
        if size is not None:
            assert isinstance(size, (int, list, tuple)), 'Size must be an integer or a pair of (height, width) or None'
        assert mode in ('RGB', 'BGR'), 'Mode is either "RGB" or "BGR"'
        assert interp in _str_to_cv2_interp, 'Interp are %s' % _str_to_cv2_interp
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.size = None if size is None else (size, size) if isinstance(size, int) else size
        self.mode = mode
        self.interp = _str_to_cv2_interp[interp]

        self.data_format = data_format
        self.channels_last = True if data_format == "channels_last" else False

    def __call__(self, paths):
        clip = []

        for file_name in paths:
            # cv2 read frame image in to H x W x BGR format
            im = cv2.imread(file_name)
            if self.mode == 'RGB':
                im = im[:, :, ::-1]

            if self.size is not None:
                # cv2.resize parameter dsize=(width, height)
                im = cv2.resize(im, dsize=self.size[::-1], interpolation=self.interp)

            clip.append(im)

        clip = np.array(clip)

        if len(clip.shape) != 4:
            clip = np.expand_dims(clip, axis=3)

        if self.channels_last:
            return clip

        return np.transpose(clip, axes=(3, 0, 1, 2))

    def __repr__(self):
        params = '(size={0}, mode={1}, interp={2}, data_format={3})'.format(self.size, self.mode, self.interp,
                                                                            self.data_format)
        return self.__class__.__name__ + params


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        params = self.__class__.__name__ + '('
        for t in self.transforms:
            params += '\n'
            params += '    {0}'.format(t)
        params += '\n)'
        return params


class RandomCrop(object):
    """Randomly crop a voxel from video clip of size height and width at random spatial location"""

    def __init__(self, height=(128, 112, 96, 84), width=(128, 112, 96, 84), data_format='channels_first'):
        assert isinstance(height, (int, list, tuple)), 'Height must be an integer or list of integers'
        assert isinstance(width, (int, list, tuple)), 'Width must be an integer or list of integers'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.height = height if isinstance(height, (list, tuple)) else [height]
        self.width = width if isinstance(width, (list, tuple)) else [width]
        assert len(self.height) == len(self.width), 'Number of crop height and crop width must be equal'

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        i = np.random.randint(len(self.height))
        size = (self.height[i], self.width[i])

        if self.channels_last:
            height, width = clip.shape[1], clip.shape[2]
        else:
            height, width = clip.shape[2], clip.shape[3]

        max_h = height - size[0]
        max_w = width - size[1]

        off_h = np.random.randint(max_h) if max_h > 0 else 0
        off_w = np.random.randint(max_w) if max_w > 0 else 0

        if self.channels_last:
            return clip[:, off_h:off_h + size[0], off_w:off_w + size[1], :]

        return clip[:, :, off_h:off_h + size[0], off_w:off_w + size[1]]

    def __repr__(self):
        params = '(height={0}, width={1}, data_format={2})'.format(self.height, self.width, self.data_format)
        return self.__class__.__name__ + params


class RandomCornerCrop(object):
    """Randomly crop a voxel from video clip of size height and width at one of 5 specific corners"""

    def __init__(self, height=(128, 112, 96, 84), width=(128, 112, 96, 84), data_format='channels_first'):
        assert isinstance(height, (int, list, tuple)), 'Height must be an integer or list of integers'
        assert isinstance(width, (int, list, tuple)), 'Width must be an integer or list of integers'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.height = height if isinstance(height, (list, tuple)) else [height]
        self.width = width if isinstance(width, (list, tuple)) else [width]
        assert len(self.height) == len(self.width), 'Number of crop height and crop width must be equal'

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        i = np.random.randint(len(self.height))
        size = (self.height[i], self.width[i])

        if self.channels_last:
            height, width = clip.shape[1], clip.shape[2]
        else:
            height, width = clip.shape[2], clip.shape[3]

        offsets = [[0, 0], [0, width - size[1]], [height - size[0], 0], [height - size[0], width - size[1]],
                   [np.ceil((height - size[0]) / 2).astype(int), np.ceil((width - size[1]) / 2).astype(int)]]

        off_h, off_w = offsets[np.random.randint(len(offsets))]

        if self.channels_last:
            return clip[:, off_h:off_h + size[0], off_w:off_w + size[1], :]

        return clip[:, :, off_h:off_h + size[0], off_w:off_w + size[1]]

    def __repr__(self):
        params = '(height={0}, width={1}, data_format={2})'.format(self.height, self.width, self.data_format)
        return self.__class__.__name__ + params


class CenterCrop(object):
    """Center crop a voxel from video clip of size height and width"""

    def __init__(self, size, data_format='channels_first'):
        assert isinstance(size, (int, list, tuple)), 'Size must be an integer or list of integers'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.size = (size, size) if isinstance(size, int) else size

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        size = self.size

        if self.channels_last:
            height, width = clip.shape[1], clip.shape[2]
        else:
            height, width = clip.shape[2], clip.shape[3]

        off_h = np.ceil((height - size[0]) / 2).astype(int)
        off_w = np.ceil((width - size[1]) / 2).astype(int)

        if self.channels_last:
            return clip[:, off_h:off_h + size[0], off_w:off_w + size[1], :]

        return clip[:, :, off_h:off_h + size[0], off_w:off_w + size[1]]

    def __repr__(self):
        params = '(size={0}, data_format={1})'.format(self.size, self.data_format)
        return self.__class__.__name__ + params


class FiveCrop(object):
    """Crop 5 voxels from video clip of size height and width"""

    def __init__(self, size, data_format='channels_first'):
        assert isinstance(size, (int, list, tuple)), 'Size must be an integer or list of integers'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.size = (size, size) if isinstance(size, int) else size

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        size = self.size

        if self.channels_last:
            height, width = clip.shape[1], clip.shape[2]
        else:
            height, width = clip.shape[2], clip.shape[3]

        offsets = [[0, 0], [0, width - size[1]], [height - size[0], 0], [height - size[0], width - size[1]],
                   [np.ceil((height - size[0]) / 2).astype(int), np.ceil((width - size[1]) / 2).astype(int)]]

        if self.channels_last:
            tl = clip[:, offsets[0][0]:offsets[0][0] + size[0], offsets[0][1]:offsets[0][1] + size[1], :]
            tr = clip[:, offsets[1][0]:offsets[1][0] + size[0], offsets[1][1]:offsets[1][1] + size[1], :]
            bl = clip[:, offsets[2][0]:offsets[2][0] + size[0], offsets[2][1]:offsets[2][1] + size[1], :]
            br = clip[:, offsets[3][0]:offsets[3][0] + size[0], offsets[3][1]:offsets[3][1] + size[1], :]
            ct = clip[:, offsets[4][0]:offsets[4][0] + size[0], offsets[4][1]:offsets[4][1] + size[1], :]
        else:
            tl = clip[:, :, offsets[0][0]:offsets[0][0] + size[0], offsets[0][1]:offsets[0][1] + size[1]]
            tr = clip[:, :, offsets[1][0]:offsets[1][0] + size[0], offsets[1][1]:offsets[1][1] + size[1]]
            bl = clip[:, :, offsets[2][0]:offsets[2][0] + size[0], offsets[2][1]:offsets[2][1] + size[1]]
            br = clip[:, :, offsets[3][0]:offsets[3][0] + size[0], offsets[3][1]:offsets[3][1] + size[1]]
            ct = clip[:, :, offsets[4][0]:offsets[4][0] + size[0], offsets[4][1]:offsets[4][1] + size[1]]

        return tl, tr, bl, br, ct

    def __repr__(self):
        params = '(size={0}, data_format={1})'.format(self.size, self.data_format)
        return self.__class__.__name__ + params


class TenCrop(object):
    """Crop 10 voxels from video clip of size height and width, i.e, crop 5 then flip"""

    def __init__(self, size, flip='horizontal', data_format='channels_first'):
        assert isinstance(size, (int, list, tuple)), 'Size must be an integer or list of integers'
        assert flip in ('horizontal', 'vertical'), 'Mode is either "horizontal" or "vertical"'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.size = (size, size) if isinstance(size, int) else size
        
        self.flip = flip
        self.vertical_flip = True if self.flip is 'vertical' else False
        
        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        size = self.size

        if self.channels_last:
            # T x H x W x C
            height, width = clip.shape[1], clip.shape[2]
            if self.vertical_flip:
                flip = np.flip(clip, axis=1)
            else:
                flip = np.flip(clip, axis=2)
        else:
            # C x T x H x W
            height, width = clip.shape[2], clip.shape[3]
            if self.vertical_flip:
                flip = np.flip(clip, axis=2)
            else:
                flip = np.flip(clip, axis=3)

        offsets = [[0, 0], [0, width - size[1]], [height - size[0], 0], [height - size[0], width - size[1]],
                   [np.ceil((height - size[0]) / 2).astype(int), np.ceil((width - size[1]) / 2).astype(int)]]

        if self.channels_last:
            c_tl = clip[:, offsets[0][0]:offsets[0][0] + size[0], offsets[0][1]:offsets[0][1] + size[1], :]
            c_tr = clip[:, offsets[1][0]:offsets[1][0] + size[0], offsets[1][1]:offsets[1][1] + size[1], :]
            c_bl = clip[:, offsets[2][0]:offsets[2][0] + size[0], offsets[2][1]:offsets[2][1] + size[1], :]
            c_br = clip[:, offsets[3][0]:offsets[3][0] + size[0], offsets[3][1]:offsets[3][1] + size[1], :]
            c_ct = clip[:, offsets[4][0]:offsets[4][0] + size[0], offsets[4][1]:offsets[4][1] + size[1], :]
            f_tl = flip[:, offsets[0][0]:offsets[0][0] + size[0], offsets[0][1]:offsets[0][1] + size[1], :]
            f_tr = flip[:, offsets[1][0]:offsets[1][0] + size[0], offsets[1][1]:offsets[1][1] + size[1], :]
            f_bl = flip[:, offsets[2][0]:offsets[2][0] + size[0], offsets[2][1]:offsets[2][1] + size[1], :]
            f_br = flip[:, offsets[3][0]:offsets[3][0] + size[0], offsets[3][1]:offsets[3][1] + size[1], :]
            f_ct = flip[:, offsets[4][0]:offsets[4][0] + size[0], offsets[4][1]:offsets[4][1] + size[1], :]
        else:
            c_tl = clip[:, :, offsets[0][0]:offsets[0][0] + size[0], offsets[0][1]:offsets[0][1] + size[1]]
            c_tr = clip[:, :, offsets[1][0]:offsets[1][0] + size[0], offsets[1][1]:offsets[1][1] + size[1]]
            c_bl = clip[:, :, offsets[2][0]:offsets[2][0] + size[0], offsets[2][1]:offsets[2][1] + size[1]]
            c_br = clip[:, :, offsets[3][0]:offsets[3][0] + size[0], offsets[3][1]:offsets[3][1] + size[1]]
            c_ct = clip[:, :, offsets[4][0]:offsets[4][0] + size[0], offsets[4][1]:offsets[4][1] + size[1]]
            f_tl = flip[:, :, offsets[0][0]:offsets[0][0] + size[0], offsets[0][1]:offsets[0][1] + size[1]]
            f_tr = flip[:, :, offsets[1][0]:offsets[1][0] + size[0], offsets[1][1]:offsets[1][1] + size[1]]
            f_bl = flip[:, :, offsets[2][0]:offsets[2][0] + size[0], offsets[2][1]:offsets[2][1] + size[1]]
            f_br = flip[:, :, offsets[3][0]:offsets[3][0] + size[0], offsets[3][1]:offsets[3][1] + size[1]]
            f_ct = flip[:, :, offsets[4][0]:offsets[4][0] + size[0], offsets[4][1]:offsets[4][1] + size[1]]

        return c_tl, c_tr, c_bl, c_br, c_ct, f_tl, f_tr, f_bl, f_br, f_ct

    def __repr__(self):
        params = '(size={0}, flip={1}, data_format={2})'.format(self.size, self.flip, self.data_format)
        return self.__class__.__name__ + params


class Resize(object):
    """Resize video clip to a defined size"""

    def __init__(self, size=(112, 112), interp='bilinear', data_format='channels_first'):
        assert isinstance(size, (int, list, tuple)), 'Size must be an integer or a pair of (height, width)'
        assert interp in _str_to_cv2_interp, 'Interp is %s' % _str_to_cv2_interp
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.size = (size, size) if isinstance(size, int) else size
        self.interp = _str_to_cv2_interp[interp]

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        if not self.channels_last:
            clip = np.transpose(clip, axes=(1, 2, 3, 0))
        # cv2.resize parameter dsize=(width, height)
        out_clip = [cv2.resize(img, dsize=self.size[::-1], interpolation=self.interp) for img in clip]
        out_clip = np.array(out_clip, dtype=clip.dtype)

        if self.channels_last:
            return out_clip

        return np.transpose(out_clip, axes=(3, 0, 1, 2))

    def __repr__(self):
        params = '(size={0}, interp={1}, data_format={2})'.format(self.size, self.interp, self.data_format)
        return self.__class__.__name__ + params


class RandomHorizontalFlip(object):
    """Perform random horizontal flip on video clip"""

    def __init__(self, p=0.5, data_format='channels_first'):
        assert 0 <= p <= 1, 'Value of p must be between 0 and 1'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.p = p

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        if np.random.rand(1, 1).squeeze() > self.p:
            if self.channels_last:
                # T x H x W x C
                return np.flip(clip, axis=2).copy()
            else:
                # C x T x H x W
                return np.flip(clip, axis=3).copy()

        return clip

    def __repr__(self):
        params = '(p={0}, data_format={1})'.format(self.p, self.data_format)
        return self.__class__.__name__ + params


class RandomVerticalFlip(object):
    """Perform random vertical flip on video clip"""

    def __init__(self, p=0.5, data_format='channels_first'):
        assert 0 <= p <= 1, 'Value of p must be between 0 and 1'
        assert data_format in ('channels_first', 'channels_last'), 'Data format is either "channels_first" or "channels_last"'

        self.p = p

        self.data_format = data_format
        self.channels_last = True if data_format == 'channels_last' else False

    def __call__(self, clip):
        if np.random.rand(1, 1).squeeze() > self.p:
            if self.channels_last:
                # T x H x W x C
                return np.flip(clip, axis=1).copy()
            else:
                # C x T x H x W
                return np.flip(clip, axis=2).copy()

        return clip

    def __repr__(self):
        params = '(p={0}, data_format={1})'.format(self.p, self.data_format)
        return self.__class__.__name__ + params


class Montage(object):
    """Create a montage image H x W x C for video clips"""

    def __init__(self, stack=True, input_format='channels_last'):
        assert input_format in ('channels_first', 'channels_last'), 'Input format is either "channels_first" or "channels_last"'

        self.stack = stack
        self.input_format = input_format
        self.channels_last = True if input_format == 'channels_last' else False

    def __call__(self, clips):
        clips = np.array(clips)

        if len(clips.shape) == 4:
            clips = np.expand_dims(clips, axis=0)

        if not self.channels_last:
            clips = np.transpose(clips, axes=(0, 2, 3, 4, 1))

        imgs = [np.hstack(clip) for clip in clips]

        if self.stack:
            return np.vstack(imgs).squeeze()

        return imgs

    def __repr__(self):
        params = '(stack={0}, data_format={1})'.format(self.stack, self.input_format)
        return self.__class__.__name__ + params


class ToTensor(object):
    """Convert a numpy array clip in range [0, 255] to a numpy array clip channels_first format in range [0.0, 1.0]"""

    def __init__(self, norm=255, input_format='channels_first'):
        assert input_format in ('channels_first', 'channels_last'), 'Input format is either "channels_first" or "channels_last"'
        self.norm = norm

        self.input_format = input_format
        self.input_channels_last = True if input_format == 'channels_last' else False

    def __call__(self, clip):
        clip = np.transpose(clip, axes=(3, 0, 1, 2)) if self.input_channels_last else clip

        if clip.dtype == np.uint8:
            return clip.astype(dtype=np.float32).__div__(self.norm)

        return clip

    def __repr__(self):
        params = '(norm={0}, input_format={1})'.format(self.norm, self.input_format)
        return self.__class__.__name__ + params


class ToBatchTensor(object):
    """Convert numpy array clips in range [0, 255] to numpy array clips channels_first format in range [0.0, 1.0]"""

    def __init__(self, norm=255, input_format='channels_first'):
        assert input_format in ('channels_first', 'channels_last'), 'Input format is either "channels_first" or "channels_last"'
        self.norm = norm

        self.input_format = input_format
        self.input_channels_last = True if input_format == 'channels_last' else False

    def __call__(self, clip):
        clip = np.transpose(clip, axes=(0, 4, 1, 2, 3)) if self.input_channels_last else clip

        if clip.dtype == np.uint8:
            return clip.astype(dtype=np.float32).__div__(self.norm)

        return clip

    def __repr__(self):
        params = '(norm={0}, input_format={1})'.format(self.norm, self.input_format)
        return self.__class__.__name__ + params


class Normalize(object):
    """Normalize a clip with mean and standard deviation (z-score normalization)"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        for t, m, s in zip(clip, self.mean, self.std):
            t.__sub__(m).__div__(s)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def _demo_read_clip():
    """
    Demo reading video clips then performing random crop and flip for augmentation
    """
    data_format = 'channels_first'
    frame_dir = './frames/'
    paths = [frame_dir + 'frm_%06d.jpg' % (f + 1) for f in range(0, 0 + 16)]

    # List of transformations used
    read_clip = Read(size=(128, 171), mode='RGB', interp='bilinear', data_format=data_format)
    montage = Montage(stack=True, data_format=data_format)
    test_transforms = Compose([RandomCrop(data_format=data_format),
                               RandomHorizontalFlip(p=0.5, data_format=data_format),
                               RandomVerticalFlip(p=0.5, data_format=data_format),
                               Resize(size=112, data_format=data_format),
                               ToTensor(input_format=data_format)])
    # verbose
    print read_clip
    for t in test_transforms.transforms:
        print t
    print montage

    # time it
    from time import time
    time_it = time()

    clips = []
    for _ in range(50):
        clip = read_clip(paths)
        clip = test_transforms(clip)
        clips.append(clip)

    print 'Time:', time() - time_it

    # Get the big image using montage
    img = montage(clips)
    print 'Min:', img.min(), 'Max:', img.max()

    # Saving to a single images, each row is each clip
    cv2.imwrite('clips.png', np.asarray(img.__mul__(255)[:, :, ::-1], dtype=np.uint8))


if __name__ == '__main__':
    _demo_read_clip()
