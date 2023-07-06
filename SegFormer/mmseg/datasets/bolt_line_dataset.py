from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BoltLineDataset(CustomDataset):
    """BoltLine dataset.

    In segmentation map annotation for BoltLine, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'line')
    
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
    def __init__(self, **kwargs):
        super(BoltLineDataset, self).__init__(
            seg_map_suffix='.png',
            **kwargs)
