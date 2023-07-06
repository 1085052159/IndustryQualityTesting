from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PointerDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'pointer')
    
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
    def __init__(self, **kwargs):
        super(PointerDataset, self).__init__(
            seg_map_suffix='.png',
            **kwargs)
