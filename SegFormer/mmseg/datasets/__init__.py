from .ade import ADE20KDataset
from .bolt_line_dataset import BoltLineDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .pascal_context import PascalContextDataset
from .pointer_dataset import PointerDataset
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .mapillary import MapillaryDataset
from .cocostuff import CocoStuff

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset', 'STAREDataset', 'MapillaryDataset', 'CocoStuff',
    'PointerDataset', 'BoltLineDataset'
]
