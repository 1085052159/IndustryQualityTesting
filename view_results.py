import os
import shutil
from glob import glob

from tqdm import tqdm


exp_name = "DJI_20230711104903_0005_V"
dst_path = "view/%s/right" % exp_name
os.makedirs(dst_path, exist_ok=True)
root_path = "tmp_results/%s" % exp_name
img_dirs = os.listdir(root_path)
for img_dir in tqdm(img_dirs):
    result_path = "%s/%s/result" % (root_path, img_dir)
    result_img_names = glob("%s/*" % result_path)
    for src_name in result_img_names:
        dst_name = "%s/%s" % (dst_path, os.path.basename(src_name))
        shutil.copy(src_name, dst_name)
