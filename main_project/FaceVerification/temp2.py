import glob
import os
from sklearn.model_selection import train_test_split
import shutil

pasmoi_paths = glob.glob("../../dataset/new_img_MBA_2023/pasmoi_crop/*")
lfw_paths = glob.glob("../../dataset/lfw/lfw_funneled/*/*")
moi_new_paths = glob.glob("../../dataset/new_img_MBA_2023/moi_crop/*")
moi_old_paths = glob.glob("../../dataset/new_img_MBA_2023/moi_crop_old/*")
moi_paths = moi_new_paths + moi_old_paths


pasmoi_paths_train, pasmoi_paths_val, _, _ = train_test_split(pasmoi_paths, range(len(pasmoi_paths)), test_size=0.33)
moi_paths_train, moi_paths_val, _, _ = train_test_split(moi_paths, range(len(moi_paths)), test_size=0.33)
lfw_paths_train, lfw_paths_val, _, _ = train_test_split(lfw_paths, range(len(lfw_paths)), test_size=0.33)
lfw_paths_train = lfw_paths_train[:2500]
lfw_paths_val = lfw_paths_val[:500]


pasmoi_paths_train = pasmoi_paths_train + lfw_paths_train
pasmoi_paths_val = pasmoi_paths_val + lfw_paths_val

for img in moi_paths_train:
    shutil.copy(img, "../../dataset/new_img_MBA_2023/train_set/moi")

for img in pasmoi_paths_train:
    shutil.copy(img, "../../dataset/new_img_MBA_2023/train_set/pasmoi")

for img in moi_paths_val:
    shutil.copy(img, "../../dataset/new_img_MBA_2023/val_set/moi")

for img in pasmoi_paths_val:
    shutil.copy(img, "../../dataset/new_img_MBA_2023/val_set/pasmoi")