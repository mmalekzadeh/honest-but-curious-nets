# Datsets

## UTKFace
1. Download the `Aligned&Cropped Faces` from this link: https://susanqq.github.io/UTKFace/  (It is currently hosted here: https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE).
2. Unzip the file `UTKFace.tar.gz` in this directory, so you will have a folder named "UTKFace" in this directory (i.e. `hbcnets/data/UTKFace`) that includes JPEG images.
3. For the first time, you need to edit `hbcnets/constants.py` and set `DATA_PREPROCESSING = 1`. With this, `main.py` processes images and creates a `.npz` file for you. After that, you have to set `DATA_PREPROCESSING = 0` for all the next runs, unless you change `IMAGE_SIZE` in `hbcnets/constants.py`.

## CelebA
1. Download the `Align&Cropped Images` from this link: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  (It is currently hosted here: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8).
2. Create a folder named `celeba` in this directory.
3. Unzip the file `img_align_celeba.zip` in the folder `celeba`, so you will have `hbcnets/data/celeba/img_align_celeba` that includes JPEG images.
4. Download the folloowing files from the above link and put them in the folder `celeba` (besides folder `img_align_celeba`).
   - list_eval_partition.txt
   - list_attr_celeba.txt
   - identity_CelebA.txt
5. For the first time, you need to edit `hbcnets/constants.py` and set `DATA_PREPROCESSING = 1`. With this, `main.py` processes images and creates a `.npz` file for you. After that, you have to set `DATA_PREPROCESSING = 0` for all the next runs, unless you change `IMAGE_SIZE` in `hbcnets/constants.py`.
