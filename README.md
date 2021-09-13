# Honest-but-Curious Nets (ACM CCS'21)
**arXiv**: https://arxiv.org/abs/2105.12049

**Title:**
Honest-but-Curious Nets: Sensitive Attributes of Private Inputs Can Be Secretly Coded into the Classifiers' Outputs
(Accepted for ACM SIGSAC Conference on Computer and Communications Security (CCS '21), November 15–19, 2021.)

**Abstract:**

> It is known that deep neural networks, trained for the classification of a non-sensitive target attribute, can reveal somesensitive attributes of their input data; through features of different granularity extracted by the classifier. We take a step forward and show that deep classifiers can be trained to secretly encode a sensitive attribute of users' input data into the classifier's outputs for the target attribute, at inference time. This results in an attack that works even if users have a full white-box view of the classifier, and can keep all internal representations hidden except for the classifier's outputs for the target attribute. We introduce an information-theoretical formulation of such attacks and present efficient empirical implementations for training honest-but-curious (HBC) classifiers based on this formulation: classifiers that can be accurate in predicting their target attribute, but can also exploit their outputs to secretly encode a sensitive attribute. Our evaluations on several tasks in real-world datasets show that a semi-trusted server can build a classifier that is not only perfectly honest but also accurately curious. Our work highlights a vulnerability that can be exploited by malicious machine learning service providers to attack their user's privacy in several seemingly safe scenarios; such as encrypted inferences, computations at the edge, or private knowledge distillation. We conclude by showing the difficulties in distinguishing between standard and HBC classifiers, discussing challenges in defending againstthis vulnerability of deep classifiers, and enumerating related open directions for future studies.

![figure](https://github.com/mmalekzadeh/honest-but-curious-nets/blob/main/figure.jpg?raw=true)


# Dataset Prepartion
## A. UTKFace
1. Download the `Aligned&Cropped Faces` from this link: https://susanqq.github.io/UTKFace/  (It is currently hosted here: https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE).
2. Unzip the file `UTKFace.tar.gz` in the directory `hbcnets/data/`, so you will have a folder named "UTKFace" (i.e. `hbcnets/data/UTKFace`) that includes JPEG images.
3. For the first time, you need to edit `hbcnets/constants.py` and set `DATA_PREPROCESSING = 1`. With this, `main.py` processes images and creates a `.npz` file for you. After that, you have to set `DATA_PREPROCESSING = 0` for all the next runs, unless you change `IMAGE_SIZE` in `hbcnets/constants.py`.

## B. CelebA
1. Download the `Align&Cropped Images` from this link: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  (It is currently hosted here: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8).
2. Create a folder named `celeba` in the directory `hbcnets/data/`.
3. Unzip the file `img_align_celeba.zip` in the folder `celeba`, so you will have `hbcnets/data/celeba/img_align_celeba` that includes JPEG images.
4. Download the folloowing files from the above link and put them in the folder `celeba` (besides folder `img_align_celeba`).
   - list_eval_partition.txt
   - list_attr_celeba.txt
   - identity_CelebA.txt
5. For the first time, you need to edit `hbcnets/constants.py` and set `DATA_PREPROCESSING = 1`. With this, `main.py` processes images and creates a `.npz` file for you. After that, you have to set `DATA_PREPROCESSING = 0` for all the next runs, unless you change `IMAGE_SIZE` in `hbcnets/constants.py`.

# How to Run Experiments

All you need is to run
```
> python main.py 
```
But before that, it will be much helpful if you:
1. Make sure you have read the paper once :-)
2. Open `hbcnets/constants.py` and set up the desired setting. There are multiple parameters that you can play with (see below).
3. Use the arguments in `setting.py` for running your desired experiments.

## Experimental Setup:
In `hbcnets/constants.py` you can find the following parameterts:

- `DATASET`: You can choose `utk_face` or `celeba`.
- `HONEST`: This is the `target` attribut (i.e., `y`) in the paper. Use this to set your desired attribute.
- `K_Y`: After setting `HONEST`, you need to set the size of possible values. This is `Y` in the paper.
- `CURIOUS`: This is the `sensitive` attribut (i.e., `s`) in the paper. Use this to set your desired attribute.
- `K_S`: After setting `CURIOUS`, you need to set the size of possible values. This is `S` in the paper.
- `BETA_X`, `BETA_Y`, `BETA_S`: These are trade-off hyper-parameteres with the same name in the paper.
- `SOFTMAX`: This allow us to decided whtehre we want to release the `soft` outputs (`True`) or the `raw` outputs (`False`).

- `RANDOM_SEED`: You can use this alongside [these lines](https://github.com/mmalekzadeh/honest-but-curious-nets/blob/fcc023098dd894509677a4997fa9db53f7f08ef0/main.py#L12)  in `main.py` to keep your results reproducible. 
- `DATA_PREPROCESSING`: When using a dataset for the first time, you need to set this to 1, after that it can be 0 to save time.
- `IMAGE_SIZE`: The default is 64 (i.e., 64x64), but you can change this to get other resolutions. Notice that if you change this, you need to set `DATA_PREPROCESSING=1` for the first time.

# Citation
Please use:

@inproceedings{malekzadeh2021honest,
  title={Honest-but-Curious Nets: Sensitive Attributes of Private Inputs 
           Can Be Secretly Coded into the Classifiers' Outputs},
  author={Mohammad Malekzadeh and Anastasia Borovykh and Deniz Gündüz},  
  booktitle={Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS '21)},
  year={2021}
}
