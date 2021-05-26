# Honest-but-Curious Nets
**arXiv**: https://arxiv.org/abs/2105.12049

**Title:**
> Honest-but-Curious Nets: Sensitive Attributes of Private Inputs can be Secretly Coded into the Entropy of Classifiers' Outputs

**Abstract:**

>It is known that deep neural networks, trained for the classification of a non-sensitive target attribute, can reveal sensitive attributes of their input data; through features of different granularity extracted by the classifier. We, taking a step forward, show that deep classifiers can be trained to secretly encode a sensitive attribute of users' input data, at inference time, into the classifier's outputs for the target attribute. An attack that works even if users have a white-box view of the classifier, and can keep all internal representations hidden except for the classifier's estimation of the target attribute. We introduce an information-theoretical formulation of such adversaries and present efficient empirical implementations for training honest-but-curious (HBC) classifiers based on this formulation: deep models that can be accurate in predicting the target attribute, but also can utilize their outputs to secretly encode a sensitive attribute. Our evaluations on several tasks in real-world datasets show that a semi-trusted server can build a classifier that is not only perfectly honest but also accurately curious. Our work highlights a vulnerability that can be exploited by malicious machine learning service providers to attack their user’s privacy in several seemingly safe scenarios; such as encrypted inferences, computations at the edge, or private knowledge distillation. We conclude by showing the difficulties in distinguishing between standard and HBC classifiers and discussing potential proactive defenses against this vulnerability of deep classifiers.

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
2. Open `hbcnets/constants.py` and set up the desired environent. There are multiple parameters there that you can play with.
3. Use the arguments in `setting.py` for running your desired experiments.


# Citation
Please use:
```
@article{malekzadeh2021honest,
  title={Honest-but-Curious Nets: Sensitive Attributes of Private Inputs 
         can be Secretly Coded into the Entropy of Classifiers' Outputs},
  author={Malekzadeh, Mohammad and Borovykh, Anastasia and Gündüz, Deniz},
  journal={},
  year={2021}
}
```
