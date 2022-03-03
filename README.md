# AGKD-BML
Pytorch implementation of ICCV 2021 paper "[AGKD-BML: Defense Against Adversarial Attack by Attention Guided Knowledge Distillation and Bi-directional Metric Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_AGKD-BML_Defense_Against_Adversarial_Attack_by_Attention_Guided_Knowledge_Distillation_ICCV_2021_paper.pdf)".

## Requirements

- Python 3.7
- Pytorch 1.3.1

## Usage

To train the model with AGKD-BML:
```
python main.py
```
## Trained model

A WideResNet-28-10 model trained by AGKD-BML on CIFAR-10 can be found [here](https://drive.google.com/file/d/1JAtLps8xqGTNCidjDBbuSQ4ljNXLt4jm/view?usp=sharing).

## Citing this work

If you find this work is useful, please cite the paper:
```
@InProceedings{Wang_2021_ICCV,
    title     = {AGKD-BML: Defense Against Adversarial Attack by Attention Guided Knowledge Distillation and Bi-Directional Metric Learning},
    author    = {Wang, Hong and Deng, Yuefan and Yoo, Shinjae and Ling, Haibin and Lin, Yuewei},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```
