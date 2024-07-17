# Bridge Past and Future
**(ECCV 2024) Official repository of paper "Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection"**



## 🔥 News

- 2024/7/17: Our arxiv paper can be found [here](http://arxiv.org/abs/2407.11499)



## 🔍 Overview

![image-20240717145206666](assets/image-20240717145206666.png)

**The overall framework of our method.** The top side illustrates the Bridge Past and Future (BPF) procedure, which identifies objects of past classes and excludes several potential objects of future classes to ensure consistent optimization during the entire training process. The bottom side shows the Distillation with Future (DwF) process, which employs both the old model $\mathcal{M}_{t-1}$ adept at detecting old categories and the interim model $\mathcal{M}_{t}^{im}$ trained on $\mathcal{D}_t$ and specialized in new categories to conduct a comprehensive distillation across all categories for the current model $\mathcal{M}_t$.



![visualization1](assets/visualization1.png)

**Visualization of Bridge Past and Future.** Boxes in $\textcolor{red}{\text{red}}$ represent the ground truth in the current stage. **(a)** In Bridge the Past, we effectively constructed pseudo labels of past classes. **(b)** In Bridge the Future, salient objects (marked in $\textcolor{green}{\text{green}}$ boxes) can be easily detected from the attention maps and are excluded from the background regions. Best viewed in color.





## 📝 TODO List
- [ ] Release the code.
- [ ] Release the checkpoint.
- [ ] Release the training script.



## 📖 Implementation







## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
   mo2024bridge,
   title={Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection},
   author={Mo, Qijie and Gao, Yipeng and Fu, Shenghao and Yan, Junkai and Wu, Ancong and Zheng, Wei-Shi},
   booktitle={ECCV},
   year={2024},
}
```
