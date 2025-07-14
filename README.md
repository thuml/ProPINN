# ProPINN

ProPINN: Demystifying Propagation Failures in Physics-Informed Neural Networks [[Paper]](https://arxiv.org/abs/2502.00803)

This paper provides a formal and in-depth study of the **propagation failure phenomenon in PINNs**, which brings the following progress for PINNs' research:

- **Theoretical understanding:** This paper proves that the root cause of propagation failure is the **lower gradient correlation of PINN models** on nearby collocation points. 
- **Effective backbone:** The theoretical finding also inspires us to present a new PINN architecture, named **ProPINN**, which can effectively unite the gradients of region points for better propagation.
- **Significant Performance:** ProPINN can reliably resolve PINN failure modes and significantly surpass advanced Transformer-based models with **46% relative improvement**.

## Demystify Propagation Failure

**(1) Propagation Failure** is first noticed by [Daw et al. (ICML 2023)](https://arxiv.org/abs/2207.02338), which (we believe) is one of the fundamental issues of PINNs. It describes a weird situation: as shown below, the equation constraint loss (see residual loss) of PINN is sufficiently small, but the approximated solution is still far from the ground truth (see error map).

<p align="center">
<img src=".\pic\visualization.png" height = "150" alt="" align=center />
<br><br>
<b>Figure 1.</b> Illustration of Propagation Failure.
</p>

**(2) FEMs vs. PINNs:** With a detailed comparison between finite element methods (FEMs) and PINNs, we theoretically prove that FEMs are under active propagation, while PINNs are more prone to suffer propagation failure due to their **single-point processing paradigm**. 

<p align="center">
<img src=".\pic\compare.png" height = "150" alt="" align=center />
<br><br>
<b>Figure 2.</b> Comparison between FEMs and PINNs.
</p>

**(3) Theoretical results:** We prove that the **gradient correlation of PINNs on nearby collocation points** is the root cause of propagation failure. As presented in Figure 1, the gradient correlation can also be an accurate criterion to identify propagation failure.

## ProPINN Architecture

The theoretical finding also inspires us to present a new PINN architecture, named ProPINN. ProPINN includes a **multi-region mixing mechanism** to augment the previous single-point processing paradigm, which can effectively unite the gradients of region points for better propagation.

<p align="center">
<img src=".\pic\propinn.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 3.</b> Overall design of ProPINN.
</p>

## Get Started

1. Prepare experiment environments: Python 3.8, PyTorch 1.13.0, CUDA 11.7

```shell
pip install -r requirements.txt
```

2. Run the scripts under the `./scripts` folder:

```shell
bash ./scripts/convection_ProPINN.sh
```

You can also find the pre-trained checkpoints under the `./checkpoints` folder.

## Results

<p align="center">
<img src=".\pic\results.png" height = "250" alt="" align=center />
<br><br>
<b>Table 1.</b> Comparison between ProPINN and previous methods.
</p>

## Case Study

We conduct experiments on standard benchmarks and **challenging fluid dynamics**.

<p align="center">
<img src=".\pic\case.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 4.</b> Showcases on standard benchmarks.
</p>

<p align="center">
<img src=".\pic\fluid.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 5.</b> Showcases on fluid dynamics.
</p>

## Citation

```
@inproceedings{wu2025propinn,
  title={ProPINN: Demystifying Propagation Failures in Physics-Informed Neural Networks},
  author={Haixu Wu and Yuezhou Ma and Hang Zhou and Huikun Weng and Jianmin Wang and Mingsheng Long},
  booktitle={arXiv preprint arXiv:2502.00803},
  year={2025}
}
```

## Contact

If you have any questions or want to use the code, please contact [wuhx23@mails.tsinghua.edu.cn](mailto:wuhx23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base or datasets:

https://github.com/thuml/RoPINN

https://github.com/AdityaLab/pinnsformer
