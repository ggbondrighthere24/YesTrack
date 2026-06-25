<h1 align="center">
  <i>YesTrack: Referring Multi-Object Tracking<br>
  via MLLM-based Yes/No Reasoning</i>
</h1>

<p align="center">
  Quansheng Hu<sup>1</sup>,&nbsp;
  Qin Sun<sup>1</sup>,&nbsp;
  Qiansen Dai<sup>1</sup>,&nbsp;
  Jin Ding<sup>1</sup>,&nbsp;<br>
  Wan Zhang<sup>1</sup>,&nbsp;
  Xue Zhou<sup>2,1,*</sup>,&nbsp;
  Jianxiao Zou<sup>2</sup>,&nbsp;<br>
  <sup>1</sup> University of Electronic Science and Technology of China, Chengdu, China<br>
  <sup>2</sup> Shenzhen Institute for Advanced Study, UESTC<br>
  <sup>*</sup> Corresponding author<br>
  📧 Primary Contact: huhansan@std.uestc.edu.cn
</p>

<p align="center">
  <img alt="Static Badge" src="https://img.shields.io/badge/ECCV 2026-%F0%9F%92%A1-%235E86C1?style=flat-square">
</p>


## :mag: Overview

**TL; DR.** We propose YesTrack, a two-stage framework that ***reformulates referring multi-object tracking as direct Yes/No reasoning with multimodal large language models***. By avoiding autoregressive caption generation and additional text-matching modules, YesTrack provides a direct and efficient solution for referring. Temporal Confidence Prior (TCP) and Temporal Reference Propagation (TRP) further improve temporal reliability, while YesTrack-MOT extends this discriminative paradigm to generic multi-object tracking.

![Overview](./assets/overview.png)


## :fire: News

- <span style="font-variant-numeric: tabular-nums;">**2026.07.12**</span>: The verified codebase, ~~requirements~~ *(completed ahead of schedule)*, and basic usage instructions are planned for release before this date :construction:.
- <span style="font-variant-numeric: tabular-nums;">**2026.06.30**</span>: The initial [requirements](./requirements.txt) file is uploaded ahead of schedule :tada:.
- <span style="font-variant-numeric: tabular-nums;">**2026.06.29**</span>: The initial codebase is released :tada:. This repository was directly converted and organized by Codex from my original project files and is still under verification :construction:. I have been a little busy lately, so updates may be slower. For urgent code issues before July 12, please contact [huhansan@std.uestc.edu.cn](mailto:huhansan@std.uestc.edu.cn) :email:. The original version can be provided if needed, although it may be somewhat messy.
