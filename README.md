<p align="center">
  <img src="Media/TinyDrive_v11.gif" alt="TinyDrive Demo" width="600"/>
</p>

# TinyDrive

**TinyDrive** is a Vision-Language Model (VLM) specifically designed for real-time Visual Question Answering (VQA) in self-driving cars.  
Built around the **T5-tiny** language model and a **custom multiscale vision encoder**, TinyDrive is lightweight enough to run at ~17 FPS on the Yahboom **Rosmaster X3 self-driving car**, powered by a **Jetson NANO 4G** board.  

The first generation of TinyDrive includes two versions:
- **TinyDrive-v11**: T5-tiny + custom vision encoder  
- **TinyDrive-v12**: T5-mini + custom vision encoder  

In this repository, we provide:
- The **Rosmaster dataset**, a custom dataset curated in our laboratory for VQA  
- A **PyTorch implementation** of TinyDrive  
- Experiments conducted on both the **Rosmaster dataset** and the **DriveLM-nuScenes dataset**  

Please refer to each directory for detailed instructions on dataset preparation and code execution.  

---

## Project Progress

### Rosmaster Experiments
- [x] Token generation  
- [x] Vision encoder  
- [x] Prioritized buffer dataset  
- [x] Training  
- [x] Testing  

### DriveLM-nuScenes Experiments
- [ ] Token generation  
- [ ] Vision encoder  
- [ ] Prioritized buffer dataset  
- [ ] Training  
- [ ] Testing  

### Rosmaster Dataset
- [ ] Object images  
- [ ] No-object images  
- [ ] Pool of QAs  
- [ ] Generating QAs  
- [ ] Splitting QAs  

---

## Paper

📄 **TinyDrive: Multiscale Visual Question Answering with Selective Token Routing for Autonomous Driving**  
[arXiv:2505.15564](https://arxiv.org/abs/2505.15564)

### Citation
```bibtex
@article{hassani2025tinydrive,
  title={TinyDrive: Multiscale Visual Question Answering with Selective Token Routing for Autonomous Driving},
  author={Hassani, Hossein and Nikan, Soodeh and Shami, Abdallah},
  journal={arXiv preprint arXiv:2505.15564},
  year={2025}
}
