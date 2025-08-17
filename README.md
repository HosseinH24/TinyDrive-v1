# TinyDrive
TinyDrive is a Vision-Language Model (VLM), specifically designed to be deployed for real-time Visual Question-Answering (VQA) in self-driving cars. Built around the T5-tiny and a custom vision encoder, TinyDrive is lightweight enough to run with an FPS of ~17 on the Yahboom Rosmaster X3 self-driving car which is running on a Jetson NANO 4G board. The first generation of TinyDrive, i.e., TinyDrive-v1, has two versions:
- TinyDrive-v11: T5-tiny + custom vision encoder
- TinyDrive-v12: T5-mini + custom vision encoder

![TinyDrive Demo](Media/TinyDrive_v11.gif)


Here’s the current status of the work:

- [x] Vision Encoder Implementation
- [ ] Ablation Studies  
