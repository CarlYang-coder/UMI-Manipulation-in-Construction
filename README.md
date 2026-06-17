# UMI-Manipulation-in-Construction

Building upon **UMI-FT** (https://github.com/real-stanford/UMI-FT) and its iPhone-based data collection framework, this project establishes an end-to-end workflow for **construction robotics**, spanning **demonstration collection**, **policy training**, and **real-time inference**.

Looking ahead, we aim to extend the system to **mobile robotic platforms**, enabling more complex construction manipulation tasks in **dynamic** and **large-scale environments**.

## Hardware Design

The hardware system is designed to support construction-oriented robotic manipulation tasks in real-world environments. It provides an integrated platform for demonstration collection, policy deployment, and real-time inference, while also offering a foundation for future extensions to mobile robotic systems operating in dynamic and large-scale construction settings.

## Data Collection

The data collection pipeline is built upon the UMI-FT framework and its iPhone-based sensing system, enabling efficient capture of human demonstrations for construction manipulation tasks. It supports the acquisition of multimodal data that can be used for downstream policy learning and system evaluation.

## Calibration on xArm

The trajectory replay module allows recorded demonstrations to be executed and analyzed within the robotic system, providing a practical mechanism for validation, debugging, and performance assessment. This capability helps ensure consistency between collected demonstrations and deployed robot behaviors.

## Multi-Model

The multi-model framework is designed to support different policy representations and learning strategies for construction robotics tasks. By enabling flexible integration of multiple models, the system can better adapt to diverse manipulation scenarios and serve as a foundation for future extensions in complex environments.
