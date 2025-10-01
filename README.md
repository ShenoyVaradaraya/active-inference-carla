# Active Inference in CARLA Simulator

An implementation of Active Inference for autonomous driving in the CARLA simulator. This project demonstrates how Active Inference principles can be applied to vehicle control and navigation in a simulated urban environment.

## Overview

This repository implements an Active Inference agent that controls a vehicle in CARLA 0.9.14. Active Inference is a theoretical framework from neuroscience that explains how agents minimize prediction errors while navigating their environment. The implementation includes sensory processing, belief updating, and action selection based on the Free Energy Principle.

## Requirements

### CARLA Simulator
- **CARLA 0.9.14** (this specific version is required)
- Follow installation instructions from :[https://learnopencv.com/ros2-and-carla-setup-guide/]

### Python Dependencies
- Python 3.7+
- numpy
- opencv-python
- carla (PythonAPI from CARLA 0.9.14)
- matplotlib
- pygame


### 3. Install OpenCV (Important!)

**Note:** If you encounter OpenCV errors, install version 1.23:

```bash
# Install specific numpy version first (required for OpenCV compatibility)
pip install numpy==1.23.0

# Then install OpenCV
pip install opencv-python
```

## Usage

### 1. Start CARLA Server

First, launch the CARLA simulator server:

```bash
# Navigate to CARLA directory
cd /path/to/CARLA_0.9.14

# Linux
./CarlaUE4.sh

# Windows
CarlaUE4.exe

# For headless mode (no rendering):
./CarlaUE4.sh -RenderOffScreen

# To use GPU
./CarlaUE4.sh -prefernvidia
```

### 2. Run the Active Inference Agent

```bash
# Clone this repository
git clone https://github.com/ShenoyVaradaraya/active-inference-carla.git
cd active-inference-carla

# Run the main script
python main.py --scenario town03 
```
If you are choosing town03, ```spawn_point.location + carla.Location(x=-TARGET_HEADWAY)``` but if you are choosing town05, ```spawn_point.location + carla.Location(y=-TARGET_HEADWAY)```. 

## Project Structure

```
active-inference-carla/
├── main.py                 # Main entry point
├── adaptive_aif.py     # Active Inference agent implementation for longitudinal 
└── adaptive_aif_with_steering.py # Active Inference agent implementation for longitudinal and lateral control
└── lead_vehicle_controller.py # Controller for lead vehicle with different behaviour
└── visualization.py # Script for HUD display in carla environment
```

## Features

- **Active Inference Framework**: Implements belief updating and action selection based on Free Energy minimization
- **Autonomous Navigation**: Controls vehicle steering, throttle, and braking
- **Real-time Visualization**: Displays agent beliefs and predictions


## Troubleshooting

### OpenCV Import Error
If you encounter errors like `ImportError: numpy.core.multiarray failed to import`:
```bash
pip uninstall numpy opencv-python
pip install numpy==1.23.0
pip install opencv-python
```

### CARLA Connection Error
- Ensure CARLA server is running before starting the agent
- Check that you're using CARLA 0.9.14 (other versions may have API incompatibilities)
- Verify the host and port settings match your CARLA server configuration

### Performance Issues
- Run CARLA in low quality mode: `./CarlaUE4.sh -quality-level=Low`
- Use synchronous mode for better stability

## References

- [CARLA Simulator Documentation](https://carla.readthedocs.io/en/0.9.14/)
- [Active Inference: A Process Theory](https://www.mitpressjournals.org/doi/full/10.1162/NECO_a_00912)
- [Free Energy Principle](https://www.nature.com/articles/nrn2787)

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Varadaraya Shenoy

## Acknowledgments

- CARLA Simulator team
- Active Inference research community

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{active_inference_carla,
  author = {Shenoy, Varadaraya},
  title = {Active Inference in CARLA Simulator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ShenoyVaradaraya/active-inference-carla}
}
```
