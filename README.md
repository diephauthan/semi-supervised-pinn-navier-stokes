# Semi-Supervised PINN for Low Reynolds Number Flow Around Elliptic Cylinders

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Semi-Supervised](https://img.shields.io/badge/Learning-Semi--Supervised-green.svg)](https://github.com/diephauthan/pinn-navier-stokes)
[![ASME FEDSM 2026](https://img.shields.io/badge/Conference-ASME%20FEDSM%202026-blue.svg)](https://event.asme.org/FEDSM)

A PyTorch implementation of **Semi-Supervised Physics-Informed Neural Networks (PINNs)** for simulating incompressible, unsteady flow over elliptic cylinders at low Reynolds numbers. This approach combines limited labeled data (initial and boundary conditions only) with physics-guided learning through Navier-Stokes residuals.

**Key Innovation**: Eliminates the need for interior velocity/pressure field data during training - achieving 97.5% reduction in labeled data requirement while maintaining high accuracy!

**Research Context**: This work was presented at ASME FEDSM 2026 (Paper ID: FEDSM2026-184556)

## 📋 Table of Contents

- [Overview](#overview)
- [Semi-Supervised Learning Approach](#semi-supervised-learning-approach)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Method](#method)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

This project implements a **Semi-Supervised Learning** approach using Physics-Informed Neural Networks (PINNs) to solve the 2D incompressible Navier-Stokes equations:

```
∂u/∂t + λ₁(u·∇u) + ∇p - λ₂∇²u = 0
∂v/∂t + λ₁(v·∇v) + ∇p - λ₂∇²v = 0
```

where:
- `u, v` are velocity components
- `p` is pressure
- `λ₁, λ₂` are learnable parameters related to Reynolds number

### Semi-Supervised Learning Strategy

The network combines:
- **Labeled data** (supervised):
  - **Initial conditions** (IC): ~5,000 points at t=0
  - **Boundary conditions** (BC): ~15,000 points across all time steps
- **Unlabeled data** (physics-guided):
  - **Physics residuals**: ~10,000 collocation points enforcing PDE constraints

This semi-supervised approach reduces the need for expensive labeled data while maintaining accuracy through physics constraints.

## ✨ Features

- **Semi-supervised learning**: Combines limited labeled data with physics constraints
- **Efficient data usage**: Learns from sparse measurements and physics laws
- **Physics-informed loss**: Integrates PDE residuals directly into training
- **Automatic differentiation**: Uses PyTorch's autograd for computing derivatives
- **Multi-loss training**: Balanced loss components for IC, BC, and physics
- **Two-stage optimization**: Adam optimizer followed by L-BFGS for fine-tuning
- **Parameter discovery**: Learns unknown PDE parameters (λ₁, λ₂) from data
- **Comprehensive logging**: Saves training history, predictions, and visualizations
- **GPU acceleration**: Supports CUDA for faster training

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- Matplotlib
- Pandas

### Setup

1. Clone the repository:
```bash
git clone https://github.com/diephauthan/semi-supervised-pinn-navier-stokes.git
cd semi-supervised-pinn-navier-stokes

```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the data file (see [Data](#data) section)

## 📊 Usage

### Basic Usage

```python
from pinn_navier_stokes import main

# Run training with default parameters
main(
    data_path='ellip_cylinder_wake.mat',
    output_dir='results',
    n_unlabeled=10000,
    nIter_adam=100000,
    use_lbfgs=True
)
```

### Custom Training

```python
from pinn_navier_stokes import PINN_ICBC, prepare_data_icbc_only
import scipy.io

# Load data
data = scipy.io.loadmat('ellip_cylinder_wake.mat')
X_star = data['X_star']
U_star = data['U_star']
t_star = data['t']

# Prepare data
data_tuple = prepare_data_icbc_only(X_star, U_star, t_star, n_unlabeled=10000)

# Define network architecture
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Initialize model
model = PINN_ICBC(
    *data_tuple, 
    layers,
    alpha_ic=1.0,      # IC loss weight
    alpha_bc=1.0,      # BC loss weight
    alpha_physics=1.0  # Physics loss weight
)

# Train
model.train(nIter_adam=100000, use_lbfgs=True)

# Make predictions
u_pred, v_pred, p_pred = model.predict(x_test, y_test, t_test)
```

### Command Line

```bash
python pinn_navier_stokes.py
```

Edit the configuration at the bottom of `pinn_navier_stokes.py`:

```python
DATA_PATH = 'ellip_cylinder_wake.mat'
OUTPUT_DIR = 'results'
N_UNLABELED = 10000
N_ITER_ADAM = 100000
USE_LBFGS = True
```

## 🧪 Method

### Network Architecture

The PINN consists of:
- **Input layer**: (x, y, t) coordinates
- **Hidden layers**: 8 layers with 20 neurons each, tanh activation
- **Output layer**: (ψ, p) - stream function and pressure

Velocities are computed from the stream function:
```
u = ∂ψ/∂y
v = -∂ψ/∂x
```

### Loss Function

The total loss is a weighted sum of three components:

```
L_total = α_IC · L_IC + α_BC · L_BC + α_physics · L_physics
```

where:
- `L_IC`: Mean squared error on initial conditions
- `L_BC`: Mean squared error on boundary conditions
- `L_physics`: Mean squared error of PDE residuals

### Training Strategy

1. **Stage 1 - Adam optimizer** (100,000 iterations):
   - Learning rate: 0.001
   - Faster convergence, exploration of parameter space

2. **Stage 2 - L-BFGS optimizer**:
   - Fine-tuning for better accuracy
   - Uses strong Wolfe line search

## 📈 Results

After training, the model generates:

### Output Files

```
results/
├── model_checkpoint_YYYYMMDD_HHMMSS.pth      # Trained model
├── 3d_contour_YYYYMMDD_HHMMSS.svg            # 3D visualization
├── pressure_comparison_YYYYMMDD_HHMMSS.svg   # Pressure fields
└── convergence_history_YYYYMMDD_HHMMSS.svg   # Training curves
```

## 📁 Project Structure

```
pinn-navier-stokes/
│
├── pinn_navier_stokes.py    # Main implementation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT license
│
├── data/                     # Data directory
    ├── ellip_cylinder_wake.mat
    ├── ellip_cylinder_wake_60deg.mat
    └── ellip_cylinder_wake_90deg.mat

```

## 💾 Data

The code works with flow simulation data for elliptic cylinders. The data file should be in `.mat` format with the following structure:

```matlab
X_star  : [N x 2]     % Spatial coordinates (x, y)
U_star  : [N x 2 x T] % Velocity field (u, v) over time
t       : [T x 1]     % Time steps
p_star  : [N x T]     % Pressure field over time
```

### Dataset Information

The reference dataset used in our paper (`ellip_cylinder_wake.mat`) contains:
- **Flow configuration**: 2D incompressible flow around an elliptic cylinder
- **Reynolds number**: Re = 1000 (based on cylinder minor axis)
- **Cylinder geometry**: Major axis = 2 units, Minor axis = 1 unit
- **Incident angles**: β = 0°, 60°, 90°
- **Domain**: [-15, 25] × [-8, 8]
- **Time interval**: 0 to 120 seconds (training uses t = 100-120s)

### Generating Your Own Data

Reference CFD simulations were generated using OpenFOAM with:
- Solver: `pimpleFoam` (transient incompressible)
- Mesh: Structured near-wall, refined wake region
- Boundary conditions: Uniform inlet (1 m/s), zero pressure outlet, no-slip walls

See the paper for detailed OpenFOAM setup and mesh generation procedures.

### Data Sources

- **Original PINN cylinder data**: [Maziar Raissi's PINNs Repository](https://github.com/maziarraissi/PINNs)
- **This work**: Elliptic cylinder simulations (contact authors for data requests)

## 🎓 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{diep2026semisupervised,
  title={Semi-supervised Physics-Informed Neural Network Simulation of Low Reynolds Number Flow Around an Elliptic Cylinder},
  author={Diep, Than and Badri, Mehdi and Doan, Minh},
  booktitle={Proceedings of the ASME 2026 Fluids Engineering Division Summer Meeting},
  year={2026},
  month={July},
  location={Bellevue, Washington},
  organization={ASME},
  note={Paper ID: FEDSM2026-184556}
}
```

### Foundational Work

This implementation builds upon the foundational PINN framework:

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original PINNs paper by [Raissi et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- PyTorch team for the excellent deep learning framework
- Scientific Python community

## 📧 Contact

- **Author**: Than Diep
- **Affiliation**: 
  - Faculty of Mechanical Engineering, Ho Chi Minh City University of Technology (HCMUT)
  - Vietnam National University - Ho Chi Minh City (VNU-HCM)
- **Email**: diephauthan@gmail.com
- **GitHub**: [@diephauthan](https://github.com/diephauthan)
- **Issues**: [GitHub Issues](https://github.com/diephauthan/pinn-navier-stokes/issues)

For questions or feedback, please open an issue on GitHub.

---

## 📄 Publication

This work is based on research presented at:

**FEDSM2026 - ASME Fluids Engineering Division Summer Meeting**
- **Paper**: "Semi-supervised Physics-Informed Neural Network Simulation of Low Reynolds Number Flow Around an Elliptic Cylinder"
- **Authors**: Than Diep, Mehdi Badri, and Minh Doan
- **Conference**: ASME 2026 FEDSM, July 26-29, 2026, Bellevue, Washington
- **Paper ID**: FEDSM2026-184556

### Acknowledgments

This research is funded by Vietnam National University Ho Chi Minh City (VNU-HCM) under grant number **C2025-20-34**. We acknowledge Ho Chi Minh City University of Technology (HCMUT), VNU-HCM for supporting this study.

---

**Note**: This implementation demonstrates semi-supervised learning for PDEs, combining limited labeled data with physics constraints. For research and educational purposes.
