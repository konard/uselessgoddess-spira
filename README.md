# Spira

High-performance hydrodynamic simulation of nasal sinus airflow using the Lattice Boltzmann Method (LBM) in Rust.

## Overview

Spira simulates fluid dynamics using the D3Q19 Lattice Boltzmann Method with GPU compute shaders. The system:

- Ingests 3D voxel data (currently using a procedural phantom sinus structure)
- Computes fluid dynamics on the GPU using WGSL compute shaders
- Visualizes the airflow velocity field in real-time

## Technology Stack

- **Language**: Rust (2024 edition)
- **Engine**: Bevy 0.17
- **Graphics API**: wgpu (via Bevy) using WGSL for shaders
- **Simulation Method**: LBM D3Q19 with BGK collision operator

## Features

### Data Structures
- `Texture3D` (Storage Texture) for voxel grid geometry (Wall vs Air)
- Ping-pong `Texture3D` buffers for distribution functions (f_in / f_out)
- Velocity field texture for visualization

### Compute Pipeline (Physics)
- LBM D3Q19 Compute Shader in WGSL
- **Streaming Step**: Propagates distribution functions to neighbors
- **Collision Step**: Relaxes towards equilibrium using BGK operator
- **Boundary Conditions**:
  - Bounce-back for voxel walls (bone)
  - Pressure differential inlet/outlet on Z-axis edges

### Visualization
- 3D volume rendering with orbital camera
- Interactive camera controls

### Phantom Sinus Generator
- Procedural noise-based hollow ellipsoid structure
- Simulates maxillary sinus geometry with inlet/outlet channels

## Building

```bash
# Install dependencies (Linux)
sudo apt-get install libwayland-dev libxkbcommon-dev libudev-dev libasound2-dev pkg-config

# Build
cargo build --release

# Run
cargo run --release
```

## Controls

- **Arrow Keys**: Rotate camera
- **+/-**: Zoom in/out

## Architecture

```
src/
├── main.rs          # Main application, Bevy setup, LBM plugin
assets/
└── shaders/
    └── lbm.wgsl     # LBM compute shader (D3Q19)
```

## License

MIT
