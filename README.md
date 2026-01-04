# Spira - Dental Fluid Simulation

High-performance dental fluid simulation for maxillary sinus rinsing using the Lattice Boltzmann Method (LBM) D3Q19.

## Overview

This application simulates saline solution flow through a patient's nasal sinus cavity, providing real-time visualization of the fluid dynamics during a sinus rinsing procedure.

## Features

- **LBM D3Q19 Physics**: Full 3D Lattice Boltzmann Method with 19 velocity directions
- **Real-time Simulation**: GPU-accelerated compute shaders for blazing fast performance
- **Volumetric Rendering**: Raymarched water visualization with Fresnel reflections and specular highlights
- **Procedural Sinus Cavity**: SDF-based hollow sphere with noise mimicking organic bone structure
- **Extensible Architecture**: VoxelDataProvider trait for future DICOM/CT scan integration

## Architecture

### Engine
- **Bevy 0.17**: Latest game engine with ECS architecture
- **wgpu**: Cross-platform GPU compute via WebGPU API

### Physics Kernel
- D3Q19 Lattice Boltzmann with BGK collision operator
- Guo forcing scheme for gravity
- Bounce-back boundary conditions (no-slip on bone voxels)
- Free surface approximation for liquid behavior

### Rendering
- Compute-based volumetric raymarching
- Water shading with Fresnel, fake refraction, and Blinn-Phong specular
- Gradient-based normal calculation from density field

## Project Structure

```
spira/
├── Cargo.toml              # Rust dependencies
├── src/
│   └── main.rs             # Application entry point and plugin
└── assets/
    └── shaders/
        ├── lbm.wgsl        # LBM compute shader (init, collide/stream)
        ├── inject.wgsl     # Fluid injection compute shader
        └── render.wgsl     # Volumetric raymarching shader
```

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

### Controls
- **W/S**: Move camera forward/backward
- **A/D**: Move camera left/right
- **Q/E**: Move camera up/down
- The camera also auto-rotates around the simulation volume

## Configuration

Key parameters in `src/main.rs`:

| Constant | Default | Description |
|----------|---------|-------------|
| `GRID_SIZE` | 128 | Simulation grid dimensions (128³ voxels) |
| `WORKGROUP_SIZE` | 8 | Compute shader workgroup size |
| `DISPLAY_FACTOR` | 4 | Window size multiplier |

Fluid parameters can be adjusted in `FluidParams::default()`:
- `viscosity`: Kinematic viscosity (affects relaxation time)
- `gravity`: Gravity vector (strong for liquid simulation)
- `injection_point`: Syringe tip position
- `injection_velocity`: Injection direction and speed
- `injection_rate`: Fluid density added per timestep

## Future Work

- **Sparse Block Grid**: Only simulate active blocks containing fluid
- **DICOM Integration**: Load real CT scan data via VoxelDataProvider trait
- **512³ Resolution**: Scale up with sparse optimization
- **Multi-fluid Simulation**: Saline + air interaction

## License

MIT OR Apache-2.0
