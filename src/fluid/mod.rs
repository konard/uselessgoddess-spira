//! GPU-accelerated Position Based Fluids (PBF) simulation for Bevy.
//!
//! This module implements a real-time fluid simulation using Position Based Fluids,
//! a method that formulates fluid simulation as a constraint satisfaction problem.
//!
//! # Architecture
//!
//! The simulation follows this pipeline each frame:
//! 1. **Apply forces** - External forces (gravity) are applied to particles
//! 2. **Predict positions** - New positions are predicted based on velocities
//! 3. **Build spatial hash** - Particles are binned into a spatial hash grid
//! 4. **Solve constraints** - Density constraints are iteratively solved (PBF core)
//! 5. **Update velocities** - Velocities are computed from position changes
//! 6. **Apply viscosity** - XSPH viscosity is applied for smoother flow
//!
//! # Usage
//!
//! ```rust,ignore
//! use bevy::prelude::*;
//! use spira::fluid::FluidPlugin;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(FluidPlugin)
//!         .run();
//! }
//! ```

mod params;
mod plugin;
mod simulation;

pub use params::FluidParams;
pub use plugin::FluidPlugin;
pub use simulation::{spawn_particle_block, FluidSimulation, ParticleStagingBuffer};
