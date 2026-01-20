//! Position Based Fluids (PBF) simulation module for Bevy.
//!
//! This module provides a GPU-accelerated fluid simulation using the Position Based Fluids method,
//! which offers excellent stability and performance for real-time applications.
//!
//! # Architecture
//!
//! The simulation is structured in the following components:
//!
//! - [`params`]: Simulation parameters (density, viscosity, etc.)
//! - [`particle`]: Particle data structures and spawning
//! - [`spatial`]: Spatial hashing for efficient neighbor search
//! - [`solver`]: PBF constraint solver (density constraints)
//! - [`boundary`]: Boundary handling and collision detection
//! - [`render`]: Particle rendering (sprites and screen-space)
//! - [`plugin`]: Bevy plugin for easy integration
//!
//! # Example
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use spira::fluid::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(FluidPlugin::default())
//!         .add_systems(Startup, spawn_fluid)
//!         .run();
//! }
//!
//! fn spawn_fluid(mut commands: Commands) {
//!     // Spawn a block of fluid particles
//!     commands.spawn(FluidEmitter {
//!         shape: EmitterShape::Box {
//!             half_extents: Vec3::new(1.0, 2.0, 1.0),
//!         },
//!         particle_spacing: 0.1,
//!         initial_velocity: Vec3::ZERO,
//!         ..default()
//!     });
//! }
//! ```

pub mod params;
pub mod particle;
pub mod spatial;
pub mod solver;
pub mod boundary;
pub mod render;
pub mod plugin;

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::params::*;
    pub use super::particle::*;
    pub use super::spatial::*;
    pub use super::solver::*;
    pub use super::boundary::*;
    pub use super::render::*;
    pub use super::plugin::*;
}
