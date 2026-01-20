//! Spira - GPU-accelerated fluid simulation for Bevy
//!
//! This library provides Position Based Fluids (PBF) simulation for the Bevy game engine,
//! optimized for real-time applications.
//!
//! # Features
//!
//! - **PBF Simulation**: Stable, unconditionally convergent fluid simulation
//! - **GPU Acceleration**: Compute shader-based simulation (planned)
//! - **CPU Fallback**: CPU-based simulation for debugging and compatibility
//! - **Flexible Rendering**: Multiple rendering methods (sprites, meshes, screen-space)
//! - **Boundary Handling**: Box, plane, and sphere boundaries
//! - **Easy Integration**: Simple Bevy plugin interface
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use spira::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(FluidPlugin::default().cpu_mode())
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands) {
//!     // Add a camera
//!     commands.spawn((
//!         Camera3d::default(),
//!         Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
//!     ));
//!
//!     // Add a light
//!     commands.spawn((
//!         PointLight {
//!             shadows_enabled: true,
//!             ..default()
//!         },
//!         Transform::from_xyz(4.0, 8.0, 4.0),
//!     ));
//!
//!     // Add a fluid emitter
//!     commands.spawn((
//!         FluidEmitter::box_emitter(Vec3::new(1.0, 2.0, 1.0), 0.1),
//!         Transform::from_xyz(0.0, 3.0, 0.0),
//!     ));
//!
//!     // Add boundary container
//!     commands.spawn(BoxBoundaryBundle {
//!         boundary: BoxBoundary::new(
//!             Vec3::new(-3.0, 0.0, -3.0),
//!             Vec3::new(3.0, 10.0, 3.0),
//!         ),
//!         ..default()
//!     });
//! }
//! ```
//!
//! # Architecture
//!
//! The library is organized into the following modules:
//!
//! - [`fluid`]: Core fluid simulation module
//!   - [`fluid::params`]: Simulation parameters
//!   - [`fluid::particle`]: Particle data structures
//!   - [`fluid::spatial`]: Spatial hashing for neighbor search
//!   - [`fluid::solver`]: PBF constraint solver
//!   - [`fluid::boundary`]: Boundary handling
//!   - [`fluid::render`]: Rendering systems
//!   - [`fluid::plugin`]: Bevy plugin

pub mod fluid;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::fluid::prelude::*;
}
