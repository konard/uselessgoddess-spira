//! Fluid simulation parameters.
//!
//! These parameters control the behavior and appearance of the fluid simulation.
//! They can be modified at runtime through the Bevy resource system.

use bevy::prelude::*;

/// Parameters controlling the fluid simulation behavior.
///
/// These values are tuned for water-like behavior by default. Modifying them
/// allows simulating different fluid types (honey, oil, etc.).
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct FluidParams {
    /// Rest density of the fluid in kg/m³.
    /// Water is approximately 1000 kg/m³.
    pub rest_density: f32,

    /// Particle radius for rendering and collision.
    /// Typically h/4 where h is the smoothing radius.
    pub particle_radius: f32,

    /// Smoothing kernel radius (h).
    /// Larger values = smoother but more expensive simulation.
    /// Typical range: 0.1 - 0.5 for normalized coordinates.
    pub smoothing_radius: f32,

    /// Relaxation parameter for constraint solving.
    /// Values close to 0 make the fluid more compressible.
    /// Typical range: 0.01 - 0.1
    pub relaxation_epsilon: f32,

    /// XSPH viscosity coefficient.
    /// Higher values = more viscous fluid (honey-like).
    /// Typical range: 0.0 - 0.5 for water.
    pub viscosity: f32,

    /// Vorticity confinement coefficient.
    /// Restores rotational motion lost due to numerical damping.
    /// Typical range: 0.0 - 0.1
    pub vorticity_epsilon: f32,

    /// Surface tension coefficient (artificial).
    /// Creates cohesion between particles at the surface.
    /// Typical range: 0.0 - 0.1
    pub surface_tension: f32,

    /// Gravity acceleration vector.
    pub gravity: Vec3,

    /// Number of constraint solver iterations per substep.
    /// More iterations = more accurate but slower.
    /// Typical range: 2 - 6
    pub solver_iterations: u32,

    /// Number of substeps per frame.
    /// More substeps = more stable at high speeds.
    /// Typical range: 1 - 4
    pub substeps: u32,

    /// Maximum number of particles.
    /// This determines the buffer sizes on the GPU.
    pub max_particles: u32,

    /// Size of the spatial hash grid (cells per axis).
    /// Should cover the simulation domain efficiently.
    pub grid_size: UVec3,

    /// Simulation domain bounds (min corner).
    pub bounds_min: Vec3,

    /// Simulation domain bounds (max corner).
    pub bounds_max: Vec3,

    /// Timestep for simulation (will be multiplied by substeps).
    /// Use None to use frame delta time.
    pub fixed_timestep: Option<f32>,

    /// Coefficient of restitution for boundary collisions.
    /// 0.0 = fully inelastic, 1.0 = fully elastic.
    pub boundary_restitution: f32,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            rest_density: 1000.0,
            particle_radius: 0.02,
            smoothing_radius: 0.08,
            relaxation_epsilon: 0.01,
            viscosity: 0.01,
            vorticity_epsilon: 0.001,
            surface_tension: 0.0001,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver_iterations: 4,
            substeps: 2,
            max_particles: 65536,
            grid_size: UVec3::new(64, 64, 64),
            bounds_min: Vec3::new(-2.0, -0.5, -2.0),
            bounds_max: Vec3::new(2.0, 3.0, 2.0),
            fixed_timestep: None,
            boundary_restitution: 0.3,
        }
    }
}

impl FluidParams {
    /// Creates parameters optimized for real-time water simulation.
    pub fn water() -> Self {
        Self::default()
    }

    /// Creates parameters for honey-like viscous fluid.
    pub fn honey() -> Self {
        Self {
            viscosity: 0.3,
            vorticity_epsilon: 0.0,
            ..Self::default()
        }
    }

    /// Creates parameters for fast, splashy water (games).
    pub fn splashy() -> Self {
        Self {
            viscosity: 0.001,
            vorticity_epsilon: 0.01,
            surface_tension: 0.0005,
            solver_iterations: 3,
            substeps: 3,
            ..Self::default()
        }
    }

    /// Calculates the cell size based on smoothing radius.
    /// Cell size should be >= smoothing radius for correct neighbor search.
    pub fn cell_size(&self) -> f32 {
        self.smoothing_radius
    }

    /// Calculates particle mass based on rest density and particle spacing.
    pub fn particle_mass(&self) -> f32 {
        let volume = (2.0 * self.particle_radius).powi(3);
        self.rest_density * volume
    }
}

/// GPU-compatible uniform buffer for shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FluidParamsUniform {
    // Simulation parameters
    pub rest_density: f32,
    pub smoothing_radius: f32,
    pub smoothing_radius_sq: f32,
    pub relaxation_epsilon: f32,

    pub viscosity: f32,
    pub vorticity_epsilon: f32,
    pub surface_tension: f32,
    pub particle_mass: f32,

    pub gravity: [f32; 3],
    pub dt: f32,

    // Grid parameters
    pub grid_size: [u32; 3],
    pub num_particles: u32,

    pub bounds_min: [f32; 3],
    pub cell_size: f32,

    pub bounds_max: [f32; 3],
    pub boundary_restitution: f32,

    // Kernel coefficients (precomputed)
    pub poly6_coeff: f32,
    pub spiky_grad_coeff: f32,
    pub viscosity_laplacian_coeff: f32,
    pub _padding: f32,
}

impl FluidParamsUniform {
    pub fn from_params(params: &FluidParams, dt: f32, num_particles: u32) -> Self {
        let h = params.smoothing_radius;
        let h2 = h * h;
        let h3 = h2 * h;
        let h6 = h3 * h3;
        let h9 = h6 * h3;

        // Precompute kernel coefficients
        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h9);
        let spiky_grad_coeff = -45.0 / (std::f32::consts::PI * h6);
        let viscosity_laplacian_coeff = 45.0 / (std::f32::consts::PI * h6);

        Self {
            rest_density: params.rest_density,
            smoothing_radius: h,
            smoothing_radius_sq: h2,
            relaxation_epsilon: params.relaxation_epsilon,

            viscosity: params.viscosity,
            vorticity_epsilon: params.vorticity_epsilon,
            surface_tension: params.surface_tension,
            particle_mass: params.particle_mass(),

            gravity: params.gravity.to_array(),
            dt,

            grid_size: params.grid_size.to_array(),
            num_particles,

            bounds_min: params.bounds_min.to_array(),
            cell_size: params.cell_size(),

            bounds_max: params.bounds_max.to_array(),
            boundary_restitution: params.boundary_restitution,

            poly6_coeff,
            spiky_grad_coeff,
            viscosity_laplacian_coeff,
            _padding: 0.0,
        }
    }
}
