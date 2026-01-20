//! Fluid simulation parameters.
//!
//! This module defines all configurable parameters for the PBF simulation,
//! following the notation from the original PBF paper (Macklin & Müller, 2013).

use bevy::prelude::*;

/// Main configuration resource for the fluid simulation.
///
/// These parameters control the physical behavior and numerical accuracy
/// of the Position Based Fluids simulation.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct FluidParams {
    /// Rest density of the fluid in kg/m³.
    /// Water is approximately 1000 kg/m³.
    pub rest_density: f32,

    /// Smoothing radius (kernel support radius) in meters.
    /// This determines the range of particle interactions.
    /// Typically 4x the particle radius for good results.
    pub smoothing_radius: f32,

    /// Particle radius in meters.
    /// Visual size of particles and collision radius.
    pub particle_radius: f32,

    /// Particle mass in kg.
    /// Derived from rest_density and particle_radius if not set.
    pub particle_mass: f32,

    /// XSPH viscosity coefficient (0.0 - 1.0).
    /// Higher values create more viscous fluids.
    pub viscosity: f32,

    /// Vorticity confinement epsilon.
    /// Adds rotational details to prevent energy loss.
    pub vorticity_epsilon: f32,

    /// Surface tension coefficient.
    /// Creates cohesion between particles near the surface.
    pub surface_tension: f32,

    /// Relaxation parameter for constraint solving (typically 1.0).
    /// Lower values provide more stability but slower convergence.
    pub relaxation: f32,

    /// Number of solver iterations per substep.
    /// More iterations = better incompressibility but slower.
    pub solver_iterations: u32,

    /// Number of substeps per frame.
    /// More substeps = more accurate but slower.
    pub substeps: u32,

    /// Gravity acceleration vector in m/s².
    pub gravity: Vec3,

    /// Epsilon for constraint regularization to prevent division by zero.
    pub constraint_epsilon: f32,

    /// Kernel correction term (epsilon in lambda calculation).
    /// Prevents clustering in low-density areas.
    pub kernel_epsilon: f32,

    /// Maximum number of particles.
    /// Used for GPU buffer allocation.
    pub max_particles: u32,

    /// Enable adaptive time stepping based on CFL condition.
    pub adaptive_timestep: bool,

    /// Maximum velocity for CFL condition.
    pub max_velocity: f32,
}

impl Default for FluidParams {
    fn default() -> Self {
        let particle_radius: f32 = 0.05; // 5cm particles
        let smoothing_radius = particle_radius * 4.0;

        // Calculate mass from rest density and particle spacing
        let particle_volume = (4.0 / 3.0) * std::f32::consts::PI * particle_radius.powi(3);
        let particle_mass = 1000.0 * particle_volume; // water density

        Self {
            rest_density: 1000.0, // water
            smoothing_radius,
            particle_radius,
            particle_mass,
            viscosity: 0.01,
            vorticity_epsilon: 0.0001,
            surface_tension: 0.0,
            relaxation: 1.0,
            solver_iterations: 4,
            substeps: 2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            constraint_epsilon: 1e-6,
            kernel_epsilon: 100.0,
            max_particles: 100_000,
            adaptive_timestep: false,
            max_velocity: 10.0,
        }
    }
}

impl FluidParams {
    /// Create parameters for water-like fluid.
    pub fn water() -> Self {
        Self::default()
    }

    /// Create parameters for a more viscous fluid (like honey).
    pub fn viscous() -> Self {
        Self {
            viscosity: 0.1,
            surface_tension: 0.02,
            ..Self::default()
        }
    }

    /// Create parameters for a gas-like fluid.
    pub fn gas() -> Self {
        Self {
            rest_density: 1.2, // air density
            viscosity: 0.001,
            gravity: Vec3::ZERO,
            ..Self::default()
        }
    }

    /// Calculate the cubic smoothing kernel coefficient.
    /// Used in the Poly6 kernel: W(r, h) = coefficient * (h² - r²)³
    pub fn poly6_coefficient(&self) -> f32 {
        let h = self.smoothing_radius;
        315.0 / (64.0 * std::f32::consts::PI * h.powi(9))
    }

    /// Calculate the spiky kernel gradient coefficient.
    /// Used for pressure forces: ∇W(r, h) = coefficient * (h - r)² * (r/|r|)
    pub fn spiky_gradient_coefficient(&self) -> f32 {
        let h = self.smoothing_radius;
        -45.0 / (std::f32::consts::PI * h.powi(6))
    }

    /// Calculate the viscosity kernel Laplacian coefficient.
    pub fn viscosity_laplacian_coefficient(&self) -> f32 {
        let h = self.smoothing_radius;
        45.0 / (std::f32::consts::PI * h.powi(6))
    }

    /// Calculate the effective timestep for simulation.
    pub fn effective_timestep(&self, frame_dt: f32) -> f32 {
        frame_dt / self.substeps as f32
    }

    /// Set particle spacing and derive related parameters.
    pub fn with_particle_spacing(mut self, spacing: f32) -> Self {
        self.particle_radius = spacing / 2.0;
        self.smoothing_radius = spacing * 2.0; // Common choice: h = 2 * spacing

        // Recalculate mass
        let particle_volume =
            (4.0 / 3.0) * std::f32::consts::PI * self.particle_radius.powi(3);
        self.particle_mass = self.rest_density * particle_volume;

        self
    }

    /// Set the maximum particle count.
    pub fn with_max_particles(mut self, max: u32) -> Self {
        self.max_particles = max;
        self
    }

    /// Set solver iterations.
    pub fn with_solver_iterations(mut self, iterations: u32) -> Self {
        self.solver_iterations = iterations;
        self
    }

    /// Set substeps per frame.
    pub fn with_substeps(mut self, substeps: u32) -> Self {
        self.substeps = substeps;
        self
    }

    /// Set gravity.
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }
}

/// Runtime state of the fluid simulation.
#[derive(Resource, Default, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct FluidState {
    /// Current number of active particles.
    pub particle_count: u32,

    /// Current simulation time.
    pub time: f32,

    /// Current frame number.
    pub frame: u64,

    /// Average density error (for debugging).
    pub avg_density_error: f32,

    /// Maximum velocity in the simulation.
    pub max_velocity: f32,

    /// Whether the simulation is paused.
    pub paused: bool,

    /// Whether to run a single step when paused.
    pub step_once: bool,
}

impl FluidState {
    /// Check if the simulation should run this frame.
    pub fn should_simulate(&mut self) -> bool {
        if self.paused {
            if self.step_once {
                self.step_once = false;
                true
            } else {
                false
            }
        } else {
            true
        }
    }

    /// Toggle pause state.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Request a single simulation step.
    pub fn request_step(&mut self) {
        self.step_once = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = FluidParams::default();
        assert!(params.rest_density > 0.0);
        assert!(params.smoothing_radius > params.particle_radius);
        assert!(params.solver_iterations > 0);
    }

    #[test]
    fn test_kernel_coefficients() {
        let params = FluidParams::default();
        assert!(params.poly6_coefficient() > 0.0);
        assert!(params.spiky_gradient_coefficient() < 0.0);
        assert!(params.viscosity_laplacian_coefficient() > 0.0);
    }
}
