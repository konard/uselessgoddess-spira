//! PBF (Position Based Fluids) constraint solver.
//!
//! This module implements the core PBF algorithm as described in:
//! "Position Based Fluids" by Macklin & Müller (2013)
//!
//! The algorithm solves density constraints to achieve incompressible fluid behavior
//! using a position-based approach that is unconditionally stable.

use bevy::prelude::*;

use super::params::FluidParams;
use super::spatial::{CpuSpatialHash, SpatialHashConfig};

/// SPH kernel functions for PBF.
pub struct SphKernels;

impl SphKernels {
    /// Poly6 kernel for density estimation.
    /// W(r, h) = (315 / 64πh⁹) * (h² - r²)³ for r ≤ h
    #[inline]
    pub fn poly6(r_sq: f32, h: f32) -> f32 {
        if r_sq >= h * h {
            return 0.0;
        }
        let h_sq = h * h;
        let diff = h_sq - r_sq;
        let coefficient = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
        coefficient * diff * diff * diff
    }

    /// Poly6 kernel with precomputed coefficient.
    #[inline]
    pub fn poly6_with_coeff(r_sq: f32, h: f32, coeff: f32) -> f32 {
        if r_sq >= h * h {
            return 0.0;
        }
        let h_sq = h * h;
        let diff = h_sq - r_sq;
        coeff * diff * diff * diff
    }

    /// Gradient of Spiky kernel for pressure forces.
    /// ∇W(r, h) = -(45 / πh⁶) * (h - |r|)² * (r / |r|) for r ≤ h
    #[inline]
    pub fn spiky_gradient(r: Vec3, h: f32) -> Vec3 {
        let r_len = r.length();
        if r_len >= h || r_len < 1e-6 {
            return Vec3::ZERO;
        }
        let diff = h - r_len;
        let coefficient = -45.0 / (std::f32::consts::PI * h.powi(6));
        coefficient * diff * diff * (r / r_len)
    }

    /// Spiky gradient with precomputed coefficient.
    #[inline]
    pub fn spiky_gradient_with_coeff(r: Vec3, h: f32, coeff: f32) -> Vec3 {
        let r_len = r.length();
        if r_len >= h || r_len < 1e-6 {
            return Vec3::ZERO;
        }
        let diff = h - r_len;
        coeff * diff * diff * (r / r_len)
    }

    /// Laplacian of viscosity kernel.
    /// ∇²W(r, h) = (45 / πh⁶) * (h - |r|)
    #[inline]
    pub fn viscosity_laplacian(r_len: f32, h: f32) -> f32 {
        if r_len >= h {
            return 0.0;
        }
        let coefficient = 45.0 / (std::f32::consts::PI * h.powi(6));
        coefficient * (h - r_len)
    }
}

/// GPU-compatible solver parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSolverParams {
    // Fluid parameters (16 bytes)
    pub rest_density: f32,
    pub smoothing_radius: f32,
    pub particle_mass: f32,
    pub viscosity: f32,

    // Solver parameters (16 bytes)
    pub relaxation: f32,
    pub constraint_epsilon: f32,
    pub kernel_epsilon: f32,
    pub dt: f32,

    // Gravity and particle count (16 bytes)
    pub gravity: [f32; 3],
    pub particle_count: u32,

    // Kernel coefficients (16 bytes)
    pub poly6_coeff: f32,
    pub spiky_grad_coeff: f32,
    pub vorticity_epsilon: f32,
    pub surface_tension: f32,
}

impl GpuSolverParams {
    pub fn from_params(params: &FluidParams, particle_count: u32, dt: f32) -> Self {
        Self {
            rest_density: params.rest_density,
            smoothing_radius: params.smoothing_radius,
            particle_mass: params.particle_mass,
            viscosity: params.viscosity,
            relaxation: params.relaxation,
            constraint_epsilon: params.constraint_epsilon,
            kernel_epsilon: params.kernel_epsilon,
            dt,
            gravity: params.gravity.to_array(),
            particle_count,
            poly6_coeff: params.poly6_coefficient(),
            spiky_grad_coeff: params.spiky_gradient_coefficient(),
            vorticity_epsilon: params.vorticity_epsilon,
            surface_tension: params.surface_tension,
        }
    }
}

/// CPU-side PBF solver for testing and debugging.
pub struct CpuPbfSolver {
    /// Fluid parameters.
    params: FluidParams,
    /// Spatial hash for neighbor search.
    spatial_hash: CpuSpatialHash,

    // Particle data (SoA layout for CPU)
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    predicted_positions: Vec<Vec3>,
    densities: Vec<f32>,
    lambdas: Vec<f32>,
    delta_positions: Vec<Vec3>,
}

impl CpuPbfSolver {
    /// Create a new CPU solver.
    pub fn new(params: FluidParams, spatial_config: SpatialHashConfig) -> Self {
        Self {
            params,
            spatial_hash: CpuSpatialHash::new(spatial_config),
            positions: Vec::new(),
            velocities: Vec::new(),
            predicted_positions: Vec::new(),
            densities: Vec::new(),
            lambdas: Vec::new(),
            delta_positions: Vec::new(),
        }
    }

    /// Initialize particles.
    pub fn set_particles(&mut self, positions: Vec<Vec3>, velocities: Vec<Vec3>) {
        let n = positions.len();
        self.positions = positions;
        self.velocities = velocities;
        self.predicted_positions = vec![Vec3::ZERO; n];
        self.densities = vec![0.0; n];
        self.lambdas = vec![0.0; n];
        self.delta_positions = vec![Vec3::ZERO; n];
    }

    /// Get current particle count.
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Get positions.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Get velocities.
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Simulate one frame.
    pub fn simulate(&mut self, dt: f32) {
        let substep_dt = dt / self.params.substeps as f32;

        for _ in 0..self.params.substeps {
            self.substep(substep_dt);
        }
    }

    /// Single simulation substep.
    fn substep(&mut self, dt: f32) {
        let n = self.positions.len();
        if n == 0 {
            return;
        }

        // 1. Apply external forces and predict positions
        for i in 0..n {
            self.velocities[i] += self.params.gravity * dt;
            self.predicted_positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Build spatial hash
        self.spatial_hash.build(&self.predicted_positions);

        // 3. Solver iterations
        for _ in 0..self.params.solver_iterations {
            // Calculate density and lambda for each particle
            self.calculate_densities_and_lambdas();

            // Calculate position corrections
            self.calculate_delta_positions();

            // Apply corrections
            for i in 0..n {
                self.predicted_positions[i] += self.delta_positions[i];
            }
        }

        // 4. Update velocities and positions
        for i in 0..n {
            self.velocities[i] = (self.predicted_positions[i] - self.positions[i]) / dt;
            self.positions[i] = self.predicted_positions[i];
        }

        // 5. Apply viscosity (XSPH)
        if self.params.viscosity > 0.0 {
            self.apply_viscosity();
        }
    }

    /// Calculate densities and Lagrange multipliers (lambda).
    fn calculate_densities_and_lambdas(&mut self) {
        let h = self.params.smoothing_radius;
        let rho0 = self.params.rest_density;
        let mass = self.params.particle_mass;
        let epsilon = self.params.kernel_epsilon;
        let poly6_coeff = self.params.poly6_coefficient();
        let spiky_coeff = self.params.spiky_gradient_coefficient();

        for i in 0..self.positions.len() {
            let pos_i = self.predicted_positions[i];

            // Get neighbors
            let neighbors = self
                .spatial_hash
                .get_neighbors_filtered(pos_i, &self.predicted_positions, h);

            // Calculate density
            let mut density = 0.0;
            for &(_j, dist) in &neighbors {
                let r_sq = dist * dist;
                density += mass * SphKernels::poly6_with_coeff(r_sq, h, poly6_coeff);
            }
            self.densities[i] = density;

            // Calculate lambda using Eq. 11 from PBF paper
            let constraint = density / rho0 - 1.0;

            // Calculate gradient sum for denominator
            let mut sum_grad_sq = 0.0;
            let mut grad_i = Vec3::ZERO;

            for &(j, _) in &neighbors {
                if i == j {
                    continue;
                }
                let pos_j = self.predicted_positions[j];
                let r = pos_i - pos_j;
                let grad_j = SphKernels::spiky_gradient_with_coeff(r, h, spiky_coeff) / rho0;
                sum_grad_sq += grad_j.length_squared();
                grad_i += grad_j;
            }
            sum_grad_sq += grad_i.length_squared();

            // Lambda (Lagrange multiplier)
            self.lambdas[i] = -constraint / (sum_grad_sq + epsilon);
        }
    }

    /// Calculate position corrections (delta_p).
    fn calculate_delta_positions(&mut self) {
        let h = self.params.smoothing_radius;
        let rho0 = self.params.rest_density;
        let spiky_coeff = self.params.spiky_gradient_coefficient();

        // Tensile instability correction parameters (s_corr)
        let k_corr = 0.001; // Artificial pressure strength
        let delta_q = 0.1 * h; // Fraction of smoothing radius
        let n_corr = 4; // Power for artificial pressure

        // Precompute W(delta_q)
        let w_delta_q = SphKernels::poly6(delta_q * delta_q, h);

        for i in 0..self.positions.len() {
            let pos_i = self.predicted_positions[i];
            let lambda_i = self.lambdas[i];

            let neighbors = self
                .spatial_hash
                .get_neighbors_filtered(pos_i, &self.predicted_positions, h);

            let mut delta_p = Vec3::ZERO;

            for &(j, dist) in &neighbors {
                if i == j {
                    continue;
                }
                let pos_j = self.predicted_positions[j];
                let lambda_j = self.lambdas[j];

                // Artificial pressure (tensile instability correction)
                let w_ij = SphKernels::poly6(dist * dist, h);
                let s_corr = if w_delta_q > 1e-9 {
                    -k_corr * (w_ij / w_delta_q).powi(n_corr)
                } else {
                    0.0
                };

                let r = pos_i - pos_j;
                let grad = SphKernels::spiky_gradient_with_coeff(r, h, spiky_coeff);

                delta_p += (lambda_i + lambda_j + s_corr) * grad;
            }

            self.delta_positions[i] = delta_p / rho0;
        }
    }

    /// Apply XSPH viscosity.
    fn apply_viscosity(&mut self) {
        let h = self.params.smoothing_radius;
        let c = self.params.viscosity;
        let poly6_coeff = self.params.poly6_coefficient();

        let mut velocity_corrections = vec![Vec3::ZERO; self.positions.len()];

        for i in 0..self.positions.len() {
            let pos_i = self.positions[i];
            let vel_i = self.velocities[i];

            let neighbors = self
                .spatial_hash
                .get_neighbors_filtered(pos_i, &self.positions, h);

            let mut correction = Vec3::ZERO;

            for &(j, dist) in &neighbors {
                if i == j {
                    continue;
                }
                let vel_j = self.velocities[j];
                let w = SphKernels::poly6_with_coeff(dist * dist, h, poly6_coeff);
                correction += (vel_j - vel_i) * w;
            }

            velocity_corrections[i] = c * correction;
        }

        for i in 0..self.positions.len() {
            self.velocities[i] += velocity_corrections[i];
        }
    }

    /// Apply floor collision (simple boundary).
    pub fn apply_floor_collision(&mut self, floor_y: f32, restitution: f32) {
        for i in 0..self.positions.len() {
            if self.positions[i].y < floor_y {
                self.positions[i].y = floor_y;
                if self.velocities[i].y < 0.0 {
                    self.velocities[i].y *= -restitution;
                }
            }
        }
    }

    /// Apply box boundary collision.
    pub fn apply_box_boundary(&mut self, min: Vec3, max: Vec3, restitution: f32) {
        for i in 0..self.positions.len() {
            let mut pos = self.positions[i];
            let mut vel = self.velocities[i];

            // X bounds
            if pos.x < min.x {
                pos.x = min.x;
                if vel.x < 0.0 {
                    vel.x *= -restitution;
                }
            } else if pos.x > max.x {
                pos.x = max.x;
                if vel.x > 0.0 {
                    vel.x *= -restitution;
                }
            }

            // Y bounds
            if pos.y < min.y {
                pos.y = min.y;
                if vel.y < 0.0 {
                    vel.y *= -restitution;
                }
            } else if pos.y > max.y {
                pos.y = max.y;
                if vel.y > 0.0 {
                    vel.y *= -restitution;
                }
            }

            // Z bounds
            if pos.z < min.z {
                pos.z = min.z;
                if vel.z < 0.0 {
                    vel.z *= -restitution;
                }
            } else if pos.z > max.z {
                pos.z = max.z;
                if vel.z > 0.0 {
                    vel.z *= -restitution;
                }
            }

            self.positions[i] = pos;
            self.velocities[i] = vel;
        }
    }

    /// Get average density error (for debugging).
    pub fn average_density_error(&self) -> f32 {
        if self.densities.is_empty() {
            return 0.0;
        }
        let rho0 = self.params.rest_density;
        let sum: f32 = self
            .densities
            .iter()
            .map(|&d| (d - rho0).abs() / rho0)
            .sum();
        sum / self.densities.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly6_kernel() {
        let h = 1.0;

        // At r=0, kernel should be maximum
        let w_0 = SphKernels::poly6(0.0, h);
        assert!(w_0 > 0.0);

        // At r=h, kernel should be 0
        let w_h = SphKernels::poly6(h * h, h);
        assert!((w_h).abs() < 1e-6);

        // Kernel should decrease with distance
        let w_half = SphKernels::poly6(0.25 * h * h, h);
        assert!(w_half < w_0);
        assert!(w_half > w_h);
    }

    #[test]
    fn test_spiky_gradient() {
        let h = 1.0;

        // At r=0, gradient should be zero (undefined direction)
        let grad_0 = SphKernels::spiky_gradient(Vec3::ZERO, h);
        assert!(grad_0.length() < 1e-6);

        // Gradient should point away from neighbor
        let r = Vec3::new(0.5, 0.0, 0.0);
        let grad = SphKernels::spiky_gradient(r, h);
        assert!(grad.x < 0.0); // Points from i to j
    }

    #[test]
    fn test_cpu_solver_initialization() {
        let params = FluidParams::default();
        let spatial = SpatialHashConfig::for_domain(
            Vec3::splat(-5.0),
            Vec3::splat(5.0),
            params.smoothing_radius,
        );

        let mut solver = CpuPbfSolver::new(params, spatial);

        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
        ];
        let velocities = vec![Vec3::ZERO; 3];

        solver.set_particles(positions.clone(), velocities);

        assert_eq!(solver.particle_count(), 3);
        assert_eq!(solver.positions()[0], positions[0]);
    }
}
