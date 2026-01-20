//! Fluid simulation core logic.
//!
//! This module contains the PBF (Position Based Fluids) simulation implementation.
//! The simulation runs on CPU for the proof-of-concept, with GPU compute shader
//! support planned for future versions.

use bevy::prelude::*;

use super::params::FluidParams;

/// Resource for staging particle data before upload to simulation.
#[derive(Resource, Default)]
pub struct ParticleStagingBuffer {
    /// Particle positions (x, y, z, unused).
    pub positions: Vec<[f32; 4]>,
    /// Particle velocities (x, y, z, unused).
    pub velocities: Vec<[f32; 4]>,
    /// Whether new data needs to be uploaded.
    pub needs_upload: bool,
}

impl ParticleStagingBuffer {
    /// Adds a particle to the staging buffer.
    pub fn add_particle(&mut self, position: Vec3, velocity: Vec3) {
        self.positions
            .push([position.x, position.y, position.z, 0.0]);
        self.velocities
            .push([velocity.x, velocity.y, velocity.z, 0.0]);
        self.needs_upload = true;
    }

    /// Clears all staged particles.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.needs_upload = false;
    }
}

/// Main fluid simulation resource.
///
/// Contains particle state and simulation data.
#[derive(Resource)]
pub struct FluidSimulation {
    /// Particle positions (x, y, z, unused).
    pub positions: Vec<[f32; 4]>,
    /// Particle velocities (x, y, z, unused).
    pub velocities: Vec<[f32; 4]>,
    /// Predicted positions for constraint solving.
    predicted_positions: Vec<[f32; 4]>,
    /// Lambda values for density constraint.
    lambdas: Vec<f32>,
    /// Particle densities.
    densities: Vec<f32>,
    /// Delta position corrections.
    delta_positions: Vec<[f32; 4]>,
    /// Spatial hash grid (cell -> particle indices).
    grid: SpatialHash,
}

impl Default for FluidSimulation {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            predicted_positions: Vec::new(),
            lambdas: Vec::new(),
            densities: Vec::new(),
            delta_positions: Vec::new(),
            grid: SpatialHash::default(),
        }
    }
}

impl FluidSimulation {
    /// Creates a new simulation with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            predicted_positions: Vec::with_capacity(capacity),
            lambdas: Vec::with_capacity(capacity),
            densities: Vec::with_capacity(capacity),
            delta_positions: Vec::with_capacity(capacity),
            grid: SpatialHash::default(),
        }
    }

    /// Returns the number of particles.
    pub fn num_particles(&self) -> usize {
        self.positions.len()
    }

    /// Adds particles from the staging buffer.
    pub fn upload_from_staging(&mut self, staging: &mut ParticleStagingBuffer) {
        if !staging.needs_upload || staging.positions.is_empty() {
            return;
        }

        self.positions.extend_from_slice(&staging.positions);
        self.velocities.extend_from_slice(&staging.velocities);

        // Resize auxiliary buffers
        let n = self.positions.len();
        self.predicted_positions.resize(n, [0.0; 4]);
        self.lambdas.resize(n, 0.0);
        self.densities.resize(n, 0.0);
        self.delta_positions.resize(n, [0.0; 4]);

        staging.clear();
    }

    /// Runs one simulation step.
    pub fn step(&mut self, params: &FluidParams, dt: f32) {
        let n = self.positions.len();
        if n == 0 {
            return;
        }

        let substep_dt = dt / params.substeps as f32;

        for _ in 0..params.substeps {
            // 1. Apply external forces and predict positions
            self.apply_forces_and_predict(params, substep_dt);

            // 2. Build spatial hash
            self.grid.build(&self.predicted_positions, params);

            // 3. Solve density constraints (PBF core)
            for _ in 0..params.solver_iterations {
                self.compute_density_and_lambda(params);
                self.compute_delta_positions(params);
                self.apply_delta_positions();
            }

            // 4. Update velocities from position change
            self.update_velocities(substep_dt);

            // 5. Apply XSPH viscosity
            self.apply_viscosity(params);

            // 6. Commit predicted positions
            self.positions.copy_from_slice(&self.predicted_positions);
        }
    }

    fn apply_forces_and_predict(&mut self, params: &FluidParams, dt: f32) {
        let gravity = params.gravity;

        for i in 0..self.positions.len() {
            // Apply gravity
            self.velocities[i][0] += gravity.x * dt;
            self.velocities[i][1] += gravity.y * dt;
            self.velocities[i][2] += gravity.z * dt;

            // Predict position
            self.predicted_positions[i][0] = self.positions[i][0] + self.velocities[i][0] * dt;
            self.predicted_positions[i][1] = self.positions[i][1] + self.velocities[i][1] * dt;
            self.predicted_positions[i][2] = self.positions[i][2] + self.velocities[i][2] * dt;
            self.predicted_positions[i][3] = 0.0;

            // Boundary collision
            self.enforce_boundary(i, params);
        }
    }

    fn enforce_boundary(&mut self, i: usize, params: &FluidParams) {
        let restitution = params.boundary_restitution;
        let pos = &mut self.predicted_positions[i];
        let vel = &mut self.velocities[i];
        let min = params.bounds_min;
        let max = params.bounds_max;

        // X bounds
        if pos[0] < min.x {
            pos[0] = min.x;
            vel[0] = -vel[0] * restitution;
        } else if pos[0] > max.x {
            pos[0] = max.x;
            vel[0] = -vel[0] * restitution;
        }

        // Y bounds
        if pos[1] < min.y {
            pos[1] = min.y;
            vel[1] = -vel[1] * restitution;
        } else if pos[1] > max.y {
            pos[1] = max.y;
            vel[1] = -vel[1] * restitution;
        }

        // Z bounds
        if pos[2] < min.z {
            pos[2] = min.z;
            vel[2] = -vel[2] * restitution;
        } else if pos[2] > max.z {
            pos[2] = max.z;
            vel[2] = -vel[2] * restitution;
        }
    }

    fn compute_density_and_lambda(&mut self, params: &FluidParams) {
        let h = params.smoothing_radius;
        let h2 = h * h;
        let rest_density = params.rest_density;
        let epsilon = params.relaxation_epsilon;

        // Poly6 kernel coefficient
        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
        // Spiky gradient coefficient
        let spiky_grad_coeff = -45.0 / (std::f32::consts::PI * h.powi(6));

        let particle_mass = params.particle_mass();

        for i in 0..self.positions.len() {
            let pi = Vec3::new(
                self.predicted_positions[i][0],
                self.predicted_positions[i][1],
                self.predicted_positions[i][2],
            );

            let mut density = 0.0;
            let mut grad_sum_sq = 0.0;
            let mut grad_i = Vec3::ZERO;

            // Find neighbors
            let neighbors = self.grid.get_neighbors(i, &self.predicted_positions, params);

            for &j in &neighbors {
                let pj = Vec3::new(
                    self.predicted_positions[j][0],
                    self.predicted_positions[j][1],
                    self.predicted_positions[j][2],
                );

                let r = pi - pj;
                let r_len_sq = r.length_squared();

                if r_len_sq < h2 && r_len_sq > 1e-12 {
                    let r_len = r_len_sq.sqrt();
                    let h_minus_r = h - r_len;

                    // Poly6 kernel for density
                    let w = poly6_coeff * (h2 - r_len_sq).powi(3);
                    density += particle_mass * w;

                    // Spiky gradient for constraint
                    let grad_w = spiky_grad_coeff * h_minus_r * h_minus_r / r_len;
                    let grad = r.normalize_or_zero() * grad_w / rest_density;

                    grad_i += grad;
                    grad_sum_sq += grad.length_squared();
                }
            }

            // Self-contribution to density
            density += particle_mass * poly6_coeff * h2.powi(3);

            self.densities[i] = density;

            // Constraint value
            let constraint = density / rest_density - 1.0;

            // Lambda (Lagrange multiplier)
            let denom = grad_sum_sq + grad_i.length_squared() + epsilon;
            self.lambdas[i] = -constraint / denom;
        }
    }

    fn compute_delta_positions(&mut self, params: &FluidParams) {
        let h = params.smoothing_radius;
        let h2 = h * h;
        let rest_density = params.rest_density;
        let spiky_grad_coeff = -45.0 / (std::f32::consts::PI * h.powi(6));

        // Surface tension parameters
        let k = params.surface_tension;
        let delta_q = 0.1 * h;
        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
        let w_delta_q = poly6_coeff * (h2 - delta_q * delta_q).powi(3);

        for i in 0..self.positions.len() {
            let pi = Vec3::new(
                self.predicted_positions[i][0],
                self.predicted_positions[i][1],
                self.predicted_positions[i][2],
            );

            let mut delta = Vec3::ZERO;
            let lambda_i = self.lambdas[i];

            let neighbors = self.grid.get_neighbors(i, &self.predicted_positions, params);

            for &j in &neighbors {
                if i == j {
                    continue;
                }

                let pj = Vec3::new(
                    self.predicted_positions[j][0],
                    self.predicted_positions[j][1],
                    self.predicted_positions[j][2],
                );

                let r = pi - pj;
                let r_len_sq = r.length_squared();

                if r_len_sq < h2 && r_len_sq > 1e-12 {
                    let r_len = r_len_sq.sqrt();
                    let h_minus_r = h - r_len;

                    // Spiky gradient
                    let grad_w = spiky_grad_coeff * h_minus_r * h_minus_r / r_len;
                    let grad = r.normalize_or_zero() * grad_w;

                    // Surface tension (artificial pressure)
                    let w = poly6_coeff * (h2 - r_len_sq).powi(3);
                    let s_corr = -k * (w / w_delta_q).powi(4);

                    delta += (lambda_i + self.lambdas[j] + s_corr) * grad;
                }
            }

            let d = delta / rest_density;
            self.delta_positions[i] = [d.x, d.y, d.z, 0.0];
        }
    }

    fn apply_delta_positions(&mut self) {
        for i in 0..self.positions.len() {
            self.predicted_positions[i][0] += self.delta_positions[i][0];
            self.predicted_positions[i][1] += self.delta_positions[i][1];
            self.predicted_positions[i][2] += self.delta_positions[i][2];
        }
    }

    fn update_velocities(&mut self, dt: f32) {
        let inv_dt = 1.0 / dt;

        for i in 0..self.positions.len() {
            self.velocities[i][0] =
                (self.predicted_positions[i][0] - self.positions[i][0]) * inv_dt;
            self.velocities[i][1] =
                (self.predicted_positions[i][1] - self.positions[i][1]) * inv_dt;
            self.velocities[i][2] =
                (self.predicted_positions[i][2] - self.positions[i][2]) * inv_dt;
        }
    }

    fn apply_viscosity(&mut self, params: &FluidParams) {
        let h = params.smoothing_radius;
        let h2 = h * h;
        let c = params.viscosity;

        if c < 1e-6 {
            return;
        }

        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));

        // Store original velocities for XSPH
        let velocities_copy = self.velocities.clone();

        for i in 0..self.positions.len() {
            let pi = Vec3::new(
                self.predicted_positions[i][0],
                self.predicted_positions[i][1],
                self.predicted_positions[i][2],
            );
            let vi = Vec3::new(
                velocities_copy[i][0],
                velocities_copy[i][1],
                velocities_copy[i][2],
            );

            let mut velocity_correction = Vec3::ZERO;

            let neighbors = self.grid.get_neighbors(i, &self.predicted_positions, params);

            for &j in &neighbors {
                if i == j {
                    continue;
                }

                let pj = Vec3::new(
                    self.predicted_positions[j][0],
                    self.predicted_positions[j][1],
                    self.predicted_positions[j][2],
                );
                let vj = Vec3::new(
                    velocities_copy[j][0],
                    velocities_copy[j][1],
                    velocities_copy[j][2],
                );

                let r = pi - pj;
                let r_len_sq = r.length_squared();

                if r_len_sq < h2 {
                    // Poly6 kernel
                    let w = poly6_coeff * (h2 - r_len_sq).powi(3);
                    velocity_correction += (vj - vi) * w;
                }
            }

            self.velocities[i][0] += c * velocity_correction.x;
            self.velocities[i][1] += c * velocity_correction.y;
            self.velocities[i][2] += c * velocity_correction.z;
        }
    }
}

/// Spatial hash grid for neighbor search.
#[derive(Default)]
struct SpatialHash {
    /// Cell size.
    cell_size: f32,
    /// Grid dimensions.
    grid_size: UVec3,
    /// Bounds minimum.
    bounds_min: Vec3,
    /// Cell to particle indices mapping.
    cells: Vec<Vec<usize>>,
}

impl SpatialHash {
    fn build(&mut self, positions: &[[f32; 4]], params: &FluidParams) {
        self.cell_size = params.cell_size();
        self.grid_size = params.grid_size;
        self.bounds_min = params.bounds_min;

        let total_cells =
            self.grid_size.x as usize * self.grid_size.y as usize * self.grid_size.z as usize;

        // Clear and resize cells
        self.cells.clear();
        self.cells.resize(total_cells, Vec::new());

        // Insert particles into cells
        for (i, pos) in positions.iter().enumerate() {
            let cell_idx = self.position_to_cell_index(pos);
            if cell_idx < total_cells {
                self.cells[cell_idx].push(i);
            }
        }
    }

    fn position_to_cell_index(&self, pos: &[f32; 4]) -> usize {
        let local = Vec3::new(
            pos[0] - self.bounds_min.x,
            pos[1] - self.bounds_min.y,
            pos[2] - self.bounds_min.z,
        );

        let cell_x = ((local.x / self.cell_size) as u32).min(self.grid_size.x - 1);
        let cell_y = ((local.y / self.cell_size) as u32).min(self.grid_size.y - 1);
        let cell_z = ((local.z / self.cell_size) as u32).min(self.grid_size.z - 1);

        (cell_z * self.grid_size.y * self.grid_size.x + cell_y * self.grid_size.x + cell_x) as usize
    }

    fn get_neighbors(
        &self,
        particle_idx: usize,
        positions: &[[f32; 4]],
        params: &FluidParams,
    ) -> Vec<usize> {
        let mut neighbors = Vec::with_capacity(64);
        let pos = &positions[particle_idx];
        let h2 = params.smoothing_radius * params.smoothing_radius;

        // Get cell coordinates
        let local = Vec3::new(
            pos[0] - self.bounds_min.x,
            pos[1] - self.bounds_min.y,
            pos[2] - self.bounds_min.z,
        );

        let cell_x = (local.x / self.cell_size) as i32;
        let cell_y = (local.y / self.cell_size) as i32;
        let cell_z = (local.z / self.cell_size) as i32;

        // Search 3x3x3 neighborhood
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = cell_x + dx;
                    let ny = cell_y + dy;
                    let nz = cell_z + dz;

                    if nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= self.grid_size.x as i32
                        || ny >= self.grid_size.y as i32
                        || nz >= self.grid_size.z as i32
                    {
                        continue;
                    }

                    let cell_idx = (nz as u32 * self.grid_size.y * self.grid_size.x
                        + ny as u32 * self.grid_size.x
                        + nx as u32) as usize;

                    if cell_idx < self.cells.len() {
                        for &j in &self.cells[cell_idx] {
                            let pj = &positions[j];
                            let dx = pos[0] - pj[0];
                            let dy = pos[1] - pj[1];
                            let dz = pos[2] - pj[2];
                            let dist_sq = dx * dx + dy * dy + dz * dz;

                            if dist_sq < h2 {
                                neighbors.push(j);
                            }
                        }
                    }
                }
            }
        }

        neighbors
    }
}

/// Spawns a block of particles in the staging buffer.
pub fn spawn_particle_block(
    staging: &mut ParticleStagingBuffer,
    center: Vec3,
    half_extents: Vec3,
    spacing: f32,
    initial_velocity: Vec3,
) {
    let min = center - half_extents;
    let max = center + half_extents;

    let mut x = min.x;
    while x <= max.x {
        let mut y = min.y;
        while y <= max.y {
            let mut z = min.z;
            while z <= max.z {
                staging.add_particle(Vec3::new(x, y, z), initial_velocity);
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }
}
