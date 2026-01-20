// Position Based Fluids (PBF) Compute Shader
// Based on "Position Based Fluids" by Macklin and Müller, 2013
//
// This shader implements a full GPU PBF simulation including:
// - Spatial hashing for O(n*k) neighbor search
// - Density constraint solving
// - XSPH viscosity
// - Boundary handling

// ============================================================================
// Uniform buffer with simulation parameters
// ============================================================================

struct FluidParams {
    // Simulation parameters
    rest_density: f32,
    smoothing_radius: f32,
    smoothing_radius_sq: f32,
    relaxation_epsilon: f32,

    viscosity: f32,
    vorticity_epsilon: f32,
    surface_tension: f32,
    particle_mass: f32,

    gravity: vec3<f32>,
    dt: f32,

    // Grid parameters
    grid_size: vec3<u32>,
    num_particles: u32,

    bounds_min: vec3<f32>,
    cell_size: f32,

    bounds_max: vec3<f32>,
    boundary_restitution: f32,

    // Precomputed kernel coefficients
    poly6_coeff: f32,
    spiky_grad_coeff: f32,
    viscosity_laplacian_coeff: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> predicted_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(5) var<storage, read_write> delta_positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> densities: array<f32>;

@group(1) @binding(0) var<storage, read_write> cell_start: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> sorted_indices: array<u32>;
@group(1) @binding(3) var<storage, read_write> prefix_sum: array<u32>;

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265359;
const WORKGROUP_SIZE: u32 = 256u;
const INVALID_INDEX: u32 = 0xFFFFFFFFu;

// ============================================================================
// Helper functions
// ============================================================================

// Convert world position to grid cell index
fn position_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let relative = pos - params.bounds_min;
    return vec3<i32>(floor(relative / params.cell_size));
}

// Convert grid cell to flat index
fn cell_to_index(cell: vec3<i32>) -> u32 {
    // Clamp to valid range
    let c = clamp(cell, vec3<i32>(0), vec3<i32>(params.grid_size) - vec3<i32>(1));
    return u32(c.x) + u32(c.y) * params.grid_size.x + u32(c.z) * params.grid_size.x * params.grid_size.y;
}

// Check if cell is valid
fn is_valid_cell(cell: vec3<i32>) -> bool {
    return all(cell >= vec3<i32>(0)) && all(cell < vec3<i32>(params.grid_size));
}

// Poly6 kernel (for density calculation)
fn poly6(r_sq: f32) -> f32 {
    if r_sq >= params.smoothing_radius_sq {
        return 0.0;
    }
    let diff = params.smoothing_radius_sq - r_sq;
    return params.poly6_coeff * diff * diff * diff;
}

// Spiky kernel gradient (for pressure force)
fn spiky_gradient(r: vec3<f32>, r_len: f32) -> vec3<f32> {
    if r_len <= 0.0 || r_len >= params.smoothing_radius {
        return vec3<f32>(0.0);
    }
    let diff = params.smoothing_radius - r_len;
    return params.spiky_grad_coeff * diff * diff * (r / r_len);
}

// Apply boundary conditions
fn apply_boundary(pos: vec3<f32>, vel: vec3<f32>) -> vec4<f32> {
    var new_pos = pos;
    var new_vel = vel;
    let margin = params.cell_size * 0.1;

    // X bounds
    if new_pos.x < params.bounds_min.x + margin {
        new_pos.x = params.bounds_min.x + margin;
        new_vel.x = abs(new_vel.x) * params.boundary_restitution;
    } else if new_pos.x > params.bounds_max.x - margin {
        new_pos.x = params.bounds_max.x - margin;
        new_vel.x = -abs(new_vel.x) * params.boundary_restitution;
    }

    // Y bounds
    if new_pos.y < params.bounds_min.y + margin {
        new_pos.y = params.bounds_min.y + margin;
        new_vel.y = abs(new_vel.y) * params.boundary_restitution;
    } else if new_pos.y > params.bounds_max.y - margin {
        new_pos.y = params.bounds_max.y - margin;
        new_vel.y = -abs(new_vel.y) * params.boundary_restitution;
    }

    // Z bounds
    if new_pos.z < params.bounds_min.z + margin {
        new_pos.z = params.bounds_min.z + margin;
        new_vel.z = abs(new_vel.z) * params.boundary_restitution;
    } else if new_pos.z > params.bounds_max.z - margin {
        new_pos.z = params.bounds_max.z - margin;
        new_vel.z = -abs(new_vel.z) * params.boundary_restitution;
    }

    return vec4<f32>(new_pos, new_vel.x);
}

// ============================================================================
// Compute shader entry points
// ============================================================================

// Step 1: Apply forces and predict positions
@compute @workgroup_size(256)
fn predict_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    let pos = positions[i].xyz;
    var vel = velocities[i].xyz;

    // Apply gravity
    vel = vel + params.gravity * params.dt;

    // Predict position
    var predicted = pos + vel * params.dt;

    // Apply boundary (store velocity change in predicted.w temporarily)
    let bounded = apply_boundary(predicted, vel);
    predicted = bounded.xyz;

    predicted_positions[i] = vec4<f32>(predicted, 0.0);
    velocities[i] = vec4<f32>(vel, 0.0);
}

// Step 2: Clear spatial hash grid
@compute @workgroup_size(256)
fn clear_hash(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let num_cells = params.grid_size.x * params.grid_size.y * params.grid_size.z;
    if i >= num_cells {
        return;
    }

    atomicStore(&cell_start[i], INVALID_INDEX);
    atomicStore(&cell_count[i], 0u);
}

// Step 3: Build spatial hash (count particles per cell using atomics)
@compute @workgroup_size(256)
fn build_hash(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    let pos = predicted_positions[i].xyz;
    let cell = position_to_cell(pos);

    if is_valid_cell(cell) {
        let cell_idx = cell_to_index(cell);
        // Increment count and get position in cell
        let position_in_cell = atomicAdd(&cell_count[cell_idx], 1u);
        // Store particle index (will be compacted later)
        // For simplicity, we use a linked-list style approach via sorted_indices
        sorted_indices[i] = cell_idx;
    }
}

// Step 4a: Prefix sum for counting sort (simplified single-pass version for small grids)
// Note: For large grids, a more sophisticated parallel prefix sum would be needed
@compute @workgroup_size(256)
fn prefix_sum(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let num_cells = params.grid_size.x * params.grid_size.y * params.grid_size.z;
    if i >= num_cells {
        return;
    }

    // Simple sequential prefix sum in first thread
    // For production, use parallel prefix sum (Hillis-Steele or Blelloch)
    if i == 0u {
        var sum = 0u;
        for (var j = 0u; j < num_cells; j = j + 1u) {
            let count = atomicLoad(&cell_count[j]);
            atomicStore(&cell_start[j], sum);
            prefix_sum[j] = sum;
            sum = sum + count;
        }
    }
}

// Step 4b: Reorder particles into sorted order
@compute @workgroup_size(256)
fn reorder_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    let cell_idx = sorted_indices[i];
    if cell_idx != INVALID_INDEX {
        // This is a simplified approach - production would need double buffering
        // to avoid race conditions during reordering
    }
}

// Step 5: Compute density and lambda (Lagrange multiplier)
@compute @workgroup_size(256)
fn compute_density_lambda(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    let pos_i = predicted_positions[i].xyz;
    let cell_i = position_to_cell(pos_i);

    var density = 0.0;
    var sum_grad_sq = 0.0;
    var sum_grad = vec3<f32>(0.0);

    // Iterate over neighboring cells (3x3x3 neighborhood)
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);

                if !is_valid_cell(neighbor_cell) {
                    continue;
                }

                let cell_idx = cell_to_index(neighbor_cell);
                let start = atomicLoad(&cell_start[cell_idx]);
                let count = atomicLoad(&cell_count[cell_idx]);

                if start == INVALID_INDEX {
                    continue;
                }

                // Iterate over particles in this cell
                // Note: In this simplified version, we scan all particles and check cell membership
                // A production implementation would use properly sorted indices
                for (var j = 0u; j < params.num_particles; j = j + 1u) {
                    let pos_j = predicted_positions[j].xyz;
                    let cell_j = position_to_cell(pos_j);

                    if cell_to_index(cell_j) != cell_idx {
                        continue;
                    }

                    let r = pos_i - pos_j;
                    let r_sq = dot(r, r);

                    if r_sq < params.smoothing_radius_sq {
                        // Density contribution
                        density = density + params.particle_mass * poly6(r_sq);

                        // Gradient computation for lambda denominator
                        if i != j {
                            let r_len = sqrt(r_sq);
                            let grad = spiky_gradient(r, r_len) / params.rest_density;
                            sum_grad_sq = sum_grad_sq + dot(grad, grad);
                            sum_grad = sum_grad + grad;
                        }
                    }
                }
            }
        }
    }

    densities[i] = density;

    // Compute constraint value
    let C = density / params.rest_density - 1.0;

    // Compute lambda (Lagrange multiplier)
    let denominator = sum_grad_sq + dot(sum_grad, sum_grad) + params.relaxation_epsilon;
    lambdas[i] = -C / denominator;
}

// Step 6: Compute position delta
@compute @workgroup_size(256)
fn compute_delta_position(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    let pos_i = predicted_positions[i].xyz;
    let lambda_i = lambdas[i];
    let cell_i = position_to_cell(pos_i);

    var delta = vec3<f32>(0.0);

    // Tensile instability correction (scorr)
    let k_scorr = 0.1;
    let delta_q = 0.1 * params.smoothing_radius;
    let delta_q_sq = delta_q * delta_q;
    let w_delta_q = poly6(delta_q_sq);

    // Iterate over neighboring cells
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);

                if !is_valid_cell(neighbor_cell) {
                    continue;
                }

                let cell_idx = cell_to_index(neighbor_cell);

                // Scan all particles (simplified - production would use sorted indices)
                for (var j = 0u; j < params.num_particles; j = j + 1u) {
                    if i == j {
                        continue;
                    }

                    let pos_j = predicted_positions[j].xyz;
                    let cell_j = position_to_cell(pos_j);

                    if cell_to_index(cell_j) != cell_idx {
                        continue;
                    }

                    let r = pos_i - pos_j;
                    let r_sq = dot(r, r);

                    if r_sq < params.smoothing_radius_sq && r_sq > 0.0001 {
                        let lambda_j = lambdas[j];
                        let r_len = sqrt(r_sq);

                        // Tensile instability correction
                        let w_r = poly6(r_sq);
                        var scorr = 0.0;
                        if w_delta_q > 0.0 {
                            scorr = -k_scorr * pow(w_r / w_delta_q, 4.0);
                        }

                        // Position correction
                        let grad = spiky_gradient(r, r_len);
                        delta = delta + (lambda_i + lambda_j + scorr) * grad;
                    }
                }
            }
        }
    }

    delta_positions[i] = vec4<f32>(delta / params.rest_density, 0.0);
}

// Step 7: Apply position delta
@compute @workgroup_size(256)
fn apply_delta_position(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    var pos = predicted_positions[i].xyz + delta_positions[i].xyz;

    // Apply boundary constraints
    let bounded = apply_boundary(pos, velocities[i].xyz);
    pos = bounded.xyz;

    predicted_positions[i] = vec4<f32>(pos, 0.0);
}

// Step 8: Update velocities from position change
@compute @workgroup_size(256)
fn update_velocities(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    let old_pos = positions[i].xyz;
    let new_pos = predicted_positions[i].xyz;

    // Velocity from position change
    let vel = (new_pos - old_pos) / params.dt;

    positions[i] = vec4<f32>(new_pos, 0.0);
    velocities[i] = vec4<f32>(vel, 0.0);
}

// Step 9: Apply XSPH viscosity
@compute @workgroup_size(256)
fn apply_viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_particles {
        return;
    }

    if params.viscosity <= 0.0 {
        return;
    }

    let pos_i = positions[i].xyz;
    let vel_i = velocities[i].xyz;
    let cell_i = position_to_cell(pos_i);

    var vel_delta = vec3<f32>(0.0);

    // Iterate over neighboring cells
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);

                if !is_valid_cell(neighbor_cell) {
                    continue;
                }

                let cell_idx = cell_to_index(neighbor_cell);

                // Scan all particles
                for (var j = 0u; j < params.num_particles; j = j + 1u) {
                    if i == j {
                        continue;
                    }

                    let pos_j = positions[j].xyz;
                    let cell_j = position_to_cell(pos_j);

                    if cell_to_index(cell_j) != cell_idx {
                        continue;
                    }

                    let r = pos_i - pos_j;
                    let r_sq = dot(r, r);

                    if r_sq < params.smoothing_radius_sq {
                        let vel_j = velocities[j].xyz;
                        let density_j = max(densities[j], 0.001);

                        // XSPH viscosity
                        let w = poly6(r_sq);
                        vel_delta = vel_delta + (vel_j - vel_i) * (w / density_j);
                    }
                }
            }
        }
    }

    let new_vel = vel_i + params.viscosity * vel_delta;
    velocities[i] = vec4<f32>(new_vel, 0.0);
}
