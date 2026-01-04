// LBM D3Q19 Compute Shader for Nasal Sinus Airflow Simulation
// Implements streaming, BGK collision, and boundary conditions

// Simulation parameters
struct SimParams {
    grid_size: vec4<u32>,
    omega: f32,
    step: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var geometry: texture_storage_3d<r8uint, read>;
@group(0) @binding(2) var f_in: texture_storage_3d<rgba32float, read>;
@group(0) @binding(3) var f_out: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var velocity_out: texture_storage_3d<rgba32float, write>;

// D3Q19 lattice velocities
const E: array<vec3<i32>, 19> = array<vec3<i32>, 19>(
    vec3<i32>(0, 0, 0),    // 0: rest
    vec3<i32>(1, 0, 0),    // 1: +x
    vec3<i32>(-1, 0, 0),   // 2: -x
    vec3<i32>(0, 1, 0),    // 3: +y
    vec3<i32>(0, -1, 0),   // 4: -y
    vec3<i32>(0, 0, 1),    // 5: +z
    vec3<i32>(0, 0, -1),   // 6: -z
    vec3<i32>(1, 1, 0),    // 7: +x+y
    vec3<i32>(-1, 1, 0),   // 8: -x+y
    vec3<i32>(1, -1, 0),   // 9: +x-y
    vec3<i32>(-1, -1, 0),  // 10: -x-y
    vec3<i32>(1, 0, 1),    // 11: +x+z
    vec3<i32>(-1, 0, 1),   // 12: -x+z
    vec3<i32>(1, 0, -1),   // 13: +x-z
    vec3<i32>(-1, 0, -1),  // 14: -x-z
    vec3<i32>(0, 1, 1),    // 15: +y+z
    vec3<i32>(0, -1, 1),   // 16: -y+z
    vec3<i32>(0, 1, -1),   // 17: +y-z
    vec3<i32>(0, -1, -1),  // 18: -y-z
);

// D3Q19 weights
const W: array<f32, 19> = array<f32, 19>(
    1.0 / 3.0,   // rest
    1.0 / 18.0,  // face neighbors (6)
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,  // edge neighbors (12)
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
);

// Opposite direction indices for bounce-back
const OPP: array<u32, 19> = array<u32, 19>(
    0u,   // 0 -> 0
    2u,   // 1 -> 2
    1u,   // 2 -> 1
    4u,   // 3 -> 4
    3u,   // 4 -> 3
    6u,   // 5 -> 6
    5u,   // 6 -> 5
    10u,  // 7 -> 10
    9u,   // 8 -> 9
    8u,   // 9 -> 8
    7u,   // 10 -> 7
    14u,  // 11 -> 14
    13u,  // 12 -> 13
    12u,  // 13 -> 12
    11u,  // 14 -> 11
    18u,  // 15 -> 18
    17u,  // 16 -> 17
    16u,  // 17 -> 16
    15u,  // 18 -> 15
);

// Speed of sound squared (cs^2 = 1/3 for D3Q19)
const CS2: f32 = 1.0 / 3.0;

// Helper to read distribution function for direction i
fn read_f(pos: vec3<i32>, i: u32) -> f32 {
    let layer = i / 4u;
    let channel = i % 4u;
    let tex_pos = vec3<i32>(pos.x, pos.y, pos.z + i32(layer) * i32(params.grid_size.x));
    let data = textureLoad(f_in, tex_pos);

    switch channel {
        case 0u: { return data.x; }
        case 1u: { return data.y; }
        case 2u: { return data.z; }
        default: { return data.w; }
    }
}

// Helper to write distribution function for direction i
fn write_f(pos: vec3<i32>, i: u32, value: f32, current: ptr<function, array<vec4<f32>, 5>>) {
    let layer = i / 4u;
    let channel = i % 4u;

    switch channel {
        case 0u: { (*current)[layer].x = value; }
        case 1u: { (*current)[layer].y = value; }
        case 2u: { (*current)[layer].z = value; }
        default: { (*current)[layer].w = value; }
    }
}

// Compute equilibrium distribution
fn feq(i: u32, rho: f32, u: vec3<f32>) -> f32 {
    let ei = vec3<f32>(f32(E[i].x), f32(E[i].y), f32(E[i].z));
    let eu = dot(ei, u);
    let usq = dot(u, u);

    return W[i] * rho * (1.0 + eu / CS2 + (eu * eu) / (2.0 * CS2 * CS2) - usq / (2.0 * CS2));
}

// Check if position is within grid bounds
fn in_bounds(pos: vec3<i32>) -> bool {
    return pos.x >= 0 && pos.x < i32(params.grid_size.x) &&
           pos.y >= 0 && pos.y < i32(params.grid_size.y) &&
           pos.z >= 0 && pos.z < i32(params.grid_size.z);
}

// Check if cell is a wall
fn is_wall(pos: vec3<i32>) -> bool {
    if !in_bounds(pos) {
        return true;
    }
    return textureLoad(geometry, pos).r > 0u;
}

// Initialize distribution functions to equilibrium
#ifdef INIT
@compute @workgroup_size(8, 8, 8)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec3<i32>(global_id);

    if !in_bounds(pos) {
        return;
    }

    // Initial density and velocity
    let rho = 1.0;
    var u = vec3<f32>(0.0);

    // Apply inlet velocity at z=0
    if pos.z < 4 && !is_wall(pos) {
        u.z = 0.05;  // Inlet velocity
    }

    // Store equilibrium distributions
    var f_values: array<vec4<f32>, 5>;
    for (var i = 0u; i < 5u; i++) {
        f_values[i] = vec4<f32>(0.0);
    }

    for (var i = 0u; i < 19u; i++) {
        write_f(pos, i, feq(i, rho, u), &f_values);
    }

    // Write to texture
    for (var layer = 0u; layer < 5u; layer++) {
        let tex_pos = vec3<i32>(pos.x, pos.y, pos.z + i32(layer) * i32(params.grid_size.x));
        textureStore(f_out, tex_pos, f_values[layer]);
    }

    // Initialize velocity field
    textureStore(velocity_out, pos, vec4<f32>(u, rho));
}
#endif

// Main LBM kernel: streaming + collision
@compute @workgroup_size(8, 8, 8)
fn stream_collide(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec3<i32>(global_id);

    if !in_bounds(pos) {
        return;
    }

    // Skip wall cells
    if is_wall(pos) {
        // Zero velocity for walls
        textureStore(velocity_out, pos, vec4<f32>(0.0));
        return;
    }

    // Streaming step: gather from neighbors
    var f: array<f32, 19>;

    for (var i = 0u; i < 19u; i++) {
        let neighbor = pos - E[i];

        if is_wall(neighbor) {
            // Bounce-back boundary condition
            f[i] = read_f(pos, OPP[i]);
        } else if !in_bounds(neighbor) {
            // Open boundary - use equilibrium
            let is_inlet = neighbor.z < 0;
            let is_outlet = neighbor.z >= i32(params.grid_size.z);

            if is_inlet {
                // Inlet: prescribed velocity
                let rho_in = 1.0;
                let u_in = vec3<f32>(0.0, 0.0, 0.05);
                f[i] = feq(i, rho_in, u_in);
            } else if is_outlet {
                // Outlet: zero gradient (copy from current cell)
                f[i] = read_f(pos, i);
            } else {
                // Other boundaries: bounce-back
                f[i] = read_f(pos, OPP[i]);
            }
        } else {
            // Normal streaming
            f[i] = read_f(neighbor, i);
        }
    }

    // Compute macroscopic quantities
    var rho = 0.0;
    var u = vec3<f32>(0.0);

    for (var i = 0u; i < 19u; i++) {
        rho += f[i];
        u += f[i] * vec3<f32>(f32(E[i].x), f32(E[i].y), f32(E[i].z));
    }

    // Normalize velocity
    if rho > 0.0 {
        u /= rho;
    }

    // Collision step (BGK)
    var f_out_values: array<vec4<f32>, 5>;
    for (var i = 0u; i < 5u; i++) {
        f_out_values[i] = vec4<f32>(0.0);
    }

    for (var i = 0u; i < 19u; i++) {
        let f_eq = feq(i, rho, u);
        let f_new = f[i] - params.omega * (f[i] - f_eq);
        write_f(pos, i, f_new, &f_out_values);
    }

    // Write output distributions
    for (var layer = 0u; layer < 5u; layer++) {
        let tex_pos = vec3<i32>(pos.x, pos.y, pos.z + i32(layer) * i32(params.grid_size.x));
        textureStore(f_out, tex_pos, f_out_values[layer]);
    }

    // Write velocity field for visualization
    textureStore(velocity_out, pos, vec4<f32>(u, rho));
}
