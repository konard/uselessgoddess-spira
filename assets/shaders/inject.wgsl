// Fluid Injection Compute Shader
// Injects fluid at the syringe tip location

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 8u;

// D3Q19 weights
const W: array<f32, 19> = array<f32, 19>(
    1.0 / 3.0,                          // rest (0)
    1.0 / 18.0, 1.0 / 18.0,             // +-x (1-2)
    1.0 / 18.0, 1.0 / 18.0,             // +-y (3-4)
    1.0 / 18.0, 1.0 / 18.0,             // +-z (5-6)
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,  // xy edges (7-10)
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,  // xz edges (11-14)
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0   // yz edges (15-18)
);

// D3Q19 velocity directions
const E_X: array<i32, 19> = array<i32, 19>(
    0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0
);
const E_Y: array<i32, 19> = array<i32, 19>(
    0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1
);
const E_Z: array<i32, 19> = array<i32, 19>(
    0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1
);

const CS2: f32 = 1.0 / 3.0;
const CS4: f32 = 1.0 / 9.0;

// ============================================================================
// Uniforms and Bindings
// ============================================================================

struct FluidParams {
    viscosity: f32,
    gravity: vec3<f32>,
    injection_point: vec3<f32>,
    injection_velocity: vec3<f32>,
    injection_rate: f32,
    dt: f32,
    frame: u32,
    grid_size: u32,
}

// ReadWrite distribution textures for injection
@group(0) @binding(0) var dist_rw_0: texture_storage_3d<rgba32float, read_write>;
@group(0) @binding(1) var dist_rw_1: texture_storage_3d<rgba32float, read_write>;
@group(0) @binding(2) var dist_rw_2: texture_storage_3d<rgba32float, read_write>;
@group(0) @binding(3) var dist_rw_3: texture_storage_3d<rgba32float, read_write>;
@group(0) @binding(4) var dist_rw_4: texture_storage_3d<rgba32float, read_write>;
@group(0) @binding(5) var<uniform> params: FluidParams;

// ============================================================================
// Helper Functions
// ============================================================================

fn get_e_f(i: u32) -> vec3<f32> {
    return vec3<f32>(f32(E_X[i]), f32(E_Y[i]), f32(E_Z[i]));
}

fn in_bounds(pos: vec3<i32>, size: u32) -> bool {
    return pos.x >= 0 && pos.x < i32(size) &&
           pos.y >= 0 && pos.y < i32(size) &&
           pos.z >= 0 && pos.z < i32(size);
}

fn f_eq(i: u32, rho: f32, u: vec3<f32>) -> f32 {
    let e = get_e_f(i);
    let eu = dot(e, u);
    let u2 = dot(u, u);

    return W[i] * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS4) - u2 / (2.0 * CS2));
}

// ============================================================================
// Injection Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn inject_fluid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inject_center = vec3<i32>(params.injection_point);
    let pos = inject_center + vec3<i32>(gid) - vec3<i32>(WORKGROUP_SIZE / 2u);

    if !in_bounds(pos, params.grid_size) {
        return;
    }

    // Calculate distance from injection point
    let dist = length(vec3<f32>(pos) - params.injection_point);
    let inject_radius = 4.0;

    if dist > inject_radius {
        return;
    }

    // Gaussian injection profile
    let sigma = inject_radius * 0.5;
    let weight = exp(-dist * dist / (2.0 * sigma * sigma));

    // Read current distributions
    var f: array<f32, 19>;
    let f0 = textureLoad(dist_rw_0, pos);
    let f1 = textureLoad(dist_rw_1, pos);
    let f2 = textureLoad(dist_rw_2, pos);
    let f3 = textureLoad(dist_rw_3, pos);
    let f4 = textureLoad(dist_rw_4, pos);

    f[0] = f0.x; f[1] = f0.y; f[2] = f0.z; f[3] = f0.w;
    f[4] = f1.x; f[5] = f1.y; f[6] = f1.z; f[7] = f1.w;
    f[8] = f2.x; f[9] = f2.y; f[10] = f2.z; f[11] = f2.w;
    f[12] = f3.x; f[13] = f3.y; f[14] = f3.z; f[15] = f3.w;
    f[16] = f4.x; f[17] = f4.y; f[18] = f4.z;

    // Compute current macroscopic quantities
    var rho: f32 = 0.0;
    for (var i = 0u; i < 19u; i++) {
        rho += f[i];
    }
    rho = max(rho, 0.001);

    // Add injected fluid
    let inject_rho = params.injection_rate * weight;
    let inject_vel = params.injection_velocity;
    let new_rho = rho + inject_rho;

    // Blend velocity with injection velocity
    let blend = inject_rho / new_rho;
    var momentum = vec3<f32>(0.0);
    for (var i = 0u; i < 19u; i++) {
        momentum += f[i] * get_e_f(i);
    }
    let current_vel = momentum / rho;
    let new_vel = mix(current_vel, inject_vel, blend);

    // Recompute distributions for new state
    var f_new: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        f_new[i] = f_eq(i, new_rho, new_vel);
    }

    // Store updated distributions
    textureStore(dist_rw_0, pos, vec4<f32>(f_new[0], f_new[1], f_new[2], f_new[3]));
    textureStore(dist_rw_1, pos, vec4<f32>(f_new[4], f_new[5], f_new[6], f_new[7]));
    textureStore(dist_rw_2, pos, vec4<f32>(f_new[8], f_new[9], f_new[10], f_new[11]));
    textureStore(dist_rw_3, pos, vec4<f32>(f_new[12], f_new[13], f_new[14], f_new[15]));
    textureStore(dist_rw_4, pos, vec4<f32>(f_new[16], f_new[17], f_new[18], 0.0));
}
