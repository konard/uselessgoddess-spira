// Lattice Boltzmann Method (LBM) D3Q19 Compute Shader
// For dental sinus fluid simulation
//
// D3Q19 uses 19 velocity directions in 3D:
// - 1 rest velocity (0,0,0)
// - 6 face velocities (+-1,0,0), (0,+-1,0), (0,0,+-1)
// - 12 edge velocities (+-1,+-1,0), (+-1,0,+-1), (0,+-1,+-1)

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 8u;

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

// D3Q19 weights
const W: array<f32, 19> = array<f32, 19>(
    0.333333333,  // rest (0)  = 1/3
    0.055555556, 0.055555556,  // +-x (1-2)  = 1/18
    0.055555556, 0.055555556,  // +-y (3-4)
    0.055555556, 0.055555556,  // +-z (5-6)
    0.027777778, 0.027777778, 0.027777778, 0.027777778,  // xy edges (7-10) = 1/36
    0.027777778, 0.027777778, 0.027777778, 0.027777778,  // xz edges (11-14)
    0.027777778, 0.027777778, 0.027777778, 0.027777778   // yz edges (15-18)
);

// Opposite direction indices for bounce-back
const OPP: array<u32, 19> = array<u32, 19>(
    0u,  2u,  1u,  4u,  3u,  6u,  5u,
    10u, 9u, 8u, 7u,
    14u, 13u, 12u, 11u,
    18u, 17u, 16u, 15u
);

// Speed of sound squared (cs^2 = 1/3 for D3Q19)
const CS2: f32 = 0.333333333;
const INV_CS2: f32 = 3.0;
const INV_2CS4: f32 = 4.5;

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

// Input distribution textures (5 x RGBA = 20 floats, using 19)
@group(0) @binding(0) var dist_in_0: texture_storage_3d<rgba32float, read>;
@group(0) @binding(1) var dist_in_1: texture_storage_3d<rgba32float, read>;
@group(0) @binding(2) var dist_in_2: texture_storage_3d<rgba32float, read>;
@group(0) @binding(3) var dist_in_3: texture_storage_3d<rgba32float, read>;
@group(0) @binding(4) var dist_in_4: texture_storage_3d<rgba32float, read>;

// Output distribution textures
@group(0) @binding(5) var dist_out_0: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var dist_out_1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var dist_out_2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var dist_out_3: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var dist_out_4: texture_storage_3d<rgba32float, write>;

// Density output
@group(0) @binding(10) var density_out: texture_storage_3d<r32float, write>;

// Velocity output
@group(0) @binding(11) var velocity_out: texture_storage_3d<rgba32float, write>;

// Boundaries (SDF: negative = solid, positive = fluid)
@group(0) @binding(12) var boundaries: texture_storage_3d<r32float, read>;

// Fluid parameters
@group(0) @binding(13) var<uniform> params: FluidParams;

// ============================================================================
// Helper Functions
// ============================================================================

// Load distribution function at index i from the 5 textures
fn load_f(pos: vec3<i32>, i: u32) -> f32 {
    let tex_idx = i / 4u;
    let comp_idx = i % 4u;

    var val: vec4<f32>;
    switch tex_idx {
        case 0u: { val = textureLoad(dist_in_0, pos); }
        case 1u: { val = textureLoad(dist_in_1, pos); }
        case 2u: { val = textureLoad(dist_in_2, pos); }
        case 3u: { val = textureLoad(dist_in_3, pos); }
        case 4u: { val = textureLoad(dist_in_4, pos); }
        default: { val = vec4<f32>(0.0); }
    }

    switch comp_idx {
        case 0u: { return val.x; }
        case 1u: { return val.y; }
        case 2u: { return val.z; }
        case 3u: { return val.w; }
        default: { return 0.0; }
    }
}

// Get velocity direction for index i
fn get_e(i: u32) -> vec3<i32> {
    return vec3<i32>(E_X[i], E_Y[i], E_Z[i]);
}

// Get velocity direction as float for index i
fn get_e_f(i: u32) -> vec3<f32> {
    return vec3<f32>(f32(E_X[i]), f32(E_Y[i]), f32(E_Z[i]));
}

// Check if position is within bounds
fn in_bounds(pos: vec3<i32>, size: u32) -> bool {
    return pos.x >= 0 && pos.x < i32(size) &&
           pos.y >= 0 && pos.y < i32(size) &&
           pos.z >= 0 && pos.z < i32(size);
}

// Check if position is solid (bone)
fn is_solid(pos: vec3<i32>) -> bool {
    if !in_bounds(pos, params.grid_size) {
        return true; // Outside bounds = solid
    }
    let sdf = textureLoad(boundaries, pos).r;
    return sdf < 0.0;
}

// Compute equilibrium distribution with improved numerical stability
fn f_eq(i: u32, rho: f32, u: vec3<f32>) -> f32 {
    let e = get_e_f(i);
    let eu = dot(e, u);
    let u2 = dot(u, u);

    // Ensure stability by clamping velocity magnitude
    let u2_clamped = min(u2, 0.1);

    // f_eq = w_i * rho * (1 + e.u/cs^2 + (e.u)^2/(2*cs^4) - u^2/(2*cs^2))
    return W[i] * rho * (1.0 + INV_CS2 * eu + INV_2CS4 * eu * eu - 1.5 * u2_clamped);
}

// ============================================================================
// Initialization Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = vec3<i32>(gid);
    let size = params.grid_size;

    if !in_bounds(pos, size) {
        return;
    }

    // Initialize to equilibrium with zero velocity
    // Only initialize fluid cells (not bone)
    let solid = is_solid(pos);
    let rho = select(1.0, 0.0, solid);
    let u = vec3<f32>(0.0);

    // Compute equilibrium distributions
    var f: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        f[i] = f_eq(i, rho, u);
    }

    // Store in output textures
    textureStore(dist_out_0, pos, vec4<f32>(f[0], f[1], f[2], f[3]));
    textureStore(dist_out_1, pos, vec4<f32>(f[4], f[5], f[6], f[7]));
    textureStore(dist_out_2, pos, vec4<f32>(f[8], f[9], f[10], f[11]));
    textureStore(dist_out_3, pos, vec4<f32>(f[12], f[13], f[14], f[15]));
    textureStore(dist_out_4, pos, vec4<f32>(f[16], f[17], f[18], 0.0));

    // Initialize density and velocity
    textureStore(density_out, pos, vec4<f32>(rho, 0.0, 0.0, 0.0));
    textureStore(velocity_out, pos, vec4<f32>(u, 0.0));
}

// ============================================================================
// Collide and Stream Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn collide_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = vec3<i32>(gid);
    let size = params.grid_size;

    if !in_bounds(pos, size) {
        return;
    }

    // Skip solid cells
    if is_solid(pos) {
        textureStore(density_out, pos, vec4<f32>(0.0));
        textureStore(velocity_out, pos, vec4<f32>(0.0));
        textureStore(dist_out_0, pos, vec4<f32>(0.0));
        textureStore(dist_out_1, pos, vec4<f32>(0.0));
        textureStore(dist_out_2, pos, vec4<f32>(0.0));
        textureStore(dist_out_3, pos, vec4<f32>(0.0));
        textureStore(dist_out_4, pos, vec4<f32>(0.0));
        return;
    }

    // ========================================
    // STREAMING: Pull distributions from neighbors
    // ========================================
    var f: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let e = get_e(i);
        let neighbor_pos = pos - e; // Pull from opposite direction

        if is_solid(neighbor_pos) {
            // Bounce-back from solid: use opposite direction from current cell
            f[i] = load_f(pos, OPP[i]);
        } else if !in_bounds(neighbor_pos, size) {
            // Outside boundary: use equilibrium at rest
            f[i] = f_eq(i, 1.0, vec3<f32>(0.0));
        } else {
            // Normal streaming
            f[i] = load_f(neighbor_pos, i);
        }
    }

    // ========================================
    // COMPUTE MACROSCOPIC QUANTITIES
    // ========================================
    var rho: f32 = 0.0;
    var momentum: vec3<f32> = vec3<f32>(0.0);

    for (var i = 0u; i < 19u; i++) {
        rho += f[i];
        momentum += f[i] * get_e_f(i);
    }

    // Clamp density to valid range for stability
    rho = clamp(rho, 0.5, 2.0);
    var u = momentum / rho;

    // Clamp velocity for stability
    let speed = length(u);
    if speed > 0.1 {
        u = u * (0.1 / speed);
    }

    // ========================================
    // ADD GRAVITY FORCE (simple forcing)
    // ========================================
    let force = params.gravity;
    u += force * 0.5;

    // ========================================
    // COLLISION: BGK relaxation
    // ========================================
    // Relaxation time: tau = 0.5 + 3 * viscosity
    // Higher viscosity = higher tau = slower relaxation = more stable
    let tau = 0.5 + 3.0 * params.viscosity;
    let omega = 1.0 / tau;

    var f_out: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let f_eq_i = f_eq(i, rho, u);

        // BGK collision: f_out = f - omega * (f - f_eq)
        f_out[i] = f[i] - omega * (f[i] - f_eq_i);

        // Simple forcing term (Shan-Chen style)
        let e = get_e_f(i);
        let force_contrib = W[i] * INV_CS2 * dot(e, force);
        f_out[i] += force_contrib;

        // Ensure positivity for stability
        f_out[i] = max(f_out[i], 0.0);
    }

    // ========================================
    // STORE RESULTS
    // ========================================
    textureStore(dist_out_0, pos, vec4<f32>(f_out[0], f_out[1], f_out[2], f_out[3]));
    textureStore(dist_out_1, pos, vec4<f32>(f_out[4], f_out[5], f_out[6], f_out[7]));
    textureStore(dist_out_2, pos, vec4<f32>(f_out[8], f_out[9], f_out[10], f_out[11]));
    textureStore(dist_out_3, pos, vec4<f32>(f_out[12], f_out[13], f_out[14], f_out[15]));
    textureStore(dist_out_4, pos, vec4<f32>(f_out[16], f_out[17], f_out[18], 0.0));

    textureStore(density_out, pos, vec4<f32>(rho, 0.0, 0.0, 0.0));
    textureStore(velocity_out, pos, vec4<f32>(u, 0.0));
}
