// Lattice Boltzmann Method D3Q19 Compute Shader
//
// Simplified LBM simulation step for dental fluid visualization.
// Uses ping-pong textures to alternate between read and write.

// Workgroup size for 3D compute dispatch
const WORKGROUP_SIZE: u32 = 4u;

// D3Q19 velocity set has 19 directions
const Q: u32 = 19u;

// D3Q19 velocity directions
const C: array<vec3<i32>, 19> = array<vec3<i32>, 19>(
    vec3<i32>( 0,  0,  0),  // 0: rest
    vec3<i32>( 1,  0,  0),  // 1: +x
    vec3<i32>(-1,  0,  0),  // 2: -x
    vec3<i32>( 0,  1,  0),  // 3: +y
    vec3<i32>( 0, -1,  0),  // 4: -y
    vec3<i32>( 0,  0,  1),  // 5: +z
    vec3<i32>( 0,  0, -1),  // 6: -z
    vec3<i32>( 1,  1,  0),  // 7
    vec3<i32>(-1, -1,  0),  // 8
    vec3<i32>( 1, -1,  0),  // 9
    vec3<i32>(-1,  1,  0),  // 10
    vec3<i32>( 1,  0,  1),  // 11
    vec3<i32>(-1,  0, -1),  // 12
    vec3<i32>( 1,  0, -1),  // 13
    vec3<i32>(-1,  0,  1),  // 14
    vec3<i32>( 0,  1,  1),  // 15
    vec3<i32>( 0, -1, -1),  // 16
    vec3<i32>( 0,  1, -1),  // 17
    vec3<i32>( 0, -1,  1),  // 18
);

// D3Q19 weights
const W: array<f32, 19> = array<f32, 19>(
    0.333333333,  // rest
    0.055555556, 0.055555556, 0.055555556,
    0.055555556, 0.055555556, 0.055555556,
    0.027777778, 0.027777778, 0.027777778, 0.027777778,
    0.027777778, 0.027777778, 0.027777778, 0.027777778,
    0.027777778, 0.027777778, 0.027777778, 0.027777778,
);

// Speed of sound squared (c_s^2 = 1/3 for D3Q19)
const CS2: f32 = 0.333333333;

// Relaxation time
const TAU: f32 = 0.6;

// Bind group: ping-pong textures
@group(0) @binding(0) var input_tex: texture_storage_3d<rgba32float, read>;
@group(0) @binding(1) var output_tex: texture_storage_3d<rgba32float, write>;

// Compute equilibrium distribution
fn equilibrium(q: u32, rho: f32, u: vec3<f32>) -> f32 {
    let c = vec3<f32>(C[q]);
    let cu = dot(c, u);
    let u2 = dot(u, u);
    return W[q] * rho * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2));
}

// Hash function for pseudo-random initialization
fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    return state;
}

fn random_float(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

// Main simulation step
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (id.x >= dims.x || id.y >= dims.y || id.z >= dims.z) {
        return;
    }

    // Read current state from input texture
    // RGBA: R=density, G=velocity.x, B=velocity.y, A=velocity.z
    let current = textureLoad(input_tex, id);
    var rho = current.r;
    var u = current.gba;

    // Get size for boundary and source calculations
    let size = f32(dims.x);
    let center = size / 2.0;
    let pos = vec3<f32>(id);

    // Initialize with some fluid if density is zero
    if (rho < 0.001) {
        // Create a spherical source near the top-center
        let source_pos = vec3<f32>(center, size * 0.8, center);
        let dist_to_source = length(pos - source_pos);

        if (dist_to_source < 5.0) {
            // Inject fluid at source
            rho = 1.0;
            u = vec3<f32>(0.0, -0.05, 0.0); // Downward velocity
        }
    }

    // Simple diffusion step
    // Average neighboring densities
    var neighbor_sum: f32 = 0.0;
    var neighbor_count: f32 = 0.0;

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dz: i32 = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }

                let neighbor_pos = vec3<i32>(id) + vec3<i32>(dx, dy, dz);

                // Bounds check
                if (neighbor_pos.x >= 0 && neighbor_pos.x < i32(dims.x) &&
                    neighbor_pos.y >= 0 && neighbor_pos.y < i32(dims.y) &&
                    neighbor_pos.z >= 0 && neighbor_pos.z < i32(dims.z)) {

                    let n = textureLoad(input_tex, vec3<u32>(neighbor_pos));
                    neighbor_sum += n.r;
                    neighbor_count += 1.0;
                }
            }
        }
    }

    // Diffusion with gravity
    if (neighbor_count > 0.0) {
        let avg = neighbor_sum / neighbor_count;
        rho = mix(rho, avg, 0.1);
    }

    // Apply gravity (fluid flows down)
    u.y -= 0.001;
    u = u * 0.99; // Damping

    // Advect density
    let advected_pos = pos - u * 10.0;
    if (advected_pos.x >= 0.0 && advected_pos.x < size &&
        advected_pos.y >= 0.0 && advected_pos.y < size &&
        advected_pos.z >= 0.0 && advected_pos.z < size) {
        // Simple advection
        let advected = textureLoad(input_tex, vec3<u32>(advected_pos));
        rho = mix(rho, advected.r, 0.3);
    }

    // Boundary conditions - bounce at walls
    let margin = 2.0;
    if (pos.x < margin || pos.x > size - margin ||
        pos.z < margin || pos.z > size - margin) {
        u = vec3<f32>(0.0);
        rho *= 0.5;
    }

    // Floor boundary
    if (pos.y < margin) {
        u.y = max(u.y, 0.0);
        rho = max(rho, 0.0);
    }

    // Ceiling - let fluid pass through (open boundary)
    if (pos.y > size - margin) {
        // Open boundary at top for injection
    }

    // Create visual pattern using density and velocity
    let speed = length(u);
    let output_val = vec4<f32>(
        clamp(rho, 0.0, 1.0),
        u.x + 0.5,
        u.y + 0.5,
        u.z + 0.5
    );

    textureStore(output_tex, id, output_val);
}
