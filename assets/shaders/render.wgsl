// Volumetric Raymarching Shader for Fluid Rendering
// Renders the fluid density field with water-like appearance
//
// Features:
// - Ray-box intersection for volume bounds
// - Density accumulation along rays
// - Gradient-based normal calculation
// - Fresnel reflections
// - Fake refraction for water look
// - Specular highlights (Blinn-Phong)

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 8u;
const MAX_STEPS: u32 = 256u;
const STEP_SIZE: f32 = 0.5;
const PI: f32 = 3.14159265359;

// ============================================================================
// Uniforms and Bindings
// ============================================================================

struct RenderParams {
    camera_pos: vec3<f32>,
    _pad0: f32,
    camera_target: vec3<f32>,
    _pad1: f32,
    light_dir: vec3<f32>,
    _pad2: f32,
    grid_size: f32,
    density_threshold: f32,
    water_color: vec3<f32>,
    _pad3: f32,
}

@group(0) @binding(0) var density_tex: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var velocity_tex: texture_storage_3d<rgba32float, read>;
@group(0) @binding(2) var boundaries_tex: texture_storage_3d<r32float, read>;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> params: RenderParams;

// ============================================================================
// Helper Functions
// ============================================================================

// Sample density at a position (with trilinear interpolation)
fn sample_density(pos: vec3<f32>) -> f32 {
    let size = params.grid_size;

    // Check bounds
    if any(pos < vec3<f32>(0.0)) || any(pos >= vec3<f32>(size)) {
        return 0.0;
    }

    let ipos = vec3<i32>(pos);
    return textureLoad(density_tex, ipos).r;
}

// Sample density with trilinear interpolation
fn sample_density_smooth(pos: vec3<f32>) -> f32 {
    let size = params.grid_size;

    // Clamp to valid range
    let clamped = clamp(pos, vec3<f32>(0.5), vec3<f32>(size - 1.5));

    let p0 = floor(clamped);
    let p1 = p0 + vec3<f32>(1.0);
    let t = clamped - p0;

    let i0 = vec3<i32>(p0);
    let i1 = vec3<i32>(p1);

    // Sample 8 corners of the voxel
    let d000 = textureLoad(density_tex, vec3<i32>(i0.x, i0.y, i0.z)).r;
    let d100 = textureLoad(density_tex, vec3<i32>(i1.x, i0.y, i0.z)).r;
    let d010 = textureLoad(density_tex, vec3<i32>(i0.x, i1.y, i0.z)).r;
    let d110 = textureLoad(density_tex, vec3<i32>(i1.x, i1.y, i0.z)).r;
    let d001 = textureLoad(density_tex, vec3<i32>(i0.x, i0.y, i1.z)).r;
    let d101 = textureLoad(density_tex, vec3<i32>(i1.x, i0.y, i1.z)).r;
    let d011 = textureLoad(density_tex, vec3<i32>(i0.x, i1.y, i1.z)).r;
    let d111 = textureLoad(density_tex, vec3<i32>(i1.x, i1.y, i1.z)).r;

    // Trilinear interpolation
    let d00 = mix(d000, d100, t.x);
    let d01 = mix(d001, d101, t.x);
    let d10 = mix(d010, d110, t.x);
    let d11 = mix(d011, d111, t.x);

    let d0 = mix(d00, d10, t.y);
    let d1 = mix(d01, d11, t.y);

    return mix(d0, d1, t.z);
}

// Sample boundary SDF
fn sample_boundary(pos: vec3<f32>) -> f32 {
    let size = params.grid_size;

    if any(pos < vec3<f32>(0.0)) || any(pos >= vec3<f32>(size)) {
        return -1.0; // Outside = solid
    }

    let ipos = vec3<i32>(pos);
    return textureLoad(boundaries_tex, ipos).r;
}

// Compute normal from density gradient
fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = 1.0;
    let dx = sample_density_smooth(pos + vec3<f32>(eps, 0.0, 0.0)) -
             sample_density_smooth(pos - vec3<f32>(eps, 0.0, 0.0));
    let dy = sample_density_smooth(pos + vec3<f32>(0.0, eps, 0.0)) -
             sample_density_smooth(pos - vec3<f32>(0.0, eps, 0.0));
    let dz = sample_density_smooth(pos + vec3<f32>(0.0, 0.0, eps)) -
             sample_density_smooth(pos - vec3<f32>(0.0, 0.0, eps));

    let gradient = vec3<f32>(dx, dy, dz);
    let len = length(gradient);

    if len < 0.0001 {
        return vec3<f32>(0.0, 1.0, 0.0); // Default up
    }

    return gradient / len;
}

// Ray-box intersection
fn ray_box_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;

    let t1 = (box_min - ray_origin) * inv_dir;
    let t2 = (box_max - ray_origin) * inv_dir;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    return vec2<f32>(max(t_enter, 0.0), t_exit);
}

// Fresnel approximation (Schlick)
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// ============================================================================
// Main Raymarching Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn raymarch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_size = textureDimensions(output_tex);
    let pixel = vec2<i32>(gid.xy);

    if pixel.x >= i32(output_size.x) || pixel.y >= i32(output_size.y) {
        return;
    }

    // Compute UV coordinates
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(output_size) * 2.0 - 1.0;

    // Setup camera
    let camera_pos = params.camera_pos;
    let camera_target = params.camera_target;
    let camera_up = vec3<f32>(0.0, 1.0, 0.0);

    // Camera basis
    let forward = normalize(camera_target - camera_pos);
    let right = normalize(cross(forward, camera_up));
    let up = cross(right, forward);

    // Field of view
    let fov = 1.0; // ~60 degrees
    let ray_dir = normalize(forward + uv.x * right * fov + uv.y * up * fov);

    // Volume bounds
    let box_min = vec3<f32>(0.0);
    let box_max = vec3<f32>(params.grid_size);

    // Ray-box intersection
    let t_range = ray_box_intersect(camera_pos, ray_dir, box_min, box_max);

    // Background color (dark blue gradient)
    let sky_color = mix(
        vec3<f32>(0.05, 0.05, 0.15),
        vec3<f32>(0.1, 0.15, 0.3),
        uv.y * 0.5 + 0.5
    );

    var final_color = sky_color;
    var alpha = 0.0;

    if t_range.y > t_range.x {
        // Ray hits the volume
        var t = t_range.x;
        var accumulated_density = 0.0;
        var hit_pos = vec3<f32>(0.0);
        var hit_found = false;

        // Raymarch through volume
        for (var i = 0u; i < MAX_STEPS; i++) {
            let pos = camera_pos + ray_dir * t;

            // Check if inside cavity (not bone)
            let sdf = sample_boundary(pos);

            if sdf > 0.0 {
                // Inside fluid region
                let density = sample_density_smooth(pos);

                if density > params.density_threshold {
                    // Found water surface
                    hit_pos = pos;
                    hit_found = true;
                    accumulated_density = density;
                    break;
                }

                // Accumulate for fog effect
                accumulated_density += density * STEP_SIZE * 0.1;
            } else if sdf < -2.0 {
                // Deep inside bone - render as bone color
                final_color = vec3<f32>(0.9, 0.85, 0.8); // Bone color
                alpha = 1.0;
                break;
            }

            t += STEP_SIZE;
            if t > t_range.y {
                break;
            }
        }

        if hit_found {
            // Compute water shading
            let normal = compute_normal(hit_pos);

            // View direction
            let view_dir = normalize(camera_pos - hit_pos);

            // Light direction
            let light_dir = normalize(params.light_dir);

            // Fresnel
            let cos_theta = max(dot(normal, view_dir), 0.0);
            let fresnel = fresnel_schlick(cos_theta, 0.02); // Water F0 ~ 0.02

            // Diffuse lighting
            let ndotl = max(dot(normal, light_dir), 0.0);
            let diffuse = ndotl * 0.6;

            // Specular (Blinn-Phong)
            let half_vec = normalize(light_dir + view_dir);
            let ndoth = max(dot(normal, half_vec), 0.0);
            let specular = pow(ndoth, 64.0) * 1.5;

            // Ambient
            let ambient = 0.2;

            // Fake subsurface scattering / refraction tint
            let thickness = min(accumulated_density * 0.5, 1.0);
            let subsurface_color = mix(params.water_color, vec3<f32>(0.1, 0.3, 0.5), thickness);

            // Combine
            let water_color = subsurface_color * (ambient + diffuse) + vec3<f32>(1.0) * specular;

            // Apply fresnel (more reflection at grazing angles)
            final_color = mix(water_color, sky_color * 0.5, fresnel);
            alpha = 0.9;

            // Depth fog
            let dist = length(hit_pos - camera_pos);
            let fog = 1.0 - exp(-dist * 0.005);
            final_color = mix(final_color, sky_color, fog);
        } else if accumulated_density > 0.01 {
            // Volume fog from scattered fluid
            let fog_color = params.water_color * 0.3;
            let fog_alpha = min(accumulated_density, 0.5);
            final_color = mix(sky_color, fog_color, fog_alpha);
            alpha = fog_alpha;
        }
    }

    // Output with gamma correction
    let gamma_corrected = pow(final_color, vec3<f32>(1.0 / 2.2));
    textureStore(output_tex, pixel, vec4<f32>(gamma_corrected, 1.0));
}
