// Volumetric Raymarching Shader for LBM Visualization
// Renders velocity field from the simulation as a color-mapped volume

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct VolumeMaterial {
    // Camera position in world space
    camera_pos: vec3<f32>,
    // Volume bounds (min corner)
    volume_min: vec3<f32>,
    // Volume bounds (max corner)
    volume_max: vec3<f32>,
    // Grid size
    grid_size: u32,
    // Step size for raymarching
    step_size: f32,
    // Maximum velocity for normalization
    max_velocity: f32,
}

@group(2) @binding(0) var<uniform> material: VolumeMaterial;
@group(2) @binding(1) var velocity_texture: texture_3d<f32>;
@group(2) @binding(2) var velocity_sampler: sampler;
@group(2) @binding(3) var geometry_texture: texture_3d<u32>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let world_from_local = get_world_from_local(vertex.instance_index);
    let world_position = world_from_local * vec4<f32>(vertex.position, 1.0);

    out.world_position = world_position.xyz;
    out.world_normal = (world_from_local * vec4<f32>(vertex.normal, 0.0)).xyz;
    out.clip_position = mesh_position_local_to_clip(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );

    return out;
}

// Ray-box intersection
fn intersect_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;
    let t1 = (box_min - ray_origin) * inv_dir;
    let t2 = (box_max - ray_origin) * inv_dir;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    return vec2<f32>(t_enter, t_exit);
}

// Color map: cold (blue) to hot (red)
fn velocity_to_color(velocity_magnitude: f32) -> vec3<f32> {
    let t = clamp(velocity_magnitude, 0.0, 1.0);

    // Cool to warm colormap
    let color1 = vec3<f32>(0.0, 0.0, 0.5);   // Dark blue
    let color2 = vec3<f32>(0.0, 0.5, 1.0);   // Cyan
    let color3 = vec3<f32>(0.0, 1.0, 0.5);   // Green-cyan
    let color4 = vec3<f32>(1.0, 1.0, 0.0);   // Yellow
    let color5 = vec3<f32>(1.0, 0.0, 0.0);   // Red

    if (t < 0.25) {
        return mix(color1, color2, t * 4.0);
    } else if (t < 0.5) {
        return mix(color2, color3, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(color3, color4, (t - 0.5) * 4.0);
    } else {
        return mix(color4, color5, (t - 0.75) * 4.0);
    }
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Ray direction from camera to fragment
    let ray_origin = material.camera_pos;
    let ray_dir = normalize(in.world_position - ray_origin);

    // Find intersection with volume bounds
    let t_range = intersect_box(ray_origin, ray_dir, material.volume_min, material.volume_max);

    if (t_range.x > t_range.y) {
        // No intersection
        discard;
    }

    let t_start = max(t_range.x, 0.0);
    let t_end = t_range.y;

    // Raymarching
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    var t = t_start;

    let step_size = material.step_size;
    let volume_size = material.volume_max - material.volume_min;
    let grid_size_f = f32(material.grid_size);

    while (t < t_end && accumulated_alpha < 0.95) {
        let sample_pos = ray_origin + ray_dir * t;

        // Convert world position to texture coordinates [0, 1]
        let tex_coord = (sample_pos - material.volume_min) / volume_size;

        // Skip if outside bounds
        if (any(tex_coord < vec3<f32>(0.0)) || any(tex_coord > vec3<f32>(1.0))) {
            t += step_size;
            continue;
        }

        // Sample geometry (check if it's air)
        let grid_pos = vec3<i32>(tex_coord * grid_size_f);
        let is_wall = textureLoad(geometry_texture, grid_pos, 0).r > 0u;

        if (!is_wall) {
            // Sample velocity field
            let velocity_data = textureSampleLevel(velocity_texture, velocity_sampler, tex_coord, 0.0);
            let velocity = velocity_data.xyz;
            let velocity_mag = length(velocity) / material.max_velocity;

            if (velocity_mag > 0.001) {
                // Get color from velocity magnitude
                let sample_color = velocity_to_color(velocity_mag);

                // Opacity based on velocity magnitude
                let sample_alpha = clamp(velocity_mag * 2.0, 0.0, 0.5) * step_size * 10.0;

                // Front-to-back compositing
                let weight = sample_alpha * (1.0 - accumulated_alpha);
                accumulated_color += sample_color * weight;
                accumulated_alpha += weight;
            }
        }

        t += step_size;
    }

    if (accumulated_alpha < 0.01) {
        discard;
    }

    return vec4<f32>(accumulated_color, accumulated_alpha);
}
