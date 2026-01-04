//! Spira - Nasal Sinus Airflow Simulation using Lattice Boltzmann Method
//!
//! This application implements a D3Q19 LBM simulation with GPU compute shaders
//! and real-time volumetric visualization using raymarching.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{texture_storage_3d, uniform_buffer},
            encase::ShaderType,
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            Extent3d, PipelineCache, ShaderStages, StorageTextureAccess, TextureDimension,
            TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use noise::{NoiseFn, Perlin};

/// Grid size for the simulation (must be divisible by workgroup size 8)
const GRID_SIZE: u32 = 64;

/// Workgroup size for compute shaders
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Spira - Sinus Airflow Simulation".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(LbmPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (rotate_camera, handle_input))
        .run();
}

/// Plugin that handles all LBM simulation and visualization
struct LbmPlugin;

impl Plugin for LbmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LbmTextures>()
            .init_resource::<SimulationParams>()
            .add_plugins(ExtractResourcePlugin::<LbmTextures>::default())
            .add_plugins(ExtractResourcePlugin::<SimulationParams>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                Render,
                prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .init_resource::<LbmPipeline>();

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(LbmNodeLabel, LbmNode::default());
        render_graph.add_node_edge(LbmNodeLabel, bevy::render::graph::CameraDriverLabel);
    }
}

/// Uniform buffer for simulation parameters
#[derive(Resource, Clone, Copy, Pod, Zeroable, ExtractResource, ShaderType)]
#[repr(C)]
struct SimulationParams {
    /// Grid dimensions
    grid_size: [u32; 4],
    /// Relaxation parameter (omega = 1/tau)
    omega: f32,
    /// Current simulation step (for ping-pong)
    step: u32,
    /// Padding for alignment
    _padding: [u32; 2],
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            grid_size: [GRID_SIZE, GRID_SIZE, GRID_SIZE, 0],
            omega: 1.8, // Relaxation parameter for BGK
            step: 0,
            _padding: [0; 2],
        }
    }
}

/// Container for all LBM textures
#[derive(Resource, Clone, ExtractResource)]
struct LbmTextures {
    /// Geometry texture (0 = air, 1 = wall/bone)
    geometry: Handle<Image>,
    /// Distribution functions buffer A (19 channels packed)
    f_in: Handle<Image>,
    /// Distribution functions buffer B (19 channels packed)
    f_out: Handle<Image>,
    /// Velocity field for visualization (xyz = velocity, w = density)
    velocity: Handle<Image>,
}

impl FromWorld for LbmTextures {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();

        let size = Extent3d {
            width: GRID_SIZE,
            height: GRID_SIZE,
            depth_or_array_layers: GRID_SIZE,
        };

        // Geometry texture - R8Uint (0=air, 1=wall)
        let geometry = create_3d_texture(&mut images, size, TextureFormat::R8Uint);

        // Distribution functions - we need 19 values per cell
        // Using 5 RGBA32Float textures = 20 channels (19 used + 1 padding)
        // For simplicity, we'll pack into a larger texture with GRID_SIZE * 5 depth
        let f_size = Extent3d {
            width: GRID_SIZE,
            height: GRID_SIZE,
            depth_or_array_layers: GRID_SIZE * 5, // 5 layers * 4 channels = 20 values
        };
        let f_in = create_3d_texture(&mut images, f_size, TextureFormat::Rgba32Float);
        let f_out = create_3d_texture(&mut images, f_size, TextureFormat::Rgba32Float);

        // Velocity field for visualization
        let velocity = create_3d_texture(&mut images, size, TextureFormat::Rgba32Float);

        // Generate phantom sinus geometry
        generate_sinus_phantom(&mut images, &geometry);

        Self {
            geometry,
            f_in,
            f_out,
            velocity,
        }
    }
}

fn create_3d_texture(
    images: &mut Assets<Image>,
    size: Extent3d,
    format: TextureFormat,
) -> Handle<Image> {
    let bytes_per_pixel = format.block_copy_size(None).unwrap_or(4) as usize;
    let data_size =
        size.width as usize * size.height as usize * size.depth_or_array_layers as usize * bytes_per_pixel;

    let mut image = Image::new_fill(
        size,
        TextureDimension::D3,
        &vec![0u8; bytes_per_pixel],
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(vec![0u8; data_size]);
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

    images.add(image)
}

/// Generates a procedural phantom sinus structure using Perlin noise
fn generate_sinus_phantom(images: &mut Assets<Image>, geometry_handle: &Handle<Image>) {
    let Some(image) = images.get_mut(geometry_handle) else {
        return;
    };
    let Some(ref mut data) = image.data else {
        return;
    };

    let perlin = Perlin::new(42);
    let size = GRID_SIZE as usize;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;

                // Normalized coordinates centered at 0.5
                let nx = x as f64 / size as f64 - 0.5;
                let ny = y as f64 / size as f64 - 0.5;
                let nz = z as f64 / size as f64 - 0.5;

                // Distance from center
                let dist = (nx * nx + ny * ny + nz * nz).sqrt();

                // Create a hollow ellipsoid with noise for realistic sinus shape
                let noise_val = perlin.get([nx * 4.0, ny * 4.0, nz * 4.0]) * 0.15;
                let inner_radius = 0.25 + noise_val;
                let outer_radius = 0.45 + noise_val * 0.5;

                // Wall if between inner and outer radius, or at boundaries
                let is_wall = dist < inner_radius
                    || dist > outer_radius
                    || x < 2
                    || x >= size - 2
                    || y < 2
                    || y >= size - 2;

                // Create inlet/outlet channels at z boundaries
                let channel_dist = ((nx * nx + ny * ny) as f64).sqrt();
                let is_channel = channel_dist < 0.1 && (z < 4 || z >= size - 4);

                data[idx] = if is_wall && !is_channel { 1 } else { 0 };
            }
        }
    }
}

/// Render graph label for LBM compute node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct LbmNodeLabel;

/// Pipeline resource containing compute shader pipelines
#[derive(Resource)]
struct LbmPipeline {
    bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    stream_collide_pipeline: CachedComputePipelineId,
}

impl FromWorld for LbmPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "lbm_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // Simulation params uniform
                    uniform_buffer::<SimulationParams>(false),
                    // Geometry texture
                    texture_storage_3d(TextureFormat::R8Uint, StorageTextureAccess::ReadOnly),
                    // f_in
                    texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                    // f_out
                    texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    // velocity output
                    texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let shader = asset_server.load("shaders/lbm.wgsl");

        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("lbm_init_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec!["INIT".into()],
            entry_point: Some("init".into()),
            zero_initialize_workgroup_memory: false,
        });

        let stream_collide_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("lbm_stream_collide_pipeline".into()),
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader,
                shader_defs: vec![],
                entry_point: Some("stream_collide".into()),
                zero_initialize_workgroup_memory: false,
            });

        Self {
            bind_group_layout,
            init_pipeline,
            stream_collide_pipeline,
        }
    }
}

/// Bind groups for LBM compute passes
#[derive(Resource, Default)]
struct LbmBindGroups {
    /// Bind group for even steps (f_in -> f_out)
    even: Option<BindGroup>,
    /// Bind group for odd steps (f_out -> f_in)
    odd: Option<BindGroup>,
}

fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<LbmPipeline>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    textures: Res<LbmTextures>,
    params: Res<SimulationParams>,
    render_device: Res<RenderDevice>,
) {
    let (Some(geometry), Some(f_in), Some(f_out), Some(velocity)) = (
        gpu_images.get(&textures.geometry),
        gpu_images.get(&textures.f_in),
        gpu_images.get(&textures.f_out),
        gpu_images.get(&textures.velocity),
    ) else {
        return;
    };

    // Create uniform buffer
    let params_buffer = render_device.create_buffer_with_data(&bevy::render::render_resource::BufferInitDescriptor {
        label: Some("lbm_params_buffer"),
        contents: bytemuck::bytes_of(&*params),
        usage: bevy::render::render_resource::BufferUsages::UNIFORM | bevy::render::render_resource::BufferUsages::COPY_DST,
    });

    // Even step: read from f_in, write to f_out
    let even = render_device.create_bind_group(
        "lbm_bind_group_even",
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            params_buffer.as_entire_binding(),
            &geometry.texture_view,
            &f_in.texture_view,
            &f_out.texture_view,
            &velocity.texture_view,
        )),
    );

    // Odd step: read from f_out, write to f_in
    let params_buffer_odd = render_device.create_buffer_with_data(&bevy::render::render_resource::BufferInitDescriptor {
        label: Some("lbm_params_buffer_odd"),
        contents: bytemuck::bytes_of(&*params),
        usage: bevy::render::render_resource::BufferUsages::UNIFORM | bevy::render::render_resource::BufferUsages::COPY_DST,
    });

    let odd = render_device.create_bind_group(
        "lbm_bind_group_odd",
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            params_buffer_odd.as_entire_binding(),
            &geometry.texture_view,
            &f_out.texture_view,
            &f_in.texture_view,
            &velocity.texture_view,
        )),
    );

    commands.insert_resource(LbmBindGroups {
        even: Some(even),
        odd: Some(odd),
    });
}

/// Compute node for LBM simulation
#[derive(Default)]
struct LbmNode {
    initialized: bool,
    step: u32,
}

impl render_graph::Node for LbmNode {
    fn update(&mut self, _world: &mut World) {}

    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline = world.resource::<LbmPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let bind_groups = world.resource::<LbmBindGroups>();

        let workgroups = GRID_SIZE / WORKGROUP_SIZE;

        // Initialization pass (run once)
        if !self.initialized {
            if let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.init_pipeline)
            {
                if let Some(bind_group) = &bind_groups.even {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.set_pipeline(init_pipeline);
                    pass.dispatch_workgroups(workgroups, workgroups, workgroups);
                }
            }
        }

        // Stream and collide pass
        if let Some(compute_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.stream_collide_pipeline)
        {
            // Select bind group based on step parity
            let bind_group = if self.step % 2 == 0 {
                &bind_groups.even
            } else {
                &bind_groups.odd
            };

            if let Some(bg) = bind_group {
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(compute_pipeline);
                pass.dispatch_workgroups(workgroups, workgroups, workgroups);
            }
        }

        Ok(())
    }
}

/// Sets up the 3D scene with camera and visualization cube
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    _textures: Res<LbmTextures>,
) {
    // Visualization cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.5, 0.8, 0.5),
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        VisualizationCube,
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera {
            radius: 5.0,
            theta: std::f32::consts::FRAC_PI_4,
            phi: std::f32::consts::FRAC_PI_4,
        },
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        affects_lightmapped_meshes: true,
    });

    println!("Spira simulation initialized. Grid size: {}³", GRID_SIZE);
    println!("Controls: Arrow keys to rotate camera, +/- to zoom");
}

/// Marker component for the visualization cube
#[derive(Component)]
struct VisualizationCube;

/// Orbit camera controller
#[derive(Component)]
struct OrbitCamera {
    radius: f32,
    theta: f32, // Horizontal angle
    phi: f32,   // Vertical angle
}

fn rotate_camera(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    for (mut transform, mut orbit) in query.iter_mut() {
        // Auto-rotate slowly
        orbit.theta += time.delta_secs() * 0.1;

        let x = orbit.radius * orbit.phi.sin() * orbit.theta.cos();
        let y = orbit.radius * orbit.phi.cos();
        let z = orbit.radius * orbit.phi.sin() * orbit.theta.sin();

        transform.translation = Vec3::new(x, y, z);
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut OrbitCamera>,
) {
    for mut orbit in query.iter_mut() {
        if keyboard.pressed(KeyCode::ArrowLeft) {
            orbit.theta -= 0.02;
        }
        if keyboard.pressed(KeyCode::ArrowRight) {
            orbit.theta += 0.02;
        }
        if keyboard.pressed(KeyCode::ArrowUp) {
            orbit.phi = (orbit.phi - 0.02).max(0.1);
        }
        if keyboard.pressed(KeyCode::ArrowDown) {
            orbit.phi = (orbit.phi + 0.02).min(std::f32::consts::PI - 0.1);
        }
        if keyboard.pressed(KeyCode::Equal) || keyboard.pressed(KeyCode::NumpadAdd) {
            orbit.radius = (orbit.radius - 0.1).max(2.0);
        }
        if keyboard.pressed(KeyCode::Minus) || keyboard.pressed(KeyCode::NumpadSubtract) {
            orbit.radius = (orbit.radius + 0.1).min(15.0);
        }
    }
}
