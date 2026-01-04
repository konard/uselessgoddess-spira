//! Blazing LBM fluid simulation for dental sinus rinsing.
//!
//! This application simulates saline solution flow through a maxillary sinus cavity
//! using the Lattice Boltzmann Method (LBM) D3Q19.
//!
//! Controls:
//! - WASD: Move camera horizontally
//! - Q/E: Move camera up/down
//! - Arrow keys: Rotate camera
//! - R: Restart simulation
//! - Space: Toggle pause
//! - 1: Density coloring mode
//! - 2: Velocity/speed coloring mode
//! - 3: Pressure coloring mode
//! - +/-: Adjust injection rate

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::PipelineCacheError,
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

// ============================================================================
// Constants
// ============================================================================

/// Grid size for the MVP simulation (can be scaled up later for 512^3 sparse)
const GRID_SIZE: u32 = 64; // Reduced for better stability and performance
/// Workgroup size for compute shaders (8x8x8 = 512 threads per workgroup)
const WORKGROUP_SIZE: u32 = 8;
/// Voxel rendering size
const VOXEL_SIZE: f32 = 1.0;
/// Density threshold for rendering
const DENSITY_THRESHOLD: f32 = 0.5;

/// Shader asset paths
const LBM_SHADER_PATH: &str = "shaders/lbm.wgsl";
const INJECT_SHADER_PATH: &str = "shaders/inject.wgsl";

// ============================================================================
// Voxel World Trait (for future DICOM loader extensibility)
// ============================================================================

/// Trait for providing voxel world data.
/// This allows swapping the procedural generator with a DICOM loader later.
pub trait VoxelDataProvider: Send + Sync + 'static {
    /// Returns the grid dimensions (x, y, z)
    fn dimensions(&self) -> (u32, u32, u32);
    /// Returns true if the voxel at (x, y, z) is bone (solid boundary)
    fn is_bone(&self, x: u32, y: u32, z: u32) -> bool;
    /// Returns the signed distance field value at (x, y, z)
    /// Negative inside bone, positive inside air/cavity
    fn sdf(&self, x: u32, y: u32, z: u32) -> f32;
}

// ============================================================================
// Coloring Mode
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum ColoringMode {
    #[default]
    Density,
    Velocity,
    Pressure,
}

// ============================================================================
// Resources
// ============================================================================

/// Resource holding the voxel world data (collision boundaries).
/// Uses a procedural hollow sphere with noise to mimic a sinus cavity.
#[derive(Resource, Clone)]
pub struct VoxelWorld {
    pub dimensions: UVec3,
    /// SDF data: negative = bone, positive = air cavity
    /// Stored in a flat array, indexed as [z * dim.y * dim.x + y * dim.x + x]
    pub sdf_data: Vec<f32>,
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self::procedural_sinus(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    }
}

impl VoxelWorld {
    /// Generate a procedural sinus cavity (hollow sphere with noise)
    pub fn procedural_sinus(dim_x: u32, dim_y: u32, dim_z: u32) -> Self {
        let dimensions = UVec3::new(dim_x, dim_y, dim_z);
        let center = Vec3::new(dim_x as f32, dim_y as f32, dim_z as f32) * 0.5;
        let outer_radius = (dim_x.min(dim_y).min(dim_z) as f32) * 0.45;
        let inner_radius = outer_radius * 0.85;

        let total_voxels = (dim_x * dim_y * dim_z) as usize;
        let mut sdf_data = Vec::with_capacity(total_voxels);

        for z in 0..dim_z {
            for y in 0..dim_y {
                for x in 0..dim_x {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let dist_from_center = (pos - center).length();

                    // Add noise for organic shape variation
                    let noise = Self::simple_noise(pos * 0.1) * 3.0;

                    // SDF: positive inside cavity, negative in bone
                    let sdf = if dist_from_center < inner_radius + noise {
                        // Inside the cavity (air)
                        inner_radius + noise - dist_from_center
                    } else if dist_from_center < outer_radius + noise {
                        // Inside the bone shell
                        -(dist_from_center - inner_radius - noise)
                    } else {
                        // Outside (air, but we treat as boundary)
                        dist_from_center - outer_radius - noise
                    };

                    sdf_data.push(sdf);
                }
            }
        }

        Self {
            dimensions,
            sdf_data,
        }
    }

    /// Simple 3D noise function (deterministic pseudo-random)
    fn simple_noise(p: Vec3) -> f32 {
        let n = (p.x * 127.1 + p.y * 311.7 + p.z * 74.7).sin() * 43758.5453;
        n.fract() * 2.0 - 1.0
    }

    /// Get SDF value at position
    pub fn get_sdf(&self, x: u32, y: u32, z: u32) -> f32 {
        if x >= self.dimensions.x || y >= self.dimensions.y || z >= self.dimensions.z {
            return -1.0; // Outside bounds = solid
        }
        let idx =
            (z * self.dimensions.y * self.dimensions.x + y * self.dimensions.x + x) as usize;
        self.sdf_data.get(idx).copied().unwrap_or(-1.0)
    }
}

impl VoxelDataProvider for VoxelWorld {
    fn dimensions(&self) -> (u32, u32, u32) {
        (self.dimensions.x, self.dimensions.y, self.dimensions.z)
    }

    fn is_bone(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_sdf(x, y, z) < 0.0
    }

    fn sdf(&self, x: u32, y: u32, z: u32) -> f32 {
        self.get_sdf(x, y, z)
    }
}

/// Fluid simulation parameters
#[derive(Resource, Clone, Copy, ExtractResource, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct FluidParams {
    /// Kinematic viscosity (affects relaxation time)
    pub viscosity: f32,
    /// Gravity vector (strong for liquid simulation)
    pub gravity: Vec3,
    /// Injection point in grid coordinates
    pub injection_point: Vec3,
    /// Injection velocity/direction
    pub injection_velocity: Vec3,
    /// Injection rate (density per timestep)
    pub injection_rate: f32,
    /// Simulation time step (fixed for stability)
    pub dt: f32,
    /// Current simulation frame (for ping-pong)
    pub frame: u32,
    /// Grid dimensions
    pub grid_size: u32,
}

impl Default for FluidParams {
    fn default() -> Self {
        let grid_size = GRID_SIZE;
        let center = grid_size as f32 * 0.5;

        Self {
            viscosity: 0.1, // Higher viscosity for stability
            gravity: Vec3::new(0.0, -0.001, 0.0), // Gentler gravity
            // Injection at the top of the cavity
            injection_point: Vec3::new(center, center + 10.0, center),
            injection_velocity: Vec3::new(0.0, -0.05, 0.0),
            injection_rate: 0.1, // Lower injection rate for stability
            dt: 1.0,
            frame: 0,
            grid_size,
        }
    }
}

/// Simulation state resource
#[derive(Resource, Default)]
pub struct SimulationState {
    pub paused: bool,
    pub needs_restart: bool,
    pub coloring_mode: ColoringMode,
}

/// Camera controller state
#[derive(Resource)]
pub struct CameraController {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for CameraController {
    fn default() -> Self {
        let grid_size = GRID_SIZE as f32;
        Self {
            yaw: 0.5,
            pitch: 0.3,
            distance: grid_size * 2.5,
            target: Vec3::splat(grid_size * 0.5),
        }
    }
}

// ============================================================================
// LBM Textures Resource
// ============================================================================

/// Holds the 3D textures for LBM simulation (ping-pong pattern)
/// Each texture stores the 19 distribution functions per voxel
#[derive(Resource, Clone, ExtractResource)]
pub struct LbmTextures {
    /// Distribution functions texture A (19 channels packed into RGBA textures)
    /// We use 5 RGBA32Float textures to store 19 floats (5*4 = 20, 1 unused)
    pub distributions_a: [Handle<Image>; 5],
    /// Distribution functions texture B (for ping-pong)
    pub distributions_b: [Handle<Image>; 5],
    /// Fluid density texture (single float per voxel)
    pub density: Handle<Image>,
    /// Fluid velocity texture (vec3 per voxel)
    pub velocity: Handle<Image>,
    /// Boundary/SDF texture (solid voxels)
    pub boundaries: Handle<Image>,
}

/// Resource holding bind groups for the compute passes
#[derive(Resource)]
pub struct LbmBindGroups {
    /// Bind groups for LBM step (0 = A->B, 1 = B->A)
    pub lbm_step: [BindGroup; 2],
    /// Bind group for injection
    pub injection: BindGroup,
}

// ============================================================================
// Fluid Mesh Components
// ============================================================================

/// Marker component for fluid voxel entities
#[derive(Component)]
pub struct FluidVoxel {
    pub grid_pos: UVec3,
}

/// Marker component for bone/boundary voxel entities
#[derive(Component)]
pub struct BoneVoxel;

/// Component to track fluid visualization data
#[derive(Resource, Default)]
pub struct FluidVisualization {
    /// Density values extracted from GPU (for CPU-side visualization)
    pub density_data: Vec<f32>,
    /// Velocity data extracted from GPU
    pub velocity_data: Vec<Vec4>,
    /// Needs GPU readback
    pub needs_update: bool,
    /// Frame counter for throttling updates
    pub update_counter: u32,
}

// ============================================================================
// Pipeline Resource
// ============================================================================

#[derive(Resource)]
pub struct FluidSimPipeline {
    pub bind_group_layout_lbm: BindGroupLayout,
    pub bind_group_layout_inject: BindGroupLayout,
    pub init_pipeline: CachedComputePipelineId,
    pub collide_stream_pipeline: CachedComputePipelineId,
    pub injection_pipeline: CachedComputePipelineId,
}

// ============================================================================
// Plugin
// ============================================================================

/// Main plugin for the sinus fluid simulation
pub struct SinusFluidPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct FluidSimLabel;

impl Plugin for SinusFluidPlugin {
    fn build(&self, app: &mut App) {
        // Add resource extraction
        app.add_plugins((
            ExtractResourcePlugin::<LbmTextures>::default(),
            ExtractResourcePlugin::<FluidParams>::default(),
        ));

        // Setup render app
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_fluid_pipeline)
            .add_systems(
                Render,
                prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            );

        // Add to render graph
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(FluidSimLabel, FluidSimNode::default());
        render_graph.add_node_edge(FluidSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}

// ============================================================================
// Pipeline Initialization
// ============================================================================

fn init_fluid_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // LBM step bind group layout (distributions in/out, density, velocity, boundaries, params)
    let bind_group_layout_lbm = render_device.create_bind_group_layout(
        "LBM_BindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Input distributions (5 textures x RGBA = 20 floats, use 19)
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                // Output distributions (5 textures)
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                // Density output
                texture_storage_3d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                // Velocity output
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                // Boundaries (read-only)
                texture_storage_3d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                // Params uniform
                uniform_buffer::<FluidParams>(false),
            ),
        ),
    );

    // Injection bind group layout
    let bind_group_layout_inject = render_device.create_bind_group_layout(
        "Injection_BindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Distributions to modify
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                // Params uniform
                uniform_buffer::<FluidParams>(false),
            ),
        ),
    );

    // Load shaders via asset server
    let lbm_shader = asset_server.load(LBM_SHADER_PATH);
    let inject_shader = asset_server.load(INJECT_SHADER_PATH);

    // Create pipelines
    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from("LBM_Init_Pipeline")),
        layout: vec![bind_group_layout_lbm.clone()],
        push_constant_ranges: vec![],
        shader: lbm_shader.clone(),
        shader_defs: vec![],
        entry_point: Some(Cow::from("init")),
        zero_initialize_workgroup_memory: true,
    });

    let collide_stream_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from("LBM_CollideStream_Pipeline")),
        layout: vec![bind_group_layout_lbm.clone()],
        push_constant_ranges: vec![],
        shader: lbm_shader.clone(),
        shader_defs: vec![],
        entry_point: Some(Cow::from("collide_stream")),
        zero_initialize_workgroup_memory: true,
    });

    let injection_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from("Injection_Pipeline")),
        layout: vec![bind_group_layout_inject.clone()],
        push_constant_ranges: vec![],
        shader: inject_shader,
        shader_defs: vec![],
        entry_point: Some(Cow::from("inject_fluid")),
        zero_initialize_workgroup_memory: true,
    });

    commands.insert_resource(FluidSimPipeline {
        bind_group_layout_lbm,
        bind_group_layout_inject,
        init_pipeline,
        collide_stream_pipeline,
        injection_pipeline,
    });
}

// ============================================================================
// Bind Group Preparation
// ============================================================================

fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<FluidSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    lbm_textures: Res<LbmTextures>,
    fluid_params: Res<FluidParams>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // Get GPU texture views
    let dist_a: Vec<_> = lbm_textures
        .distributions_a
        .iter()
        .filter_map(|h| gpu_images.get(h))
        .collect();
    let dist_b: Vec<_> = lbm_textures
        .distributions_b
        .iter()
        .filter_map(|h| gpu_images.get(h))
        .collect();

    if dist_a.len() != 5 || dist_b.len() != 5 {
        return; // Textures not ready yet
    }

    let density = match gpu_images.get(&lbm_textures.density) {
        Some(img) => img,
        None => return,
    };
    let velocity = match gpu_images.get(&lbm_textures.velocity) {
        Some(img) => img,
        None => return,
    };
    let boundaries = match gpu_images.get(&lbm_textures.boundaries) {
        Some(img) => img,
        None => return,
    };

    // Create uniform buffers
    let mut fluid_uniform = UniformBuffer::from(*fluid_params);
    fluid_uniform.write_buffer(&render_device, &queue);

    // LBM step bind groups (A->B and B->A)
    let lbm_bind_group_0 = render_device.create_bind_group(
        Some("LBM_BindGroup_0"),
        &pipeline.bind_group_layout_lbm,
        &BindGroupEntries::sequential((
            // Input A
            &dist_a[0].texture_view,
            &dist_a[1].texture_view,
            &dist_a[2].texture_view,
            &dist_a[3].texture_view,
            &dist_a[4].texture_view,
            // Output B
            &dist_b[0].texture_view,
            &dist_b[1].texture_view,
            &dist_b[2].texture_view,
            &dist_b[3].texture_view,
            &dist_b[4].texture_view,
            // Density, Velocity, Boundaries
            &density.texture_view,
            &velocity.texture_view,
            &boundaries.texture_view,
            &fluid_uniform,
        )),
    );

    let lbm_bind_group_1 = render_device.create_bind_group(
        Some("LBM_BindGroup_1"),
        &pipeline.bind_group_layout_lbm,
        &BindGroupEntries::sequential((
            // Input B
            &dist_b[0].texture_view,
            &dist_b[1].texture_view,
            &dist_b[2].texture_view,
            &dist_b[3].texture_view,
            &dist_b[4].texture_view,
            // Output A
            &dist_a[0].texture_view,
            &dist_a[1].texture_view,
            &dist_a[2].texture_view,
            &dist_a[3].texture_view,
            &dist_a[4].texture_view,
            // Density, Velocity, Boundaries
            &density.texture_view,
            &velocity.texture_view,
            &boundaries.texture_view,
            &fluid_uniform,
        )),
    );

    // Injection bind group (uses current output distributions based on frame)
    // We'll use dist_b for injection after A->B step on even frames
    let injection_bind_group = render_device.create_bind_group(
        Some("Injection_BindGroup"),
        &pipeline.bind_group_layout_inject,
        &BindGroupEntries::sequential((
            &dist_b[0].texture_view,
            &dist_b[1].texture_view,
            &dist_b[2].texture_view,
            &dist_b[3].texture_view,
            &dist_b[4].texture_view,
            &fluid_uniform,
        )),
    );

    commands.insert_resource(LbmBindGroups {
        lbm_step: [lbm_bind_group_0, lbm_bind_group_1],
        injection: injection_bind_group,
    });
}

// ============================================================================
// Render Graph Node
// ============================================================================

enum FluidSimState {
    Loading,
    Init,
    Running(usize),
}

pub struct FluidSimNode {
    state: FluidSimState,
}

impl Default for FluidSimNode {
    fn default() -> Self {
        Self {
            state: FluidSimState::Loading,
        }
    }
}

impl render_graph::Node for FluidSimNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = match world.get_resource::<FluidSimPipeline>() {
            Some(p) => p,
            None => return,
        };
        let pipeline_cache = world.resource::<PipelineCache>();

        match self.state {
            FluidSimState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = FluidSimState::Init;
                    }
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        error!("Failed to load LBM shader: {err}");
                    }
                    _ => {}
                }
            }
            FluidSimState::Init => {
                // Check if all pipelines are ready
                let init_ready = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline),
                    CachedPipelineState::Ok(_)
                );
                let collide_ready = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.collide_stream_pipeline),
                    CachedPipelineState::Ok(_)
                );
                let inject_ready = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.injection_pipeline),
                    CachedPipelineState::Ok(_)
                );

                if init_ready && collide_ready && inject_ready {
                    self.state = FluidSimState::Running(0);
                }
            }
            FluidSimState::Running(idx) => {
                // Alternate between 0 and 1 for ping-pong
                self.state = FluidSimState::Running(1 - idx);
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = match world.get_resource::<LbmBindGroups>() {
            Some(bg) => bg,
            None => return Ok(()),
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = match world.get_resource::<FluidSimPipeline>() {
            Some(p) => p,
            None => return Ok(()),
        };

        let workgroups = GRID_SIZE / WORKGROUP_SIZE;

        match self.state {
            FluidSimState::Loading => {}
            FluidSimState::Init => {
                // Run initialization
                let init_pipeline = match pipeline_cache.get_compute_pipeline(pipeline.init_pipeline)
                {
                    Some(p) => p,
                    None => return Ok(()),
                };

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("LBM_Init"),
                        timestamp_writes: None,
                    });

                pass.set_bind_group(0, &bind_groups.lbm_step[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(workgroups, workgroups, workgroups);
            }
            FluidSimState::Running(index) => {
                let collide_pipeline = match pipeline_cache
                    .get_compute_pipeline(pipeline.collide_stream_pipeline)
                {
                    Some(p) => p,
                    None => return Ok(()),
                };
                let inject_pipeline =
                    match pipeline_cache.get_compute_pipeline(pipeline.injection_pipeline) {
                        Some(p) => p,
                        None => return Ok(()),
                    };

                // LBM Collide + Stream step
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("LBM_CollideStream"),
                            timestamp_writes: None,
                        });
                    pass.set_bind_group(0, &bind_groups.lbm_step[index], &[]);
                    pass.set_pipeline(collide_pipeline);
                    pass.dispatch_workgroups(workgroups, workgroups, workgroups);
                }

                // Injection step
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("LBM_Injection"),
                            timestamp_writes: None,
                        });
                    pass.set_bind_group(0, &bind_groups.injection, &[]);
                    pass.set_pipeline(inject_pipeline);
                    // Dispatch a small region around injection point
                    pass.dispatch_workgroups(2, 2, 2);
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Setup Systems
// ============================================================================

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Create the voxel world
    let voxel_world = VoxelWorld::default();

    // Create 3D textures for LBM distributions (ping-pong)
    let create_distribution_texture = || {
        let mut image = Image::new_fill(
            Extent3d {
                width: GRID_SIZE,
                height: GRID_SIZE,
                depth_or_array_layers: GRID_SIZE,
            },
            TextureDimension::D3,
            &[0u8; 16], // RGBA32Float = 16 bytes
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_descriptor.usage =
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        image
    };

    let distributions_a: [Handle<Image>; 5] = std::array::from_fn(|_| {
        images.add(create_distribution_texture())
    });
    let distributions_b: [Handle<Image>; 5] = std::array::from_fn(|_| {
        images.add(create_distribution_texture())
    });

    // Density texture
    let mut density_image = Image::new_fill(
        Extent3d {
            width: GRID_SIZE,
            height: GRID_SIZE,
            depth_or_array_layers: GRID_SIZE,
        },
        TextureDimension::D3,
        &[0u8; 4], // R32Float = 4 bytes
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    density_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
    let density = images.add(density_image);

    // Velocity texture
    let mut velocity_image = Image::new_fill(
        Extent3d {
            width: GRID_SIZE,
            height: GRID_SIZE,
            depth_or_array_layers: GRID_SIZE,
        },
        TextureDimension::D3,
        &[0u8; 16], // RGBA32Float for vec3 + padding
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    velocity_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
    let velocity = images.add(velocity_image);

    // Boundaries texture (from voxel world SDF)
    let boundary_data: Vec<u8> = voxel_world
        .sdf_data
        .iter()
        .flat_map(|&sdf| sdf.to_le_bytes())
        .collect();

    let mut boundaries_image = Image::new(
        Extent3d {
            width: GRID_SIZE,
            height: GRID_SIZE,
            depth_or_array_layers: GRID_SIZE,
        },
        TextureDimension::D3,
        boundary_data,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    boundaries_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let boundaries = images.add(boundaries_image);

    // Insert LBM textures resource
    commands.insert_resource(LbmTextures {
        distributions_a,
        distributions_b,
        density: density.clone(),
        velocity: velocity.clone(),
        boundaries,
    });

    // Insert other resources
    commands.insert_resource(voxel_world.clone());
    commands.insert_resource(FluidParams::default());
    commands.insert_resource(SimulationState::default());
    commands.insert_resource(CameraController::default());
    commands.insert_resource(FluidVisualization::default());

    // Create mesh for fluid voxels (small cube)
    let cube_mesh = meshes.add(Cuboid::new(VOXEL_SIZE * 0.8, VOXEL_SIZE * 0.8, VOXEL_SIZE * 0.8));

    // Create materials for fluid visualization
    let water_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.5, 0.9, 0.7),
        alpha_mode: AlphaMode::Blend,
        metallic: 0.0,
        reflectance: 0.5,
        perceptual_roughness: 0.1,
        ..default()
    });

    // Create bone material (semi-transparent)
    let bone_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.9, 0.85, 0.75, 0.3),
        alpha_mode: AlphaMode::Blend,
        metallic: 0.0,
        reflectance: 0.2,
        perceptual_roughness: 0.8,
        ..default()
    });

    // Store materials as resources
    commands.insert_resource(FluidMaterials {
        water: water_material.clone(),
        bone: bone_material.clone(),
        cube_mesh: cube_mesh.clone(),
    });

    // Spawn boundary visualization (sparse - only surface voxels)
    let grid_size = GRID_SIZE as i32;
    let center = grid_size as f32 * 0.5;

    // Sample boundary for visualization (every N voxels for performance)
    let sample_step = 4;
    for z in (0..grid_size).step_by(sample_step) {
        for y in (0..grid_size).step_by(sample_step) {
            for x in (0..grid_size).step_by(sample_step) {
                let sdf = voxel_world.get_sdf(x as u32, y as u32, z as u32);
                // Only render near-surface bone voxels
                if sdf < 0.0 && sdf > -5.0 {
                    commands.spawn((
                        Mesh3d(cube_mesh.clone()),
                        MeshMaterial3d(bone_material.clone()),
                        Transform::from_translation(Vec3::new(
                            x as f32 - center + 0.5,
                            y as f32 - center + 0.5,
                            z as f32 - center + 0.5,
                        )).with_scale(Vec3::splat(sample_step as f32)),
                        BoneVoxel,
                    ));
                }
            }
        }
    }

    // Spawn 3D camera
    let camera_controller = CameraController::default();
    let camera_pos = calculate_camera_position(&camera_controller);

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(camera_pos).looking_at(camera_controller.target - Vec3::splat(center), Vec3::Y),
    ));

    // Spawn directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));

    // Spawn ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 500.0,
        affects_lightmapped_meshes: false,
    });

    // Spawn UI text for controls
    commands.spawn((
        Text::new("Controls:\nWASD - Move camera\nQ/E - Up/Down\nArrows - Rotate\nR - Restart\nSpace - Pause\n1/2/3 - Color mode\n+/- Injection rate"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

/// Materials for fluid rendering
#[derive(Resource)]
pub struct FluidMaterials {
    pub water: Handle<StandardMaterial>,
    pub bone: Handle<StandardMaterial>,
    pub cube_mesh: Handle<Mesh>,
}

fn calculate_camera_position(controller: &CameraController) -> Vec3 {
    let x = controller.distance * controller.yaw.cos() * controller.pitch.cos();
    let y = controller.distance * controller.pitch.sin();
    let z = controller.distance * controller.yaw.sin() * controller.pitch.cos();
    controller.target + Vec3::new(x, y, z)
}

/// Update fluid params each frame
fn update_simulation(
    mut params: ResMut<FluidParams>,
    sim_state: Res<SimulationState>,
    time: Res<Time>,
) {
    if sim_state.paused {
        return;
    }

    params.frame = params.frame.wrapping_add(1);

    // Gentle wobble on injection point for natural flow
    let t = time.elapsed_secs();
    let wobble = (t * 1.5).sin() * 1.0;
    let center = GRID_SIZE as f32 * 0.5;
    params.injection_point.x = center + wobble;
}

/// Camera control system
fn camera_control(
    mut camera_controller: ResMut<CameraController>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    let dt = time.delta_secs();
    let move_speed = 50.0 * dt;
    let rotate_speed = 1.5 * dt;
    let center = GRID_SIZE as f32 * 0.5;

    // Rotation
    if keyboard.pressed(KeyCode::ArrowLeft) {
        camera_controller.yaw += rotate_speed;
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        camera_controller.yaw -= rotate_speed;
    }
    if keyboard.pressed(KeyCode::ArrowUp) {
        camera_controller.pitch = (camera_controller.pitch + rotate_speed).min(1.4);
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        camera_controller.pitch = (camera_controller.pitch - rotate_speed).max(-1.4);
    }

    // Movement (relative to camera orientation)
    let forward = Vec3::new(
        -camera_controller.yaw.sin(),
        0.0,
        -camera_controller.yaw.cos(),
    ).normalize();
    let right = Vec3::new(forward.z, 0.0, -forward.x);

    if keyboard.pressed(KeyCode::KeyW) {
        camera_controller.target += forward * move_speed;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        camera_controller.target -= forward * move_speed;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        camera_controller.target -= right * move_speed;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        camera_controller.target += right * move_speed;
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        camera_controller.target.y -= move_speed;
    }
    if keyboard.pressed(KeyCode::KeyE) {
        camera_controller.target.y += move_speed;
    }

    // Zoom
    if keyboard.pressed(KeyCode::Minus) || keyboard.pressed(KeyCode::NumpadSubtract) {
        camera_controller.distance *= 1.0 + dt;
    }
    if keyboard.pressed(KeyCode::Equal) || keyboard.pressed(KeyCode::NumpadAdd) {
        camera_controller.distance *= 1.0 - dt;
    }
    camera_controller.distance = camera_controller.distance.clamp(20.0, 500.0);

    // Update camera transform
    let camera_pos = calculate_camera_position(&camera_controller);
    for mut transform in camera_query.iter_mut() {
        *transform = Transform::from_translation(camera_pos)
            .looking_at(camera_controller.target - Vec3::splat(center), Vec3::Y);
    }
}

/// Handle keyboard input for simulation controls
fn handle_input(
    mut sim_state: ResMut<SimulationState>,
    mut params: ResMut<FluidParams>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    // Pause/unpause
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.paused = !sim_state.paused;
    }

    // Restart simulation
    if keyboard.just_pressed(KeyCode::KeyR) {
        sim_state.needs_restart = true;
        params.frame = 0;
    }

    // Coloring modes
    if keyboard.just_pressed(KeyCode::Digit1) {
        sim_state.coloring_mode = ColoringMode::Density;
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        sim_state.coloring_mode = ColoringMode::Velocity;
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        sim_state.coloring_mode = ColoringMode::Pressure;
    }

    // Adjust injection rate
    if keyboard.pressed(KeyCode::BracketRight) {
        params.injection_rate = (params.injection_rate + 0.01).min(1.0);
    }
    if keyboard.pressed(KeyCode::BracketLeft) {
        params.injection_rate = (params.injection_rate - 0.01).max(0.01);
    }
}

/// Update fluid visualization by reading density texture
fn update_fluid_visualization(
    mut commands: Commands,
    mut fluid_vis: ResMut<FluidVisualization>,
    images: Res<Assets<Image>>,
    lbm_textures: Res<LbmTextures>,
    sim_state: Res<SimulationState>,
    voxel_world: Res<VoxelWorld>,
    materials: Res<FluidMaterials>,
    mut mat_assets: ResMut<Assets<StandardMaterial>>,
    existing_voxels: Query<Entity, With<FluidVoxel>>,
) {
    // Throttle updates for performance (every 10 frames)
    fluid_vis.update_counter += 1;
    if fluid_vis.update_counter % 10 != 0 {
        return;
    }

    // Read density data from the image asset
    let density_image = match images.get(&lbm_textures.density) {
        Some(img) => img,
        None => return,
    };

    let velocity_image = match images.get(&lbm_textures.velocity) {
        Some(img) => img,
        None => return,
    };

    // Clear existing fluid voxels
    for entity in existing_voxels.iter() {
        commands.entity(entity).despawn();
    }

    let grid_size = GRID_SIZE as usize;
    let center = GRID_SIZE as f32 * 0.5;

    // Parse density and velocity data - in Bevy 0.17, data is Option<Vec<u8>>
    let density_bytes = match &density_image.data {
        Some(data) => data,
        None => return,
    };
    let velocity_bytes = match &velocity_image.data {
        Some(data) => data,
        None => return,
    };

    // Sample grid for visualization (every 2 voxels for performance)
    let sample_step = 2;

    for z in (0..grid_size).step_by(sample_step) {
        for y in (0..grid_size).step_by(sample_step) {
            for x in (0..grid_size).step_by(sample_step) {
                let idx = z * grid_size * grid_size + y * grid_size + x;

                // Check if this is inside the cavity (not bone)
                let sdf = voxel_world.get_sdf(x as u32, y as u32, z as u32);
                if sdf <= 0.0 {
                    continue; // Skip bone voxels
                }

                // Read density (R32Float = 4 bytes per voxel)
                let density_offset = idx * 4;
                if density_offset + 4 > density_bytes.len() {
                    continue;
                }

                let density = f32::from_le_bytes([
                    density_bytes[density_offset],
                    density_bytes[density_offset + 1],
                    density_bytes[density_offset + 2],
                    density_bytes[density_offset + 3],
                ]);

                // Only render voxels with significant density
                if density < DENSITY_THRESHOLD {
                    continue;
                }

                // Read velocity (RGBA32Float = 16 bytes per voxel)
                let velocity_offset = idx * 16;
                let velocity = if velocity_offset + 16 <= velocity_bytes.len() {
                    Vec3::new(
                        f32::from_le_bytes([
                            velocity_bytes[velocity_offset],
                            velocity_bytes[velocity_offset + 1],
                            velocity_bytes[velocity_offset + 2],
                            velocity_bytes[velocity_offset + 3],
                        ]),
                        f32::from_le_bytes([
                            velocity_bytes[velocity_offset + 4],
                            velocity_bytes[velocity_offset + 5],
                            velocity_bytes[velocity_offset + 6],
                            velocity_bytes[velocity_offset + 7],
                        ]),
                        f32::from_le_bytes([
                            velocity_bytes[velocity_offset + 8],
                            velocity_bytes[velocity_offset + 9],
                            velocity_bytes[velocity_offset + 10],
                            velocity_bytes[velocity_offset + 11],
                        ]),
                    )
                } else {
                    Vec3::ZERO
                };

                // Calculate color based on mode
                let color = match sim_state.coloring_mode {
                    ColoringMode::Density => {
                        // Blue to white based on density
                        let t = ((density - DENSITY_THRESHOLD) / (2.0 - DENSITY_THRESHOLD)).clamp(0.0, 1.0);
                        Color::srgba(
                            0.2 + t * 0.8,
                            0.5 + t * 0.5,
                            0.9,
                            0.6 + t * 0.3,
                        )
                    }
                    ColoringMode::Velocity => {
                        // Color based on velocity magnitude (cool to warm)
                        let speed = velocity.length();
                        let t = (speed * 20.0).clamp(0.0, 1.0);
                        Color::srgba(
                            t,
                            0.3 + (1.0 - t) * 0.4,
                            1.0 - t,
                            0.6 + t * 0.3,
                        )
                    }
                    ColoringMode::Pressure => {
                        // Pressure ~ density in LBM (cs^2 * rho)
                        let pressure = density / 3.0; // cs^2 = 1/3
                        let t = ((pressure - 0.3) / 0.4).clamp(0.0, 1.0);
                        Color::srgba(
                            0.2 + t * 0.7,
                            0.8 - t * 0.5,
                            0.3 + (1.0 - t) * 0.5,
                            0.6 + t * 0.3,
                        )
                    }
                };

                // Create material for this voxel
                let material = mat_assets.add(StandardMaterial {
                    base_color: color,
                    alpha_mode: AlphaMode::Blend,
                    metallic: 0.0,
                    reflectance: 0.5,
                    perceptual_roughness: 0.1,
                    ..default()
                });

                // Spawn fluid voxel
                commands.spawn((
                    Mesh3d(materials.cube_mesh.clone()),
                    MeshMaterial3d(material),
                    Transform::from_translation(Vec3::new(
                        x as f32 - center + 0.5,
                        y as f32 - center + 0.5,
                        z as f32 - center + 0.5,
                    )).with_scale(Vec3::splat(sample_step as f32 * 0.9)),
                    FluidVoxel {
                        grid_pos: UVec3::new(x as u32, y as u32, z as u32),
                    },
                ));
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Spira - Dental Fluid Simulation".to_string(),
                    resolution: bevy::window::WindowResolution::new(1024, 768),
                    ..default()
                }),
                ..default()
            }),
            SinusFluidPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            update_simulation,
            camera_control,
            handle_input,
            update_fluid_visualization,
        ))
        .run();
}
