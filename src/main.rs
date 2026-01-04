//! Blazing LBM fluid simulation for dental sinus rinsing.
//!
//! This application simulates saline solution flow through a maxillary sinus cavity
//! using the Lattice Boltzmann Method (LBM) D3Q19 with Free Surface approximation.

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
const GRID_SIZE: u32 = 128;
/// Workgroup size for compute shaders (8x8x8 = 512 threads per workgroup)
const WORKGROUP_SIZE: u32 = 8;
/// Display window scale factor
const DISPLAY_FACTOR: u32 = 4;

/// Shader asset paths
const LBM_SHADER_PATH: &str = "shaders/lbm.wgsl";
const INJECT_SHADER_PATH: &str = "shaders/inject.wgsl";
const RENDER_SHADER_PATH: &str = "shaders/render.wgsl";

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
                    let noise = Self::simple_noise(pos * 0.1) * 5.0;

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
            viscosity: 0.02,
            gravity: Vec3::new(0.0, -0.003, 0.0), // Strong downward gravity
            // Injection at the top of the cavity
            injection_point: Vec3::new(center, center + 20.0, center),
            injection_velocity: Vec3::new(0.0, -0.1, 0.0),
            injection_rate: 0.5,
            dt: 1.0,
            frame: 0,
            grid_size,
        }
    }
}

/// Resource for render parameters (camera, lighting)
#[derive(Resource, Clone, Copy, ExtractResource, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct RenderParams {
    /// Camera position in world space
    pub camera_pos: Vec3,
    pub _pad0: f32,
    /// Camera target (look-at point)
    pub camera_target: Vec3,
    pub _pad1: f32,
    /// Light direction (normalized)
    pub light_dir: Vec3,
    pub _pad2: f32,
    /// Grid size for ray calculations
    pub grid_size: f32,
    /// Density threshold for isosurface
    pub density_threshold: f32,
    /// Water color tint
    pub water_color: Vec3,
    pub _pad3: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        let grid_size = GRID_SIZE as f32;
        let center = grid_size * 0.5;

        Self {
            camera_pos: Vec3::new(center * 2.5, center * 1.5, center * 2.5),
            _pad0: 0.0,
            camera_target: Vec3::new(center, center, center),
            _pad1: 0.0,
            light_dir: Vec3::new(0.5, 1.0, 0.3).normalize(),
            _pad2: 0.0,
            grid_size,
            density_threshold: 0.3,
            water_color: Vec3::new(0.2, 0.5, 0.9),
            _pad3: 0.0,
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
    /// Output render target for raymarching
    pub render_target: Handle<Image>,
}

/// Resource holding bind groups for the compute passes
#[derive(Resource)]
pub struct LbmBindGroups {
    /// Bind groups for LBM step (0 = A->B, 1 = B->A)
    pub lbm_step: [BindGroup; 2],
    /// Bind group for injection
    pub injection: BindGroup,
    /// Bind group for rendering
    pub render: BindGroup,
}

// ============================================================================
// Pipeline Resource
// ============================================================================

#[derive(Resource)]
pub struct FluidSimPipeline {
    pub bind_group_layout_lbm: BindGroupLayout,
    pub bind_group_layout_inject: BindGroupLayout,
    pub bind_group_layout_render: BindGroupLayout,
    pub init_pipeline: CachedComputePipelineId,
    pub collide_stream_pipeline: CachedComputePipelineId,
    pub injection_pipeline: CachedComputePipelineId,
    pub render_pipeline: CachedComputePipelineId,
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
            ExtractResourcePlugin::<RenderParams>::default(),
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

    // Render bind group layout
    let bind_group_layout_render = render_device.create_bind_group_layout(
        "Render_BindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Density texture (read)
                texture_storage_3d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                // Velocity texture (read)
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                // Boundaries texture (read)
                texture_storage_3d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                // Output image (write)
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                // Render params uniform
                uniform_buffer::<RenderParams>(false),
            ),
        ),
    );

    // Load shaders via asset server
    let lbm_shader = asset_server.load(LBM_SHADER_PATH);
    let inject_shader = asset_server.load(INJECT_SHADER_PATH);
    let render_shader = asset_server.load(RENDER_SHADER_PATH);

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

    let render_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from("Render_Pipeline")),
        layout: vec![bind_group_layout_render.clone()],
        push_constant_ranges: vec![],
        shader: render_shader,
        shader_defs: vec![],
        entry_point: Some(Cow::from("raymarch")),
        zero_initialize_workgroup_memory: true,
    });

    commands.insert_resource(FluidSimPipeline {
        bind_group_layout_lbm,
        bind_group_layout_inject,
        bind_group_layout_render,
        init_pipeline,
        collide_stream_pipeline,
        injection_pipeline,
        render_pipeline,
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
    render_params: Res<RenderParams>,
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
    let render_target = match gpu_images.get(&lbm_textures.render_target) {
        Some(img) => img,
        None => return,
    };

    // Create uniform buffers
    let mut fluid_uniform = UniformBuffer::from(*fluid_params);
    fluid_uniform.write_buffer(&render_device, &queue);

    let mut render_uniform = UniformBuffer::from(*render_params);
    render_uniform.write_buffer(&render_device, &queue);

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

    // Injection bind group (uses current output distributions)
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

    // Render bind group
    let render_bind_group = render_device.create_bind_group(
        Some("Render_BindGroup"),
        &pipeline.bind_group_layout_render,
        &BindGroupEntries::sequential((
            &density.texture_view,
            &velocity.texture_view,
            &boundaries.texture_view,
            &render_target.texture_view,
            &render_uniform,
        )),
    );

    commands.insert_resource(LbmBindGroups {
        lbm_step: [lbm_bind_group_0, lbm_bind_group_1],
        injection: injection_bind_group,
        render: render_bind_group,
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
                let render_ready = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.render_pipeline),
                    CachedPipelineState::Ok(_)
                );

                if init_ready && collide_ready && inject_ready && render_ready {
                    self.state = FluidSimState::Running(0);
                }
            }
            FluidSimState::Running(0) => {
                self.state = FluidSimState::Running(1);
            }
            FluidSimState::Running(1) => {
                self.state = FluidSimState::Running(0);
            }
            FluidSimState::Running(_) => unreachable!(),
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
                let render_pipeline =
                    match pipeline_cache.get_compute_pipeline(pipeline.render_pipeline) {
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
                    // Only dispatch a small region around injection point
                    pass.dispatch_workgroups(2, 2, 2);
                }

                // Raymarching render step
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("Raymarching"),
                            timestamp_writes: None,
                        });
                    pass.set_bind_group(0, &bind_groups.render, &[]);
                    pass.set_pipeline(render_pipeline);
                    // Render to window-sized output
                    let render_workgroups = (GRID_SIZE * DISPLAY_FACTOR) / WORKGROUP_SIZE;
                    pass.dispatch_workgroups(render_workgroups, render_workgroups, 1);
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Setup Systems
// ============================================================================

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
        RenderAssetUsages::RENDER_WORLD,
    );
    density_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
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
        RenderAssetUsages::RENDER_WORLD,
    );
    velocity_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
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

    // Render target (2D)
    let window_size = GRID_SIZE * DISPLAY_FACTOR;
    let mut render_target_image = Image::new_fill(
        Extent3d {
            width: window_size,
            height: window_size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    render_target_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let render_target = images.add(render_target_image);

    // Insert resources
    commands.insert_resource(voxel_world);
    commands.insert_resource(FluidParams::default());
    commands.insert_resource(RenderParams::default());
    commands.insert_resource(LbmTextures {
        distributions_a,
        distributions_b,
        density,
        velocity,
        boundaries,
        render_target: render_target.clone(),
    });

    // Spawn sprite to display the render target
    commands.spawn((
        Sprite {
            image: render_target,
            custom_size: Some(Vec2::splat(window_size as f32)),
            ..default()
        },
        Transform::default(),
    ));

    // Spawn camera
    commands.spawn(Camera2d);
}

/// Update fluid params each frame
fn update_simulation(mut params: ResMut<FluidParams>, time: Res<Time>) {
    params.frame = params.frame.wrapping_add(1);

    // Optionally animate injection point or other params
    let t = time.elapsed_secs();
    let wobble = (t * 2.0).sin() * 2.0;
    params.injection_point.x = (GRID_SIZE as f32 * 0.5) + wobble;
}

/// Camera control system
fn camera_control(
    mut render_params: ResMut<RenderParams>,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    let t = time.elapsed_secs();
    let center = GRID_SIZE as f32 * 0.5;
    let radius = center * 2.0;

    // Auto-rotate camera around the volume
    let angle = t * 0.2;
    render_params.camera_pos = Vec3::new(
        center + angle.cos() * radius,
        center * 1.2,
        center + angle.sin() * radius,
    );

    // Manual camera control
    let speed = 50.0 * time.delta_secs();
    if keyboard.pressed(KeyCode::KeyW) {
        render_params.camera_pos.z -= speed;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        render_params.camera_pos.z += speed;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        render_params.camera_pos.x -= speed;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        render_params.camera_pos.x += speed;
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        render_params.camera_pos.y -= speed;
    }
    if keyboard.pressed(KeyCode::KeyE) {
        render_params.camera_pos.y += speed;
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Spira - Dental Fluid Simulation".to_string(),
                    resolution: bevy::window::WindowResolution::new(
                        GRID_SIZE * DISPLAY_FACTOR,
                        GRID_SIZE * DISPLAY_FACTOR,
                    ),
                    ..default()
                }),
                ..default()
            }),
            SinusFluidPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (update_simulation, camera_control))
        .run();
}
