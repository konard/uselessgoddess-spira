//! Physics module for LBM D3Q19 fluid simulation.
//!
//! This module implements a Lattice Boltzmann Method (LBM) solver using the D3Q19 velocity set
//! for simulating incompressible liquid flow (water/saline) in complex geometries.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::texture_storage_3d, *},
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::PipelineCacheError,
};
use std::borrow::Cow;

/// Shader paths for LBM compute kernels
const LBM_SHADER_PATH: &str = "shaders/lbm.wgsl";

/// Default simulation grid size (can be scaled down for performance)
pub const DEFAULT_GRID_SIZE: u32 = 64;

/// Workgroup size for compute dispatches
const WORKGROUP_SIZE: u32 = 4;

/// Plugin that manages the LBM fluid simulation compute pipeline
pub struct FluidCorePlugin {
    /// Grid resolution (uniform in all dimensions for this PoC)
    pub grid_size: u32,
}

impl Default for FluidCorePlugin {
    fn default() -> Self {
        Self {
            grid_size: DEFAULT_GRID_SIZE,
        }
    }
}

/// Label for the LBM compute node in the render graph
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct LbmComputeLabel;

impl Plugin for FluidCorePlugin {
    fn build(&self, app: &mut App) {
        // Insert grid configuration
        app.insert_resource(SimulationConfig {
            grid_size: self.grid_size,
        });

        // Add resource extraction plugins
        app.add_plugins((
            ExtractResourcePlugin::<SimulationConfig>::default(),
            ExtractResourcePlugin::<FluidTextures>::default(),
            ExtractResourcePlugin::<InjectionState>::default(),
        ));

        // Configure render app
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, init_lbm_pipeline)
            .add_systems(
                Render,
                prepare_lbm_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(LbmComputeLabel, LbmComputeNode::default());
        render_graph.add_node_edge(LbmComputeLabel, bevy::render::graph::CameraDriverLabel);
    }
}

/// Configuration for the fluid simulation
#[derive(Resource, Clone, ExtractResource)]
pub struct SimulationConfig {
    /// Grid resolution (cubic)
    pub grid_size: u32,
}

/// State of the fluid injection source
#[derive(Resource, Clone, ExtractResource)]
#[allow(dead_code)]
pub struct InjectionState {
    /// Origin point of injection in voxel coordinates
    pub origin: Vec3,
    /// Direction of injection (normalized)
    pub direction: Vec3,
    /// Injection pressure/velocity magnitude
    pub pressure: f32,
    /// Radius of injection source in voxels
    pub radius: f32,
    /// Whether injection is active
    pub active: bool,
}

impl Default for InjectionState {
    fn default() -> Self {
        Self {
            origin: Vec3::new(32.0, 50.0, 32.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            pressure: 0.1,
            radius: 3.0,
            active: false,
        }
    }
}

/// Textures for fluid density and velocity visualization (ping-pong buffers)
#[derive(Resource, Clone, ExtractResource)]
pub struct FluidTextures {
    /// Density field A
    pub density_a: Handle<Image>,
    /// Density field B
    pub density_b: Handle<Image>,
}

/// Bind groups for the LBM compute shader
#[derive(Resource)]
struct LbmBindGroups {
    /// Bind group for pass A -> B
    a_to_b: BindGroup,
    /// Bind group for pass B -> A
    b_to_a: BindGroup,
}

/// Pipeline resource for LBM compute shaders
#[derive(Resource)]
struct LbmPipeline {
    /// Bind group layout
    bind_group_layout: BindGroupLayout,
    /// Simulation step pipeline
    step_pipeline: CachedComputePipelineId,
}

fn init_lbm_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let bind_group_layout = render_device.create_bind_group_layout(
        Some("lbm_bind_group_layout"),
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Input density texture (read)
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                // Output density texture (write)
                texture_storage_3d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );

    let shader = asset_server.load(LBM_SHADER_PATH);

    let step_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(Cow::from("lbm_step")),
        layout: vec![bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("step")),
        ..default()
    });

    commands.insert_resource(LbmPipeline {
        bind_group_layout,
        step_pipeline,
    });
}

fn prepare_lbm_bind_groups(
    mut commands: Commands,
    pipeline: Option<Res<LbmPipeline>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fluid_textures: Option<Res<FluidTextures>>,
    render_device: Res<RenderDevice>,
) {
    let Some(pipeline) = pipeline else { return };
    let Some(fluid_textures) = fluid_textures else {
        return;
    };

    let Some(view_a) = gpu_images.get(&fluid_textures.density_a) else {
        return;
    };
    let Some(view_b) = gpu_images.get(&fluid_textures.density_b) else {
        return;
    };

    let a_to_b = render_device.create_bind_group(
        Some("lbm_a_to_b"),
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((&view_a.texture_view, &view_b.texture_view)),
    );

    let b_to_a = render_device.create_bind_group(
        Some("lbm_b_to_a"),
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((&view_b.texture_view, &view_a.texture_view)),
    );

    commands.insert_resource(LbmBindGroups { a_to_b, b_to_a });
}

/// State machine for the LBM compute node
#[derive(Default)]
enum LbmState {
    #[default]
    Loading,
    Running {
        buffer_index: usize,
    },
}

/// Render graph node that executes LBM compute passes
struct LbmComputeNode {
    state: LbmState,
}

impl Default for LbmComputeNode {
    fn default() -> Self {
        Self {
            state: LbmState::Loading,
        }
    }
}

impl render_graph::Node for LbmComputeNode {
    fn update(&mut self, world: &mut World) {
        let Some(pipeline) = world.get_resource::<LbmPipeline>() else {
            return;
        };
        let pipeline_cache = world.resource::<PipelineCache>();

        match &self.state {
            LbmState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.step_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = LbmState::Running { buffer_index: 0 };
                    }
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        bevy::log::error!("Failed to load LBM shader: {err}");
                    }
                    _ => {}
                }
            }
            LbmState::Running { buffer_index } => {
                self.state = LbmState::Running {
                    buffer_index: 1 - buffer_index,
                };
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let Some(bind_groups) = world.get_resource::<LbmBindGroups>() else {
            return Ok(());
        };
        let Some(pipeline) = world.get_resource::<LbmPipeline>() else {
            return Ok(());
        };
        let Some(config) = world.get_resource::<SimulationConfig>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();

        let workgroups = config.grid_size.div_ceil(WORKGROUP_SIZE);

        match &self.state {
            LbmState::Loading => {}
            LbmState::Running { buffer_index } => {
                let Some(step_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.step_pipeline)
                else {
                    return Ok(());
                };

                let bind_group = if *buffer_index == 0 {
                    &bind_groups.a_to_b
                } else {
                    &bind_groups.b_to_a
                };

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lbm_step"),
                        ..default()
                    });

                pass.set_pipeline(step_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups, workgroups, workgroups);
            }
        }

        Ok(())
    }
}

/// Creates a 3D texture for the fluid density field
pub fn create_fluid_texture(images: &mut Assets<Image>, grid_size: u32) -> Handle<Image> {
    let size = Extent3d {
        width: grid_size,
        height: grid_size,
        depth_or_array_layers: grid_size,
    };

    // Initialize with zeros (RGBA32Float = 16 bytes per pixel)
    let data = vec![0u8; (grid_size * grid_size * grid_size * 16) as usize];

    let mut image = Image::new(
        size,
        TextureDimension::D3,
        data,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

    images.add(image)
}
