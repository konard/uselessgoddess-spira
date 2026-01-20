//! Bevy plugin for fluid simulation.
//!
//! This module provides the main plugin for integrating PBF fluid simulation
//! into a Bevy application.

use bevy::prelude::*;

use super::{
    boundary::{BoxBoundary, FluidBoundary, PlaneBoundary, SphereBoundary},
    params::{FluidParams, FluidState},
    particle::{
        FluidEmitter, FluidParticle, FluidParticleBundle, ParticleDeltaPosition, ParticleDensity,
        ParticleLambda, ParticlePosition, ParticlePredictedPosition, ParticleVelocity,
    },
    render::{
        create_particle_material, create_particle_mesh, FluidParticleMaterial, FluidParticleMesh,
        FluidParticleVisual, FluidRenderConfig,
    },
    solver::CpuPbfSolver,
    spatial::SpatialHashConfig,
};

/// Main plugin for fluid simulation.
///
/// Add this plugin to your Bevy app to enable fluid simulation.
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use spira::fluid::prelude::*;
///
/// fn main() {
///     App::new()
///         .add_plugins(DefaultPlugins)
///         .add_plugins(FluidPlugin::default())
///         .run();
/// }
/// ```
#[derive(Default)]
pub struct FluidPlugin {
    /// Custom fluid parameters.
    pub params: Option<FluidParams>,
    /// Custom spatial hash configuration.
    pub spatial_config: Option<SpatialHashConfig>,
    /// Custom render configuration.
    pub render_config: Option<FluidRenderConfig>,
    /// Whether to use CPU simulation (for debugging).
    pub use_cpu_simulation: bool,
}

impl FluidPlugin {
    /// Create a plugin with custom parameters.
    pub fn with_params(params: FluidParams) -> Self {
        Self {
            params: Some(params),
            ..default()
        }
    }

    /// Use CPU-based simulation (slower but useful for debugging).
    pub fn cpu_mode(mut self) -> Self {
        self.use_cpu_simulation = true;
        self
    }
}

impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        // Register types for reflection
        app.register_type::<FluidParams>()
            .register_type::<FluidState>()
            .register_type::<FluidRenderConfig>()
            .register_type::<SpatialHashConfig>()
            .register_type::<FluidParticle>()
            .register_type::<ParticlePosition>()
            .register_type::<ParticleVelocity>()
            .register_type::<ParticlePredictedPosition>()
            .register_type::<ParticleDensity>()
            .register_type::<ParticleLambda>()
            .register_type::<ParticleDeltaPosition>()
            .register_type::<FluidBoundary>()
            .register_type::<BoxBoundary>()
            .register_type::<PlaneBoundary>()
            .register_type::<SphereBoundary>()
            .register_type::<FluidEmitter>()
            .register_type::<FluidParticleVisual>();

        // Insert resources
        let params = self.params.clone().unwrap_or_default();
        let spatial_config = self.spatial_config.clone().unwrap_or_else(|| {
            SpatialHashConfig::for_domain(
                Vec3::splat(-20.0),
                Vec3::splat(20.0),
                params.smoothing_radius,
            )
        });
        let render_config = self.render_config.clone().unwrap_or_default();

        app.insert_resource(params.clone())
            .insert_resource(FluidState::default())
            .insert_resource(spatial_config.clone())
            .insert_resource(render_config.clone());

        // Insert CPU solver resource if in CPU mode
        if self.use_cpu_simulation {
            app.insert_resource(CpuPbfSolverResource(CpuPbfSolver::new(
                params,
                spatial_config,
            )));
        }

        // Add systems
        app.add_systems(Startup, setup_fluid_rendering)
            .add_systems(
                Update,
                (
                    process_emitters,
                    sync_particles_to_solver.after(process_emitters),
                )
                    .chain(),
            );

        // Add simulation systems based on mode
        if self.use_cpu_simulation {
            app.add_systems(
                FixedUpdate,
                (
                    cpu_simulate_fluid,
                    apply_boundaries,
                    sync_solver_to_particles,
                )
                    .chain(),
            );
        }

        // Add rendering systems
        app.add_systems(
            PostUpdate,
            (update_particle_visuals, spawn_particle_visuals).chain(),
        );
    }
}

/// Resource wrapper for the CPU solver.
#[derive(Resource)]
pub struct CpuPbfSolverResource(pub CpuPbfSolver);

/// Setup system for fluid rendering resources.
fn setup_fluid_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    render_config: Res<FluidRenderConfig>,
    params: Res<FluidParams>,
) {
    // Create shared mesh and material
    let mesh = meshes.add(create_particle_mesh(1));
    let material = materials.add(create_particle_material(&render_config));

    commands.insert_resource(FluidParticleMesh(mesh));
    commands.insert_resource(FluidParticleMaterial(material));

    info!(
        "Fluid rendering initialized with particle radius: {:.3}m",
        params.particle_radius
    );
}

/// Process fluid emitters and spawn particles.
fn process_emitters(
    mut commands: Commands,
    mut emitters: Query<(&mut FluidEmitter, &Transform)>,
    params: Res<FluidParams>,
    mut state: ResMut<FluidState>,
    time: Res<Time>,
) {
    for (mut emitter, transform) in emitters.iter_mut() {
        if !emitter.enabled {
            continue;
        }

        // Check particle limit
        if emitter.max_particles > 0 && emitter.particles_spawned >= emitter.max_particles {
            continue;
        }

        // Check global particle limit
        if state.particle_count >= params.max_particles {
            continue;
        }

        let should_spawn = if emitter.one_shot {
            !emitter.has_spawned
        } else {
            emitter.accumulated_time += time.delta_secs();
            let particles_to_spawn =
                (emitter.accumulated_time * emitter.emission_rate) as u32;
            if particles_to_spawn > 0 {
                emitter.accumulated_time -= particles_to_spawn as f32 / emitter.emission_rate;
                true
            } else {
                false
            }
        };

        if should_spawn {
            let positions = emitter.generate_positions(transform);
            let count = positions.len() as u32;

            // Respect limits
            let available = params.max_particles - state.particle_count;
            let emitter_available = if emitter.max_particles > 0 {
                emitter.max_particles - emitter.particles_spawned
            } else {
                u32::MAX
            };
            let to_spawn = count.min(available).min(emitter_available);

            for pos in positions.into_iter().take(to_spawn as usize) {
                commands.spawn(
                    FluidParticleBundle::new(pos)
                        .with_velocity(emitter.initial_velocity)
                        .with_color(emitter.color),
                );
            }

            state.particle_count += to_spawn;
            emitter.particles_spawned += to_spawn;

            if emitter.one_shot {
                emitter.has_spawned = true;
            }

            info!(
                "Spawned {} fluid particles (total: {})",
                to_spawn, state.particle_count
            );
        }
    }
}

/// Sync ECS particles to the CPU solver.
fn sync_particles_to_solver(
    particles: Query<
        (&ParticlePosition, &ParticleVelocity),
        (With<FluidParticle>, Changed<ParticlePosition>),
    >,
    all_particles: Query<(&ParticlePosition, &ParticleVelocity), With<FluidParticle>>,
    mut solver: Option<ResMut<CpuPbfSolverResource>>,
) {
    let Some(solver) = solver.as_mut() else {
        return;
    };

    // Check if we need to rebuild (particles added/removed/changed)
    if !particles.is_empty() {
        let positions: Vec<Vec3> = all_particles.iter().map(|(p, _)| p.0).collect();
        let velocities: Vec<Vec3> = all_particles.iter().map(|(_, v)| v.0).collect();
        solver.0.set_particles(positions, velocities);
    }
}

/// CPU fluid simulation system.
fn cpu_simulate_fluid(
    mut solver: ResMut<CpuPbfSolverResource>,
    mut state: ResMut<FluidState>,
    time: Res<Time<Fixed>>,
) {
    if !state.should_simulate() {
        return;
    }

    let dt = time.delta_secs();
    solver.0.simulate(dt);

    state.time += dt;
    state.frame += 1;
    state.avg_density_error = solver.0.average_density_error();
}

/// Apply boundary conditions.
fn apply_boundaries(
    mut solver: Option<ResMut<CpuPbfSolverResource>>,
    box_boundaries: Query<&BoxBoundary, With<FluidBoundary>>,
    plane_boundaries: Query<&PlaneBoundary, With<FluidBoundary>>,
    _sphere_boundaries: Query<&SphereBoundary, With<FluidBoundary>>,
    params: Res<FluidParams>,
) {
    let Some(solver) = solver.as_mut() else {
        return;
    };

    // Apply box boundaries
    for boundary in box_boundaries.iter() {
        solver.0.apply_box_boundary(boundary.min, boundary.max, boundary.restitution);
    }

    // Apply plane boundaries (floor, walls)
    for boundary in plane_boundaries.iter() {
        let positions = solver.0.positions().to_vec();
        let velocities = solver.0.velocities().to_vec();

        for i in 0..positions.len() {
            let mut pos = positions[i];
            let mut vel = velocities[i];
            boundary.apply_collision(&mut pos, &mut vel, params.particle_radius);
            // Note: In a real implementation, we'd need mutable access to solver internals
        }
    }
}

/// Sync solver state back to ECS particles.
fn sync_solver_to_particles(
    solver: Option<Res<CpuPbfSolverResource>>,
    mut particles: Query<
        (&mut ParticlePosition, &mut ParticleVelocity),
        With<FluidParticle>,
    >,
) {
    let Some(solver) = solver.as_ref() else {
        return;
    };

    let positions = solver.0.positions();
    let velocities = solver.0.velocities();

    for (i, (mut pos, mut vel)) in particles.iter_mut().enumerate() {
        if i < positions.len() {
            pos.0 = positions[i];
            vel.0 = velocities[i];
        }
    }
}

/// Spawn visual representations for new particles.
fn spawn_particle_visuals(
    mut commands: Commands,
    particles: Query<
        (Entity, &ParticlePosition),
        (With<FluidParticle>, Without<FluidParticleVisual>),
    >,
    mesh: Option<Res<FluidParticleMesh>>,
    material: Option<Res<FluidParticleMaterial>>,
    params: Res<FluidParams>,
    render_config: Res<FluidRenderConfig>,
) {
    let (Some(mesh), Some(material)) = (mesh.as_ref(), material.as_ref()) else {
        return;
    };

    for (entity, position) in particles.iter() {
        let scale = params.particle_radius * render_config.size_scale;
        commands.entity(entity).insert((
            FluidParticleVisual,
            Mesh3d(mesh.0.clone()),
            MeshMaterial3d(material.0.clone()),
            Transform::from_translation(position.0).with_scale(Vec3::splat(scale)),
        ));
    }
}

/// Update visual transforms from particle positions.
fn update_particle_visuals(
    mut particles: Query<
        (&ParticlePosition, &mut Transform),
        (With<FluidParticle>, With<FluidParticleVisual>),
    >,
) {
    for (position, mut transform) in particles.iter_mut() {
        transform.translation = position.0;
    }
}

/// Debug system to display fluid simulation stats.
pub fn debug_fluid_stats(state: Res<FluidState>, _params: Res<FluidParams>) {
    info!(
        "Fluid Stats - Particles: {}, Frame: {}, Density Error: {:.2}%",
        state.particle_count,
        state.frame,
        state.avg_density_error * 100.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_builds() {
        // Just verify the plugin can be created
        let _plugin = FluidPlugin::default();
        let _plugin_cpu = FluidPlugin::default().cpu_mode();
        let _plugin_custom = FluidPlugin::with_params(FluidParams::viscous());
    }
}
