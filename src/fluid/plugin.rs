//! Bevy plugin for fluid simulation.

use bevy::prelude::*;

use super::params::FluidParams;
use super::simulation::{FluidSimulation, ParticleStagingBuffer};

/// Plugin that adds Position Based Fluids simulation to a Bevy app.
///
/// # Example
///
/// ```rust,ignore
/// use bevy::prelude::*;
/// use spira::fluid::{FluidPlugin, FluidParams};
///
/// fn main() {
///     App::new()
///         .add_plugins(DefaultPlugins)
///         .add_plugins(FluidPlugin)
///         .run();
/// }
/// ```
pub struct FluidPlugin;

impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        // Register types for reflection
        app.register_type::<FluidParams>();

        // Initialize resources
        app.init_resource::<FluidParams>()
            .init_resource::<ParticleStagingBuffer>()
            .init_resource::<FluidSimulation>();

        // Add simulation systems
        app.add_systems(
            Update,
            (
                upload_staged_particles,
                run_simulation,
            )
                .chain(),
        );
    }
}

/// System to upload staged particles to the simulation.
fn upload_staged_particles(
    mut staging: ResMut<ParticleStagingBuffer>,
    mut simulation: ResMut<FluidSimulation>,
) {
    simulation.upload_from_staging(&mut staging);
}

/// System to run the fluid simulation.
fn run_simulation(
    time: Res<Time>,
    params: Res<FluidParams>,
    mut simulation: ResMut<FluidSimulation>,
) {
    let dt = params.fixed_timestep.unwrap_or(time.delta_secs());

    // Clamp dt to prevent instability
    let dt = dt.min(1.0 / 30.0);

    if dt > 0.0 {
        simulation.step(&params, dt);
    }
}
