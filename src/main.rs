//! Spira - Position Based Fluids simulation demo
//!
//! This demo showcases the PBF fluid simulation with a dam break scenario.

use bevy::prelude::*;
use spira::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Spira - PBF Fluid Simulation".to_string(),
                resolution: bevy::window::WindowResolution::new(1280, 720),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(
            FluidPlugin::with_params(
                FluidParams::default()
                    .with_particle_spacing(0.08)
                    .with_solver_iterations(4)
                    .with_substeps(2)
                    .with_max_particles(50_000),
            )
            .cpu_mode(), // Use CPU simulation for now
        )
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (handle_input, update_debug_ui))
        .run();
}

/// Set up the 3D scene with fluid, boundaries, and camera.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-4.0, 5.0, 8.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    // Lighting
    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 10.0, 4.0),
    ));

    // Directional light for ambient-like lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));

    // Ground plane (visual only)
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(12.0, 12.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.35),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Boundary container (invisible walls)
    commands.spawn(BoxBoundaryBundle {
        boundary: BoxBoundary::new(
            Vec3::new(-3.0, 0.0, -3.0),
            Vec3::new(3.0, 8.0, 3.0),
        )
        .with_restitution(0.0)
        .with_friction(0.1),
        ..default()
    });

    // Fluid emitter - dam break scenario
    // Creates a block of fluid on one side that will collapse
    commands.spawn((
        FluidEmitter::box_emitter(Vec3::new(0.8, 1.5, 0.8), 0.08)
            .with_velocity(Vec3::ZERO)
            .with_color(Color::srgb(0.2, 0.5, 0.9)),
        Transform::from_xyz(-1.5, 1.6, 0.0),
    ));

    // Debug text
    commands.spawn((
        Text::new("Spira Fluid Simulation\n\nControls:\n  Space - Pause/Resume\n  R - Reset\n  S - Step (when paused)\n\nParticles: 0"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        DebugText,
    ));
}

/// Marker for debug text.
#[derive(Component)]
struct DebugText;

/// Handle keyboard input for simulation control.
fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<FluidState>,
    mut commands: Commands,
    particles: Query<Entity, With<FluidParticle>>,
    mut emitters: Query<&mut FluidEmitter>,
) {
    // Pause/Resume
    if keyboard.just_pressed(KeyCode::Space) {
        state.toggle_pause();
    }

    // Step
    if keyboard.just_pressed(KeyCode::KeyS) && state.paused {
        state.request_step();
    }

    // Reset
    if keyboard.just_pressed(KeyCode::KeyR) {
        // Remove all particles
        for entity in particles.iter() {
            commands.entity(entity).despawn();
        }
        state.particle_count = 0;
        state.frame = 0;
        state.time = 0.0;

        // Reset emitters
        for mut emitter in emitters.iter_mut() {
            emitter.has_spawned = false;
            emitter.particles_spawned = 0;
        }
    }
}

/// Update the debug UI text.
fn update_debug_ui(state: Res<FluidState>, mut text_query: Query<&mut Text, With<DebugText>>) {
    for mut text in text_query.iter_mut() {
        let status = if state.paused { "PAUSED" } else { "Running" };
        text.0 = format!(
            "Spira Fluid Simulation ({})\n\n\
             Controls:\n  \
             Space - Pause/Resume\n  \
             R - Reset\n  \
             S - Step (when paused)\n\n\
             Particles: {}\n\
             Frame: {}\n\
             Density Error: {:.2}%",
            status,
            state.particle_count,
            state.frame,
            state.avg_density_error * 100.0
        );
    }
}
