//! Spira - GPU-accelerated Position Based Fluids simulation for Bevy
//!
//! This example demonstrates a basic fluid simulation using the PBF algorithm.
//! This is a proof-of-concept implementation showing the core architecture.

use bevy::prelude::*;

mod fluid;

use fluid::{FluidParams, FluidPlugin, ParticleStagingBuffer};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Spira - GPU Fluid Simulation".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FluidPlugin)
        .insert_resource(FluidParams::splashy())
        .add_systems(Startup, setup)
        .add_systems(Update, (camera_controls, particle_visualizer))
        .run();
}

/// Set up the demo scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut staging: ResMut<ParticleStagingBuffer>,
    params: Res<FluidParams>,
) {
    // Ground plane (container floor)
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(4.0, 4.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.35),
            perceptual_roughness: 0.8,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));

    // Container walls (visual only - collision is handled by simulation bounds)
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.6, 0.6, 0.65, 0.3),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.9,
        ..default()
    });

    // Back wall
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(4.0, 3.0))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(0.0, 1.0, -2.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
    ));

    // Left wall
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(4.0, 3.0))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(-2.0, 1.0, 0.0)
            .with_rotation(Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)),
    ));

    // Right wall
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(4.0, 3.0))),
        MeshMaterial3d(wall_material),
        Transform::from_xyz(2.0, 1.0, 0.0)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
    ));

    // Spawn initial block of particles
    fluid::spawn_particle_block(
        &mut staging,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.4, 0.3, 0.4),
        params.particle_radius * 2.0,
        Vec3::ZERO,
    );

    info!("Spawned {} initial particles", staging.positions.len());

    // Light
    commands.spawn((
        PointLight {
            intensity: 500000.0,
            shadows_enabled: true,
            range: 20.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 6.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
        CameraController::default(),
    ));

    info!("Spira Fluid Simulation Demo");
    info!("Controls:");
    info!("  WASD/QE - Move camera");
    info!("  Arrow keys - Rotate camera");
    info!("  R - Reset simulation");
}

/// Camera controller component
#[derive(Component)]
struct CameraController {
    speed: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            speed: 5.0,
            yaw: 0.0,
            pitch: -0.3,
        }
    }
}

/// Simple camera control system
fn camera_controls(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<(&mut Transform, &mut CameraController)>,
    mut staging: ResMut<ParticleStagingBuffer>,
    params: Res<FluidParams>,
) {
    for (mut transform, mut controller) in cameras.iter_mut() {
        // Arrow key rotation
        let rot_speed = 1.5 * time.delta_secs();
        if keyboard.pressed(KeyCode::ArrowLeft) {
            controller.yaw += rot_speed;
        }
        if keyboard.pressed(KeyCode::ArrowRight) {
            controller.yaw -= rot_speed;
        }
        if keyboard.pressed(KeyCode::ArrowUp) {
            controller.pitch = (controller.pitch + rot_speed).min(1.4);
        }
        if keyboard.pressed(KeyCode::ArrowDown) {
            controller.pitch = (controller.pitch - rot_speed).max(-1.4);
        }

        let rotation = Quat::from_euler(EulerRot::YXZ, controller.yaw, controller.pitch, 0.0);
        transform.rotation = rotation;

        // WASD movement
        let forward = transform.forward();
        let right = transform.right();
        let mut velocity = Vec3::ZERO;

        if keyboard.pressed(KeyCode::KeyW) {
            velocity += *forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            velocity -= *forward;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            velocity -= *right;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            velocity += *right;
        }
        if keyboard.pressed(KeyCode::KeyQ) {
            velocity -= Vec3::Y;
        }
        if keyboard.pressed(KeyCode::KeyE) {
            velocity += Vec3::Y;
        }

        if velocity.length_squared() > 0.0 {
            velocity = velocity.normalize() * controller.speed * time.delta_secs();
            transform.translation += velocity;
        }

        // Reset with R
        if keyboard.just_pressed(KeyCode::KeyR) {
            staging.clear();
            fluid::spawn_particle_block(
                &mut staging,
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::new(0.4, 0.3, 0.4),
                params.particle_radius * 2.0,
                Vec3::ZERO,
            );
            info!("Reset simulation with {} particles", staging.positions.len());
        }
    }
}

/// Component to mark particle visualizer entities
#[derive(Component)]
struct ParticleVisualizer;

/// System to visualize particles (simplified - uses instanced rendering in full version)
fn particle_visualizer(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    simulation: Option<Res<fluid::FluidSimulation>>,
    existing: Query<Entity, With<ParticleVisualizer>>,
    params: Res<FluidParams>,
) {
    // Remove old visualizers
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }

    let Some(sim) = simulation else {
        return;
    };

    if sim.positions.is_empty() {
        return;
    }

    // Create material for water
    let water_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.5, 0.9, 0.8),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.1,
        metallic: 0.0,
        ..default()
    });

    // Create small sphere mesh for particles
    let particle_mesh = meshes.add(Sphere::new(params.particle_radius).mesh().ico(1).unwrap());

    // Spawn particles (limited for performance in this simple visualizer)
    let max_display = 5000.min(sim.positions.len());
    let step = if sim.positions.len() > max_display {
        sim.positions.len() / max_display
    } else {
        1
    };

    for (i, pos) in sim.positions.iter().enumerate() {
        if i % step != 0 {
            continue;
        }

        commands.spawn((
            Mesh3d(particle_mesh.clone()),
            MeshMaterial3d(water_material.clone()),
            Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2])),
            ParticleVisualizer,
        ));
    }
}
