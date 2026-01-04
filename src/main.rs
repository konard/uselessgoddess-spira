//! Spira - High-Performance Dental Fluid Simulation
//!
//! A GPU-accelerated fluid simulation for dental irrigation visualization using
//! Lattice Boltzmann Method (LBM) D3Q19 and Bevy game engine.
//!
//! ## Features
//! - Real-time LBM fluid simulation on GPU via WGSL compute shaders
//! - 3D visualization of fluid density field
//! - Interactive injection point controls

mod physics;

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    render::settings::{PowerPreference, RenderCreation, WgpuSettings},
    window::PresentMode,
};

use physics::{
    create_fluid_texture, FluidCorePlugin, FluidTextures, InjectionState, SimulationConfig,
    DEFAULT_GRID_SIZE,
};

/// Application configuration
const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Spira - Dental Fluid Simulation".into(),
                        resolution: bevy::window::WindowResolution::new(WINDOW_WIDTH, WINDOW_HEIGHT),
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(bevy::render::RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        power_preference: PowerPreference::HighPerformance,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(FluidCorePlugin {
            grid_size: DEFAULT_GRID_SIZE,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                camera_controller,
                injection_controller,
                update_ui,
                switch_display_texture,
            ),
        )
        .run();
}

/// Marker component for the main camera
#[derive(Component)]
struct MainCamera {
    /// Distance from target
    distance: f32,
    /// Rotation around Y axis
    yaw: f32,
    /// Rotation around X axis
    pitch: f32,
}

impl Default for MainCamera {
    fn default() -> Self {
        Self {
            distance: 150.0,
            yaw: 0.5,
            pitch: 0.3,
        }
    }
}

/// UI text component
#[derive(Component)]
struct UiText;

/// Marker for the fluid visualization sprite
#[derive(Component)]
struct FluidDisplay;

/// Sets up the initial scene
fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let grid_size = DEFAULT_GRID_SIZE;

    // Create ping-pong textures for LBM simulation
    let density_a = create_fluid_texture(&mut images, grid_size);
    let density_b = create_fluid_texture(&mut images, grid_size);

    // Store references for the compute shader
    commands.insert_resource(FluidTextures {
        density_a: density_a.clone(),
        density_b: density_b.clone(),
    });

    commands.insert_resource(InjectionState::default());

    // Create a 2D slice visualization using a sprite
    // For the PoC, we'll show a 2D cross-section of the 3D fluid
    commands.spawn((
        Sprite {
            image: density_a,
            custom_size: Some(Vec2::splat(grid_size as f32 * 4.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
        FluidDisplay,
    ));

    // 3D Camera for domain visualization
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(100.0, 80.0, 150.0)
            .looking_at(Vec3::splat(grid_size as f32 / 2.0), Vec3::Y),
        MainCamera::default(),
    ));

    // Create a wireframe visualization of the simulation domain
    spawn_domain_visualization(&mut commands, grid_size as f32);

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 100.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.8, 0.85, 1.0),
        brightness: 200.0,
        affects_lightmapped_meshes: true,
    });

    // UI overlay
    commands.spawn((
        Text::new(""),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        TextColor(Color::WHITE),
        UiText,
    ));
}

/// Creates a wireframe box to show the simulation domain
fn spawn_domain_visualization(commands: &mut Commands, size: f32) {
    // We use gizmos for wireframe, but for a persistent visualization
    // we'll use simple 3D markers at the corners
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, size, 0.0),
        Vec3::new(size, size, 0.0),
        Vec3::new(0.0, 0.0, size),
        Vec3::new(size, 0.0, size),
        Vec3::new(0.0, size, size),
        Vec3::new(size, size, size),
    ];

    // Create corner markers using point lights as visual indicators
    for corner in corners {
        commands.spawn((
            PointLight {
                intensity: 5000.0,
                range: 10.0,
                color: Color::srgb(0.3, 0.5, 0.8),
                shadows_enabled: false,
                ..default()
            },
            Transform::from_translation(corner),
        ));
    }
}

/// Switch between density textures to display the most recent computation
fn switch_display_texture(
    fluid_textures: Option<Res<FluidTextures>>,
    mut sprites: Query<&mut Sprite, With<FluidDisplay>>,
    time: Res<Time>,
) {
    let Some(textures) = fluid_textures else {
        return;
    };

    // Toggle display texture every frame to show latest result
    for mut sprite in &mut sprites {
        let frame = (time.elapsed_secs() * 60.0) as u32;
        if frame % 2 == 0 {
            sprite.image = textures.density_a.clone();
        } else {
            sprite.image = textures.density_b.clone();
        }
    }
}

/// Camera orbit controller
fn camera_controller(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<bevy::input::mouse::MouseWheel>,
    mut camera: Query<(&mut Transform, &mut MainCamera)>,
    config: Option<Res<SimulationConfig>>,
) {
    let Some(config) = config else { return };
    let Ok((mut transform, mut cam)) = camera.single_mut() else {
        return;
    };

    let center = Vec3::splat(config.grid_size as f32 / 2.0);

    // Mouse drag to rotate
    if mouse.pressed(MouseButton::Right) {
        for motion in mouse_motion.read() {
            cam.yaw -= motion.delta.x * 0.01;
            cam.pitch = (cam.pitch - motion.delta.y * 0.01).clamp(-1.4, 1.4);
        }
    } else {
        mouse_motion.clear();
    }

    // Scroll to zoom
    for scroll in mouse_wheel.read() {
        cam.distance = (cam.distance - scroll.y * 10.0).clamp(30.0, 300.0);
    }

    // Keyboard rotation
    let rotation_speed = 1.5 * time.delta_secs();
    if keyboard.pressed(KeyCode::ArrowLeft) {
        cam.yaw += rotation_speed;
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        cam.yaw -= rotation_speed;
    }
    if keyboard.pressed(KeyCode::ArrowUp) {
        cam.pitch = (cam.pitch + rotation_speed).min(1.4);
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        cam.pitch = (cam.pitch - rotation_speed).max(-1.4);
    }

    // Update camera position
    let offset = Vec3::new(
        cam.distance * cam.yaw.cos() * cam.pitch.cos(),
        cam.distance * cam.pitch.sin(),
        cam.distance * cam.yaw.sin() * cam.pitch.cos(),
    );

    transform.translation = center + offset;
    transform.look_at(center, Vec3::Y);
}

/// Injection point controller
fn injection_controller(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    injection: Option<ResMut<InjectionState>>,
    config: Option<Res<SimulationConfig>>,
) {
    let Some(mut injection) = injection else { return };
    let Some(config) = config else { return };

    let speed = 30.0 * time.delta_secs();
    let max = config.grid_size as f32 - 5.0;
    let min = 5.0;

    // Move injection point with WASD + QE
    if keyboard.pressed(KeyCode::KeyW) {
        injection.origin.z = (injection.origin.z - speed).max(min);
    }
    if keyboard.pressed(KeyCode::KeyS) {
        injection.origin.z = (injection.origin.z + speed).min(max);
    }
    if keyboard.pressed(KeyCode::KeyA) {
        injection.origin.x = (injection.origin.x - speed).max(min);
    }
    if keyboard.pressed(KeyCode::KeyD) {
        injection.origin.x = (injection.origin.x + speed).min(max);
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        injection.origin.y = (injection.origin.y + speed).min(max);
    }
    if keyboard.pressed(KeyCode::KeyE) {
        injection.origin.y = (injection.origin.y - speed).max(min);
    }

    // Adjust pressure with +/-
    if keyboard.pressed(KeyCode::Equal) || keyboard.pressed(KeyCode::NumpadAdd) {
        injection.pressure = (injection.pressure + 0.1 * time.delta_secs()).min(1.0);
    }
    if keyboard.pressed(KeyCode::Minus) || keyboard.pressed(KeyCode::NumpadSubtract) {
        injection.pressure = (injection.pressure - 0.1 * time.delta_secs()).max(0.0);
    }

    // Toggle injection with Space
    if keyboard.just_pressed(KeyCode::Space) {
        injection.active = !injection.active;
    }

    // Reset with R
    if keyboard.just_pressed(KeyCode::KeyR) {
        *injection = InjectionState::default();
    }
}

/// Update UI text
fn update_ui(
    injection: Option<Res<InjectionState>>,
    config: Option<Res<SimulationConfig>>,
    mut text: Query<&mut Text, With<UiText>>,
    time: Res<Time>,
) {
    let Some(injection) = injection else { return };
    let Some(config) = config else { return };
    let Ok(mut text) = text.single_mut() else {
        return;
    };

    let fps = 1.0 / time.delta_secs();
    let status = if injection.active { "ACTIVE" } else { "PAUSED" };

    **text = format!(
        "Spira - Dental Fluid Simulation (LBM D3Q19)\n\
         FPS: {:.1}\n\
         Grid: {}^3 = {} voxels\n\
         \n\
         Injection: {}\n\
         Position: ({:.1}, {:.1}, {:.1})\n\
         Pressure: {:.2}\n\
         \n\
         Controls:\n\
         WASD   - Move injection XZ\n\
         Q/E    - Move injection Y\n\
         +/-    - Adjust pressure\n\
         Space  - Toggle injection\n\
         R      - Reset injection\n\
         Arrows - Rotate camera\n\
         Right-drag - Orbit camera\n\
         Scroll - Zoom",
        fps,
        config.grid_size,
        config.grid_size.pow(3),
        status,
        injection.origin.x,
        injection.origin.y,
        injection.origin.z,
        injection.pressure
    );
}
