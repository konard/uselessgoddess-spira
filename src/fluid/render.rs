//! Fluid rendering systems.
//!
//! This module provides multiple rendering methods for fluid particles:
//! - Point sprite rendering (fast, basic)
//! - Sphere mesh instancing (medium quality)
//! - Screen-space fluid rendering (high quality, advanced)

use bevy::prelude::*;

/// Configuration for fluid rendering.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct FluidRenderConfig {
    /// Rendering method to use.
    pub method: RenderMethod,
    /// Base color for fluid particles.
    pub base_color: Color,
    /// Scale factor for particle visual size.
    pub size_scale: f32,
    /// Enable transparency.
    pub transparent: bool,
    /// Alpha value for transparent rendering.
    pub alpha: f32,
    /// Enable velocity-based coloring.
    pub velocity_coloring: bool,
    /// Maximum velocity for color mapping.
    pub max_velocity_color: f32,
    /// Enable depth-based transparency (for screen-space rendering).
    pub depth_transparency: bool,
}

impl Default for FluidRenderConfig {
    fn default() -> Self {
        Self {
            method: RenderMethod::SphereMesh,
            base_color: Color::srgb(0.2, 0.5, 0.9),
            size_scale: 1.0,
            transparent: true,
            alpha: 0.8,
            velocity_coloring: false,
            max_velocity_color: 5.0,
            depth_transparency: false,
        }
    }
}

/// Rendering method for fluid particles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum RenderMethod {
    /// Point sprites (fastest, lowest quality).
    PointSprite,
    /// Instanced sphere meshes (good balance).
    #[default]
    SphereMesh,
    /// Screen-space fluid rendering (highest quality, most expensive).
    ScreenSpace,
}

/// Marker component for rendered fluid particles.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct FluidParticleVisual;

/// Handle to the shared fluid particle mesh.
#[derive(Resource, Clone, Debug)]
pub struct FluidParticleMesh(pub Handle<Mesh>);

/// Handle to the shared fluid particle material.
#[derive(Resource, Clone, Debug)]
pub struct FluidParticleMaterial(pub Handle<StandardMaterial>);

/// Bundle for visual representation of a fluid particle.
#[derive(Bundle)]
pub struct FluidParticleVisualBundle {
    pub marker: FluidParticleVisual,
    pub mesh: Mesh3d,
    pub material: MeshMaterial3d<StandardMaterial>,
    pub transform: Transform,
}

/// Parameters for screen-space fluid rendering (future implementation).
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct ScreenSpaceFluidParams {
    /// Blur iterations for depth smoothing.
    pub blur_iterations: u32,
    /// Blur radius.
    pub blur_radius: f32,
    /// Thickness scale.
    pub thickness_scale: f32,
    /// Index of refraction.
    pub refraction_index: f32,
    /// Fresnel power.
    pub fresnel_power: f32,
    /// Specular intensity.
    pub specular_intensity: f32,
}

impl Default for ScreenSpaceFluidParams {
    fn default() -> Self {
        Self {
            blur_iterations: 4,
            blur_radius: 5.0,
            thickness_scale: 1.0,
            refraction_index: 1.33, // Water
            fresnel_power: 5.0,
            specular_intensity: 0.5,
        }
    }
}

/// Get a color based on velocity magnitude.
pub fn velocity_to_color(velocity: Vec3, max_velocity: f32, _base_color: Color) -> Color {
    let speed = velocity.length();
    let t = (speed / max_velocity).clamp(0.0, 1.0);

    // Interpolate from blue (slow) to red (fast)
    let slow_color = Color::srgb(0.2, 0.4, 0.9);
    let fast_color = Color::srgb(0.9, 0.3, 0.2);

    let slow = slow_color.to_linear();
    let fast = fast_color.to_linear();

    Color::linear_rgba(
        slow.red + (fast.red - slow.red) * t,
        slow.green + (fast.green - slow.green) * t,
        slow.blue + (fast.blue - slow.blue) * t,
        1.0,
    )
}

/// Create a sphere mesh for particle rendering.
pub fn create_particle_mesh(subdivisions: u32) -> Mesh {
    // Use Bevy's built-in sphere
    Sphere::new(1.0)
        .mesh()
        .ico(subdivisions)
        .expect("Failed to create sphere mesh")
}

/// Create a material for fluid particles.
pub fn create_particle_material(config: &FluidRenderConfig) -> StandardMaterial {
    let mut color = config.base_color.to_linear();
    if config.transparent {
        color = color.with_alpha(config.alpha);
    }

    StandardMaterial {
        base_color: color.into(),
        alpha_mode: if config.transparent {
            AlphaMode::Blend
        } else {
            AlphaMode::Opaque
        },
        perceptual_roughness: 0.3,
        metallic: 0.0,
        reflectance: 0.5,
        ..default()
    }
}

/// Debug visualization helper - creates a wireframe box mesh.
/// Note: Uses triangles instead of lines for broader compatibility.
pub fn create_boundary_wireframe_mesh(min: Vec3, max: Vec3) -> Mesh {
    let size = max - min;

    // Create a simple box using Bevy's built-in
    Cuboid::new(size.x, size.y, size.z).mesh().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_to_color() {
        let base = Color::srgb(0.5, 0.5, 0.5);

        // Zero velocity should be blue-ish
        let slow = velocity_to_color(Vec3::ZERO, 10.0, base);
        assert!(slow.to_linear().blue > slow.to_linear().red);

        // High velocity should be red-ish
        let fast = velocity_to_color(Vec3::new(10.0, 0.0, 0.0), 10.0, base);
        assert!(fast.to_linear().red > fast.to_linear().blue);
    }

    #[test]
    fn test_particle_mesh_creation() {
        let mesh = create_particle_mesh(2);
        // Just verify it doesn't panic and has some vertices
        assert!(mesh.count_vertices() > 0);
    }
}
