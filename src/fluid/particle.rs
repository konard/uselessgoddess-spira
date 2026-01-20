//! Fluid particle data structures and spawning utilities.
//!
//! This module provides components and systems for managing fluid particles,
//! including emitters for spawning particles in various shapes.

use bevy::prelude::*;

/// Marker component for fluid particles.
///
/// This component identifies an entity as a fluid particle and should be used
/// with the required particle data components.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct FluidParticle;

/// Position of a fluid particle.
///
/// Separate from Transform for more efficient GPU processing.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParticlePosition(pub Vec3);

/// Predicted position during constraint solving.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParticlePredictedPosition(pub Vec3);

/// Velocity of a fluid particle.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParticleVelocity(pub Vec3);

/// Density of a particle (computed during simulation).
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParticleDensity(pub f32);

/// Lagrange multiplier (lambda) for density constraint.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParticleLambda(pub f32);

/// Position correction (delta_p) from constraint solving.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParticleDeltaPosition(pub Vec3);

/// Particle color for visualization.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct ParticleColor(pub Color);

impl Default for ParticleColor {
    fn default() -> Self {
        Self(Color::srgb(0.2, 0.5, 0.9)) // Blue water color
    }
}

/// Bundle for spawning a single fluid particle.
#[derive(Bundle, Default)]
pub struct FluidParticleBundle {
    pub marker: FluidParticle,
    pub position: ParticlePosition,
    pub predicted_position: ParticlePredictedPosition,
    pub velocity: ParticleVelocity,
    pub density: ParticleDensity,
    pub lambda: ParticleLambda,
    pub delta_position: ParticleDeltaPosition,
    pub color: ParticleColor,
}

impl FluidParticleBundle {
    /// Create a particle at a given position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position: ParticlePosition(position),
            predicted_position: ParticlePredictedPosition(position),
            ..default()
        }
    }

    /// Create a particle with initial velocity.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = ParticleVelocity(velocity);
        self
    }

    /// Create a particle with a specific color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = ParticleColor(color);
        self
    }
}

/// Shape for fluid emitters.
#[derive(Clone, Debug, Reflect)]
pub enum EmitterShape {
    /// Box-shaped emitter.
    Box {
        half_extents: Vec3,
    },
    /// Spherical emitter.
    Sphere {
        radius: f32,
    },
    /// Cylindrical emitter.
    Cylinder {
        radius: f32,
        half_height: f32,
    },
}

impl Default for EmitterShape {
    fn default() -> Self {
        EmitterShape::Box {
            half_extents: Vec3::splat(0.5),
        }
    }
}

/// Fluid emitter component for spawning particles.
///
/// Attach this to an entity with a Transform to spawn fluid particles
/// in the specified shape at the entity's position.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component)]
pub struct FluidEmitter {
    /// Shape of the emitter volume.
    pub shape: EmitterShape,

    /// Spacing between particles in meters.
    pub particle_spacing: f32,

    /// Initial velocity for spawned particles.
    pub initial_velocity: Vec3,

    /// Color for spawned particles.
    pub color: Color,

    /// Whether to emit particles only once (on spawn) or continuously.
    pub one_shot: bool,

    /// Whether this emitter has already spawned (for one_shot).
    pub has_spawned: bool,

    /// Emission rate in particles per second (for continuous emitters).
    pub emission_rate: f32,

    /// Accumulated time for continuous emission.
    pub accumulated_time: f32,

    /// Whether the emitter is enabled.
    pub enabled: bool,

    /// Maximum number of particles to spawn (0 = unlimited).
    pub max_particles: u32,

    /// Number of particles spawned so far.
    pub particles_spawned: u32,
}

impl Default for FluidEmitter {
    fn default() -> Self {
        Self {
            shape: EmitterShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            particle_spacing: 0.1,
            initial_velocity: Vec3::ZERO,
            color: Color::srgb(0.2, 0.5, 0.9),
            one_shot: true,
            has_spawned: false,
            emission_rate: 1000.0,
            accumulated_time: 0.0,
            enabled: true,
            max_particles: 0,
            particles_spawned: 0,
        }
    }
}

impl FluidEmitter {
    /// Create a box-shaped emitter.
    pub fn box_emitter(half_extents: Vec3, particle_spacing: f32) -> Self {
        Self {
            shape: EmitterShape::Box { half_extents },
            particle_spacing,
            one_shot: true,
            ..default()
        }
    }

    /// Create a spherical emitter.
    pub fn sphere_emitter(radius: f32, particle_spacing: f32) -> Self {
        Self {
            shape: EmitterShape::Sphere { radius },
            particle_spacing,
            one_shot: true,
            ..default()
        }
    }

    /// Set initial velocity.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.initial_velocity = velocity;
        self
    }

    /// Set particle color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Make the emitter continuous.
    pub fn continuous(mut self, rate: f32) -> Self {
        self.one_shot = false;
        self.emission_rate = rate;
        self
    }

    /// Generate particle positions for this emitter shape.
    pub fn generate_positions(&self, transform: &Transform) -> Vec<Vec3> {
        let mut positions = Vec::new();
        let spacing = self.particle_spacing;

        match &self.shape {
            EmitterShape::Box { half_extents } => {
                let min = -*half_extents;
                let max = *half_extents;

                let mut x = min.x;
                while x <= max.x {
                    let mut y = min.y;
                    while y <= max.y {
                        let mut z = min.z;
                        while z <= max.z {
                            let local_pos = Vec3::new(x, y, z);
                            let world_pos = transform.transform_point(local_pos);
                            positions.push(world_pos);
                            z += spacing;
                        }
                        y += spacing;
                    }
                    x += spacing;
                }
            }
            EmitterShape::Sphere { radius } => {
                let min = -Vec3::splat(*radius);
                let max = Vec3::splat(*radius);

                let mut x = min.x;
                while x <= max.x {
                    let mut y = min.y;
                    while y <= max.y {
                        let mut z = min.z;
                        while z <= max.z {
                            let local_pos = Vec3::new(x, y, z);
                            if local_pos.length_squared() <= radius * radius {
                                let world_pos = transform.transform_point(local_pos);
                                positions.push(world_pos);
                            }
                            z += spacing;
                        }
                        y += spacing;
                    }
                    x += spacing;
                }
            }
            EmitterShape::Cylinder { radius, half_height } => {
                let mut y = -half_height;
                while y <= *half_height {
                    let mut x = -*radius;
                    while x <= *radius {
                        let mut z = -*radius;
                        while z <= *radius {
                            if x * x + z * z <= radius * radius {
                                let local_pos = Vec3::new(x, y, z);
                                let world_pos = transform.transform_point(local_pos);
                                positions.push(world_pos);
                            }
                            z += spacing;
                        }
                        x += spacing;
                    }
                    y += spacing;
                }
            }
        }

        positions
    }
}

/// CPU-side particle data for easy manipulation before GPU transfer.
/// This is used when we need to process particles on the CPU before
/// uploading to GPU buffers.
#[derive(Clone, Debug, Default)]
pub struct ParticleData {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub colors: Vec<Color>,
}

impl ParticleData {
    /// Create new particle data with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            colors: Vec::with_capacity(capacity),
        }
    }

    /// Add a particle.
    pub fn push(&mut self, position: Vec3, velocity: Vec3, color: Color) {
        self.positions.push(position);
        self.velocities.push(velocity);
        self.colors.push(color);
    }

    /// Get the number of particles.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.colors.clear();
    }
}

/// GPU-compatible particle data layout.
/// Uses Structure of Arrays (SoA) for better GPU memory access patterns.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    /// Position (xyz) + density (w).
    pub position_density: [f32; 4],
    /// Velocity (xyz) + lambda (w).
    pub velocity_lambda: [f32; 4],
    /// Predicted position (xyz) + unused (w).
    pub predicted_position: [f32; 4],
    /// Color (rgba).
    pub color: [f32; 4],
}

impl GpuParticle {
    pub fn new(position: Vec3, velocity: Vec3, color: Color) -> Self {
        let color_linear = color.to_linear();
        Self {
            position_density: [position.x, position.y, position.z, 0.0],
            velocity_lambda: [velocity.x, velocity.y, velocity.z, 0.0],
            predicted_position: [position.x, position.y, position.z, 0.0],
            color: [color_linear.red, color_linear.green, color_linear.blue, color_linear.alpha],
        }
    }

    pub fn position(&self) -> Vec3 {
        Vec3::new(
            self.position_density[0],
            self.position_density[1],
            self.position_density[2],
        )
    }

    pub fn velocity(&self) -> Vec3 {
        Vec3::new(
            self.velocity_lambda[0],
            self.velocity_lambda[1],
            self.velocity_lambda[2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_emitter_positions() {
        let emitter = FluidEmitter::box_emitter(Vec3::splat(0.5), 0.5);
        let transform = Transform::from_translation(Vec3::ZERO);
        let positions = emitter.generate_positions(&transform);

        // Should generate a 3x3x3 grid
        assert!(!positions.is_empty());
        assert!(positions.len() <= 27); // At most 3^3 particles
    }

    #[test]
    fn test_sphere_emitter_positions() {
        let emitter = FluidEmitter::sphere_emitter(1.0, 0.5);
        let transform = Transform::from_translation(Vec3::ZERO);
        let positions = emitter.generate_positions(&transform);

        // All positions should be within the sphere
        for pos in &positions {
            assert!(pos.length_squared() <= 1.0 + 0.01); // Small epsilon for floating point
        }
    }

    #[test]
    fn test_gpu_particle_layout() {
        // Verify GPU particle size is what we expect
        assert_eq!(std::mem::size_of::<GpuParticle>(), 64);
    }
}
