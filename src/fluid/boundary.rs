//! Boundary handling and collision detection for fluid simulation.
//!
//! This module provides various boundary conditions including:
//! - Box boundaries (AABB)
//! - Plane boundaries
//! - SDF-based boundaries
//! - Mesh boundaries (with optional SPH sampling)

use bevy::prelude::*;

/// Marker component for boundary objects.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct FluidBoundary;

/// Box-shaped boundary (AABB).
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct BoxBoundary {
    /// Minimum corner of the box.
    pub min: Vec3,
    /// Maximum corner of the box.
    pub max: Vec3,
    /// Whether particles should be contained inside (true) or outside (false).
    pub contain_inside: bool,
    /// Restitution coefficient for collisions (0 = no bounce, 1 = perfect bounce).
    pub restitution: f32,
    /// Friction coefficient (0 = no friction, 1 = full stop).
    pub friction: f32,
}

impl Default for BoxBoundary {
    fn default() -> Self {
        Self {
            min: Vec3::new(-5.0, 0.0, -5.0),
            max: Vec3::new(5.0, 10.0, 5.0),
            contain_inside: true,
            restitution: 0.0,
            friction: 0.1,
        }
    }
}

impl BoxBoundary {
    /// Create a box boundary centered at origin.
    pub fn centered(half_extents: Vec3) -> Self {
        Self {
            min: -half_extents,
            max: half_extents,
            ..default()
        }
    }

    /// Create a box boundary with custom bounds.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max, ..default() }
    }

    /// Set restitution.
    pub fn with_restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    /// Set friction.
    pub fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    /// Check if a point is inside the boundary.
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Get the signed distance from a point to the boundary surface.
    /// Negative = inside, positive = outside.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        let center = (self.min + self.max) * 0.5;
        let half_extents = (self.max - self.min) * 0.5;

        let q = (point - center).abs() - half_extents;
        let outside_dist = q.max(Vec3::ZERO).length();
        let inside_dist = q.x.max(q.y).max(q.z).min(0.0);

        outside_dist + inside_dist
    }

    /// Get the closest point on the boundary surface.
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            point.x.clamp(self.min.x, self.max.x),
            point.y.clamp(self.min.y, self.max.y),
            point.z.clamp(self.min.z, self.max.z),
        )
    }

    /// Get the surface normal at the closest point.
    pub fn surface_normal(&self, point: Vec3) -> Vec3 {
        let center = (self.min + self.max) * 0.5;
        let half_extents = (self.max - self.min) * 0.5;

        let relative = point - center;
        let scaled = relative / half_extents;

        // Find which face we're closest to
        let abs_scaled = scaled.abs();

        if abs_scaled.x >= abs_scaled.y && abs_scaled.x >= abs_scaled.z {
            Vec3::new(scaled.x.signum(), 0.0, 0.0)
        } else if abs_scaled.y >= abs_scaled.x && abs_scaled.y >= abs_scaled.z {
            Vec3::new(0.0, scaled.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, scaled.z.signum())
        }
    }

    /// Apply boundary collision to a particle.
    pub fn apply_collision(
        &self,
        position: &mut Vec3,
        velocity: &mut Vec3,
        particle_radius: f32,
    ) {
        if self.contain_inside {
            // Keep particle inside
            let effective_min = self.min + Vec3::splat(particle_radius);
            let effective_max = self.max - Vec3::splat(particle_radius);

            // X axis
            if position.x < effective_min.x {
                position.x = effective_min.x;
                velocity.x = velocity.x.abs() * self.restitution;
                velocity.y *= 1.0 - self.friction;
                velocity.z *= 1.0 - self.friction;
            } else if position.x > effective_max.x {
                position.x = effective_max.x;
                velocity.x = -velocity.x.abs() * self.restitution;
                velocity.y *= 1.0 - self.friction;
                velocity.z *= 1.0 - self.friction;
            }

            // Y axis
            if position.y < effective_min.y {
                position.y = effective_min.y;
                velocity.y = velocity.y.abs() * self.restitution;
                velocity.x *= 1.0 - self.friction;
                velocity.z *= 1.0 - self.friction;
            } else if position.y > effective_max.y {
                position.y = effective_max.y;
                velocity.y = -velocity.y.abs() * self.restitution;
                velocity.x *= 1.0 - self.friction;
                velocity.z *= 1.0 - self.friction;
            }

            // Z axis
            if position.z < effective_min.z {
                position.z = effective_min.z;
                velocity.z = velocity.z.abs() * self.restitution;
                velocity.x *= 1.0 - self.friction;
                velocity.y *= 1.0 - self.friction;
            } else if position.z > effective_max.z {
                position.z = effective_max.z;
                velocity.z = -velocity.z.abs() * self.restitution;
                velocity.x *= 1.0 - self.friction;
                velocity.y *= 1.0 - self.friction;
            }
        }
    }
}

/// Plane boundary (infinite plane).
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct PlaneBoundary {
    /// Normal of the plane (pointing towards the valid side).
    pub normal: Vec3,
    /// Distance from origin along the normal.
    pub distance: f32,
    /// Restitution coefficient.
    pub restitution: f32,
    /// Friction coefficient.
    pub friction: f32,
}

impl Default for PlaneBoundary {
    fn default() -> Self {
        Self {
            normal: Vec3::Y,
            distance: 0.0,
            restitution: 0.0,
            friction: 0.1,
        }
    }
}

impl PlaneBoundary {
    /// Create a floor at a given height.
    pub fn floor(height: f32) -> Self {
        Self {
            normal: Vec3::Y,
            distance: height,
            ..default()
        }
    }

    /// Create a wall with given normal and point on the plane.
    pub fn wall(normal: Vec3, point: Vec3) -> Self {
        let normal = normal.normalize();
        Self {
            normal,
            distance: normal.dot(point),
            ..default()
        }
    }

    /// Get signed distance to the plane.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.distance
    }

    /// Apply boundary collision.
    pub fn apply_collision(
        &self,
        position: &mut Vec3,
        velocity: &mut Vec3,
        particle_radius: f32,
    ) {
        let dist = self.signed_distance(*position);
        if dist < particle_radius {
            // Push particle out
            *position += self.normal * (particle_radius - dist);

            // Reflect velocity
            let vn = self.normal.dot(*velocity);
            if vn < 0.0 {
                // Decompose velocity
                let v_normal = self.normal * vn;
                let v_tangent = *velocity - v_normal;

                // Apply restitution and friction
                *velocity = v_tangent * (1.0 - self.friction) - v_normal * self.restitution;
            }
        }
    }
}

/// Sphere boundary.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct SphereBoundary {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
    /// Whether particles should be contained inside (true) or outside (false).
    pub contain_inside: bool,
    /// Restitution coefficient.
    pub restitution: f32,
    /// Friction coefficient.
    pub friction: f32,
}

impl Default for SphereBoundary {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 5.0,
            contain_inside: true,
            restitution: 0.0,
            friction: 0.1,
        }
    }
}

impl SphereBoundary {
    /// Create a sphere boundary.
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius,
            ..default()
        }
    }

    /// Get signed distance to the sphere surface.
    /// Negative = inside, positive = outside.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        (point - self.center).length() - self.radius
    }

    /// Apply boundary collision.
    pub fn apply_collision(
        &self,
        position: &mut Vec3,
        velocity: &mut Vec3,
        particle_radius: f32,
    ) {
        let to_center = self.center - *position;
        let dist_to_center = to_center.length();

        if self.contain_inside {
            // Keep inside sphere
            let effective_radius = self.radius - particle_radius;
            if dist_to_center > effective_radius && dist_to_center > 1e-6 {
                let normal = to_center / dist_to_center;
                *position = self.center - normal * effective_radius;

                // Reflect velocity
                let vn = normal.dot(*velocity);
                if vn < 0.0 {
                    let v_normal = normal * vn;
                    let v_tangent = *velocity - v_normal;
                    *velocity = v_tangent * (1.0 - self.friction) - v_normal * self.restitution;
                }
            }
        } else {
            // Keep outside sphere
            let effective_radius = self.radius + particle_radius;
            if dist_to_center < effective_radius && dist_to_center > 1e-6 {
                let normal = -to_center / dist_to_center;
                *position = self.center + normal * effective_radius;

                // Reflect velocity
                let vn = normal.dot(*velocity);
                if vn < 0.0 {
                    let v_normal = normal * vn;
                    let v_tangent = *velocity - v_normal;
                    *velocity = v_tangent * (1.0 - self.friction) - v_normal * self.restitution;
                }
            }
        }
    }
}

/// GPU-compatible boundary data for compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBoundary {
    /// Boundary type: 0=none, 1=box, 2=plane, 3=sphere
    pub boundary_type: u32,
    /// Restitution.
    pub restitution: f32,
    /// Friction.
    pub friction: f32,
    /// Padding.
    pub _padding1: u32,

    /// For box: min corner. For plane: normal (xyz) + distance (w). For sphere: center (xyz) + radius (w).
    pub param1: [f32; 4],
    /// For box: max corner. Unused for others.
    pub param2: [f32; 4],
}

impl From<&BoxBoundary> for GpuBoundary {
    fn from(b: &BoxBoundary) -> Self {
        Self {
            boundary_type: 1,
            restitution: b.restitution,
            friction: b.friction,
            _padding1: 0,
            param1: [b.min.x, b.min.y, b.min.z, if b.contain_inside { 1.0 } else { 0.0 }],
            param2: [b.max.x, b.max.y, b.max.z, 0.0],
        }
    }
}

impl From<&PlaneBoundary> for GpuBoundary {
    fn from(b: &PlaneBoundary) -> Self {
        Self {
            boundary_type: 2,
            restitution: b.restitution,
            friction: b.friction,
            _padding1: 0,
            param1: [b.normal.x, b.normal.y, b.normal.z, b.distance],
            param2: [0.0; 4],
        }
    }
}

impl From<&SphereBoundary> for GpuBoundary {
    fn from(b: &SphereBoundary) -> Self {
        Self {
            boundary_type: 3,
            restitution: b.restitution,
            friction: b.friction,
            _padding1: 0,
            param1: [b.center.x, b.center.y, b.center.z, b.radius],
            param2: [if b.contain_inside { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        }
    }
}

/// Bundle for a box-shaped boundary container.
#[derive(Bundle, Default)]
pub struct BoxBoundaryBundle {
    pub marker: FluidBoundary,
    pub boundary: BoxBoundary,
}

/// Bundle for a plane boundary.
#[derive(Bundle, Default)]
pub struct PlaneBoundaryBundle {
    pub marker: FluidBoundary,
    pub boundary: PlaneBoundary,
}

/// Bundle for a sphere boundary.
#[derive(Bundle, Default)]
pub struct SphereBoundaryBundle {
    pub marker: FluidBoundary,
    pub boundary: SphereBoundary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_boundary_contains() {
        let boundary = BoxBoundary::new(Vec3::ZERO, Vec3::ONE);

        assert!(boundary.contains(Vec3::splat(0.5)));
        assert!(!boundary.contains(Vec3::new(1.5, 0.5, 0.5)));
    }

    #[test]
    fn test_box_boundary_collision() {
        let boundary = BoxBoundary::new(Vec3::ZERO, Vec3::ONE).with_restitution(0.5);

        let mut pos = Vec3::new(-0.1, 0.5, 0.5);
        let mut vel = Vec3::new(-1.0, 0.0, 0.0);

        boundary.apply_collision(&mut pos, &mut vel, 0.0);

        assert!(pos.x >= 0.0);
        assert!(vel.x >= 0.0); // Should have bounced
    }

    #[test]
    fn test_plane_boundary_distance() {
        let plane = PlaneBoundary::floor(0.0);

        assert_eq!(plane.signed_distance(Vec3::new(0.0, 1.0, 0.0)), 1.0);
        assert_eq!(plane.signed_distance(Vec3::new(0.0, -1.0, 0.0)), -1.0);
        assert_eq!(plane.signed_distance(Vec3::ZERO), 0.0);
    }

    #[test]
    fn test_sphere_boundary_distance() {
        let sphere = SphereBoundary::new(Vec3::ZERO, 1.0);

        assert_eq!(sphere.signed_distance(Vec3::ZERO), -1.0); // Inside
        assert_eq!(sphere.signed_distance(Vec3::new(1.0, 0.0, 0.0)), 0.0); // On surface
        assert_eq!(sphere.signed_distance(Vec3::new(2.0, 0.0, 0.0)), 1.0); // Outside
    }
}
