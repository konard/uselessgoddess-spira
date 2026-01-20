//! Spatial hashing for efficient neighbor search.
//!
//! This module implements a GPU-friendly spatial hash grid for O(n) neighbor search,
//! which is critical for PBF performance. The algorithm uses count sort with
//! parallel prefix scan for optimal GPU utilization.

use bevy::prelude::*;

/// Configuration for the spatial hash grid.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct SpatialHashConfig {
    /// Size of each grid cell (should be >= smoothing_radius).
    pub cell_size: f32,

    /// Number of cells in each dimension.
    pub grid_size: UVec3,

    /// Origin of the grid (minimum corner).
    pub grid_origin: Vec3,

    /// Maximum particles per cell (for memory allocation).
    pub max_particles_per_cell: u32,

    /// Table size for hash table (should be prime for good distribution).
    pub hash_table_size: u32,
}

impl Default for SpatialHashConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.2,
            grid_size: UVec3::new(128, 128, 128),
            grid_origin: Vec3::new(-12.8, -12.8, -12.8),
            max_particles_per_cell: 64,
            hash_table_size: 262147, // Prime number near 2^18
        }
    }
}

impl SpatialHashConfig {
    /// Create a configuration that covers a specific domain.
    pub fn for_domain(min: Vec3, max: Vec3, cell_size: f32) -> Self {
        let size = max - min;
        let grid_size = (size / cell_size).ceil().as_uvec3();

        Self {
            cell_size,
            grid_size,
            grid_origin: min,
            ..default()
        }
    }

    /// Calculate the grid cell for a position.
    pub fn position_to_cell(&self, position: Vec3) -> IVec3 {
        let relative = position - self.grid_origin;
        (relative / self.cell_size).floor().as_ivec3()
    }

    /// Calculate the hash for a cell coordinate.
    /// Uses a simple hash function that works well for spatial data.
    pub fn cell_to_hash(&self, cell: IVec3) -> u32 {
        // Large primes for hash mixing
        const P1: u32 = 73856093;
        const P2: u32 = 19349663;
        const P3: u32 = 83492791;

        let x = cell.x as u32;
        let y = cell.y as u32;
        let z = cell.z as u32;

        (x.wrapping_mul(P1) ^ y.wrapping_mul(P2) ^ z.wrapping_mul(P3)) % self.hash_table_size
    }

    /// Calculate hash directly from position.
    pub fn position_to_hash(&self, position: Vec3) -> u32 {
        self.cell_to_hash(self.position_to_cell(position))
    }

    /// Check if a cell coordinate is within the grid bounds.
    pub fn is_valid_cell(&self, cell: IVec3) -> bool {
        cell.x >= 0
            && cell.y >= 0
            && cell.z >= 0
            && (cell.x as u32) < self.grid_size.x
            && (cell.y as u32) < self.grid_size.y
            && (cell.z as u32) < self.grid_size.z
    }

    /// Get neighboring cell offsets for 3x3x3 neighborhood.
    pub fn neighbor_offsets() -> &'static [IVec3; 27] {
        static OFFSETS: [IVec3; 27] = [
            IVec3::new(-1, -1, -1),
            IVec3::new(-1, -1, 0),
            IVec3::new(-1, -1, 1),
            IVec3::new(-1, 0, -1),
            IVec3::new(-1, 0, 0),
            IVec3::new(-1, 0, 1),
            IVec3::new(-1, 1, -1),
            IVec3::new(-1, 1, 0),
            IVec3::new(-1, 1, 1),
            IVec3::new(0, -1, -1),
            IVec3::new(0, -1, 0),
            IVec3::new(0, -1, 1),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 1, -1),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 1, 1),
            IVec3::new(1, -1, -1),
            IVec3::new(1, -1, 0),
            IVec3::new(1, -1, 1),
            IVec3::new(1, 0, -1),
            IVec3::new(1, 0, 0),
            IVec3::new(1, 0, 1),
            IVec3::new(1, 1, -1),
            IVec3::new(1, 1, 0),
            IVec3::new(1, 1, 1),
        ];
        &OFFSETS
    }

    /// Get all neighbor hashes for a position.
    pub fn get_neighbor_hashes(&self, position: Vec3) -> Vec<u32> {
        let cell = self.position_to_cell(position);
        Self::neighbor_offsets()
            .iter()
            .filter_map(|offset| {
                let neighbor_cell = cell + *offset;
                if self.is_valid_cell(neighbor_cell) {
                    Some(self.cell_to_hash(neighbor_cell))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// GPU-compatible spatial hash entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuHashEntry {
    /// Start index in the sorted particle array.
    pub start: u32,
    /// Number of particles in this cell.
    pub count: u32,
    /// Padding for alignment.
    pub _padding: [u32; 2],
}

/// GPU-compatible grid cell data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGridCell {
    /// Cell coordinate.
    pub coord: [i32; 3],
    /// Hash value for this cell.
    pub hash: u32,
}

/// Parameters passed to spatial hash compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpatialParams {
    /// Cell size.
    pub cell_size: f32,
    /// Hash table size.
    pub hash_table_size: u32,
    /// Grid origin.
    pub grid_origin: [f32; 3],
    /// Number of particles.
    pub particle_count: u32,
    /// Grid size.
    pub grid_size: [u32; 3],
    /// Padding.
    pub _padding: u32,
}

impl From<(&SpatialHashConfig, u32)> for GpuSpatialParams {
    fn from((config, particle_count): (&SpatialHashConfig, u32)) -> Self {
        Self {
            cell_size: config.cell_size,
            hash_table_size: config.hash_table_size,
            grid_origin: config.grid_origin.to_array(),
            particle_count,
            grid_size: config.grid_size.to_array(),
            _padding: 0,
        }
    }
}

/// CPU-side spatial hash grid for debugging and testing.
#[derive(Default, Debug)]
pub struct CpuSpatialHash {
    /// Hash table mapping hash -> list of particle indices.
    table: Vec<Vec<usize>>,
    /// Configuration.
    config: SpatialHashConfig,
}

impl CpuSpatialHash {
    /// Create a new CPU spatial hash.
    pub fn new(config: SpatialHashConfig) -> Self {
        let table = vec![Vec::new(); config.hash_table_size as usize];
        Self { table, config }
    }

    /// Clear the hash table.
    pub fn clear(&mut self) {
        for cell in &mut self.table {
            cell.clear();
        }
    }

    /// Build the hash table from particle positions.
    pub fn build(&mut self, positions: &[Vec3]) {
        self.clear();
        for (i, &pos) in positions.iter().enumerate() {
            let hash = self.config.position_to_hash(pos) as usize;
            self.table[hash].push(i);
        }
    }

    /// Get all neighbors of a particle within the smoothing radius.
    pub fn get_neighbors(&self, position: Vec3, smoothing_radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let cell = self.config.position_to_cell(position);
        let _radius_sq = smoothing_radius * smoothing_radius;

        for offset in SpatialHashConfig::neighbor_offsets() {
            let neighbor_cell = cell + *offset;
            if self.config.is_valid_cell(neighbor_cell) {
                let hash = self.config.cell_to_hash(neighbor_cell) as usize;
                neighbors.extend(&self.table[hash]);
            }
        }

        neighbors
    }

    /// Get all neighbors with distance filtering.
    pub fn get_neighbors_filtered(
        &self,
        position: Vec3,
        positions: &[Vec3],
        smoothing_radius: f32,
    ) -> Vec<(usize, f32)> {
        let mut neighbors = Vec::new();
        let cell = self.config.position_to_cell(position);
        let radius_sq = smoothing_radius * smoothing_radius;

        for offset in SpatialHashConfig::neighbor_offsets() {
            let neighbor_cell = cell + *offset;
            if self.config.is_valid_cell(neighbor_cell) {
                let hash = self.config.cell_to_hash(neighbor_cell) as usize;
                for &idx in &self.table[hash] {
                    let dist_sq = (positions[idx] - position).length_squared();
                    if dist_sq < radius_sq {
                        neighbors.push((idx, dist_sq.sqrt()));
                    }
                }
            }
        }

        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash_config() {
        let config = SpatialHashConfig::for_domain(
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(10.0, 10.0, 10.0),
            0.5,
        );

        assert_eq!(config.grid_size, UVec3::new(40, 40, 40));
    }

    #[test]
    fn test_position_to_cell() {
        let config = SpatialHashConfig {
            cell_size: 1.0,
            grid_origin: Vec3::ZERO,
            ..default()
        };

        assert_eq!(config.position_to_cell(Vec3::new(0.5, 0.5, 0.5)), IVec3::ZERO);
        assert_eq!(config.position_to_cell(Vec3::new(1.5, 0.5, 0.5)), IVec3::new(1, 0, 0));
    }

    #[test]
    fn test_cpu_spatial_hash() {
        let config = SpatialHashConfig {
            cell_size: 1.0,
            grid_origin: Vec3::new(-5.0, -5.0, -5.0),
            grid_size: UVec3::splat(10),
            hash_table_size: 1000,
            ..default()
        };

        let mut hash = CpuSpatialHash::new(config);
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(5.0, 5.0, 5.0),
        ];

        hash.build(&positions);

        // First two particles should be in the same cell
        let neighbors = hash.get_neighbors(Vec3::new(0.0, 0.0, 0.0), 0.5);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
    }
}
