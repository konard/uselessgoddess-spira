# Исследование: SOTA GPU-симуляция жидкости для реального времени

## Содержание

1. [Введение](#введение)
2. [Обзор методов симуляции жидкости](#обзор-методов-симуляции-жидкости)
   - [Лагранжевы методы (частичные)](#лагранжевы-методы-частичные)
   - [Эйлеровы методы (сеточные)](#эйлеровы-методы-сеточные)
   - [Гибридные методы](#гибридные-методы)
3. [Детальный анализ методов](#детальный-анализ-методов)
   - [SPH (Smoothed Particle Hydrodynamics)](#sph-smoothed-particle-hydrodynamics)
   - [Position Based Fluids (PBF)](#position-based-fluids-pbf)
   - [MLS-MPM (Material Point Method)](#mls-mpm-material-point-method)
   - [FLIP/PIC/APIC](#flippicapic)
   - [Lattice Boltzmann Method (LBM)](#lattice-boltzmann-method-lbm)
4. [GPU-оптимизации](#gpu-оптимизации)
5. [Рендеринг жидкости](#рендеринг-жидкости)
6. [Существующие реализации для Rust/Bevy](#существующие-реализации-для-rustbevy)
7. [Нейросетевые подходы](#нейросетевые-подходы)
8. [Сравнительная таблица методов](#сравнительная-таблица-методов)
9. [Рекомендации для Bevy](#рекомендации-для-bevy)
10. [Источники](#источники)

---

## Введение

Данный документ представляет собой всестороннее исследование современных (state-of-the-art) методов GPU-симуляции жидкости для применения в игровых движках реального времени. Особое внимание уделено технологиям, совместимым с экосистемой Rust и игровым движком Bevy.

**Ключевые требования для realtime-симуляции:**
- Стабильные 30-60+ FPS
- Поддержка от 10,000 до 1,000,000+ частиц
- Визуально правдоподобное поведение (не обязательно физически точное)
- Низкая задержка для интерактивности

---

## Обзор методов симуляции жидкости

### Лагранжевы методы (частичные)

Частицы движутся вместе с жидкостью, неся свои физические свойства.

| Метод | Описание | Realtime-пригодность |
|-------|----------|---------------------|
| **SPH** | Сглаженная гидродинамика частиц | ⭐⭐⭐ |
| **PBF** | Позиционные ограничения | ⭐⭐⭐⭐⭐ |
| **MLS-MPM** | Гибрид частиц и сетки | ⭐⭐⭐⭐ |

### Эйлеровы методы (сеточные)

Фиксированная сетка, через которую протекает жидкость.

| Метод | Описание | Realtime-пригодность |
|-------|----------|---------------------|
| **LBM** | Решёточный метод Больцмана | ⭐⭐⭐⭐ |
| **Stable Fluids** | Полунеявный метод | ⭐⭐⭐ |

### Гибридные методы

Комбинируют преимущества обоих подходов.

| Метод | Описание | Realtime-пригодность |
|-------|----------|---------------------|
| **FLIP/PIC** | Частицы + сетка для давления | ⭐⭐⭐ |
| **APIC** | Улучшенный FLIP | ⭐⭐⭐⭐ |

---

## Детальный анализ методов

### SPH (Smoothed Particle Hydrodynamics)

**Принцип работы:**
SPH представляет жидкость как набор частиц, где физические величины (плотность, давление, скорость) интерполируются с использованием сглаживающих ядер (kernel functions).

**Варианты SPH:**

#### WCSPH (Weakly Compressible SPH)
- **Преимущества:** Простота реализации, хорошая параллелизация
- **Недостатки:** Требует малых временных шагов, видимое сжатие
- **Производительность GPU:** ~500,000 частиц при 30 FPS на RTX 3070

#### PCISPH (Predictive-Corrective Incompressible SPH)
- **Преимущества:** Лучшая несжимаемость
- **Недостатки:** Итеративный, медленнее WCSPH
- **Производительность:** 3-5 итераций для сходимости

#### IISPH (Implicit Incompressible SPH)
- **Преимущества:** Большие временные шаги, хорошая несжимаемость
- **Недостатки:** Сложнее реализация, требует линейный решатель
- **Производительность:** До 160 миллионов частиц (не realtime), deviation < 0.1%

#### DFSPH (Divergence-Free SPH)
- **Преимущества:** Самый быстрый из несжимаемых SPH, очень стабильный
- **Недостатки:** Сложность реализации
- **Производительность:** Значительно быстрее IISPH при миллионах частиц

**Ключевое узкое место SPH:** Поиск соседей (neighbor search) занимает до 50-70% времени симуляции.

**Современные оптимизации (2024-2025):**

1. **Dynamic Parallelism (CUDA):**
   - Ускорение 1.5x-3.0x по сравнению с CPU-GPU методами
   - Динамический запуск ядер с GPU
   - [Источник: ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2590123025028634)

2. **Mixed-Precision Computing:**
   - FP16 для поиска соседей, FP32 для вычислений
   - Ускорение до 1000x по сравнению с CPU
   - [Источник: ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0955799724000353)

3. **Count Sort + Parallel Prefix Scan:**
   - O(n) вместо O(n²) для поиска соседей
   - Throughput до 168,600 частиц/мс
   - [Источник: MDPI 2024](https://www.mdpi.com/2076-3417/15/17/9706)

---

### Position Based Fluids (PBF)

**Принцип работы:**
PBF формулирует симуляцию жидкости как задачу удовлетворения позиционных ограничений постоянной плотности. Метод наследует стабильность Position Based Dynamics (PBD).

**Ключевые преимущества для realtime:**
- ✅ Безусловная стабильность при больших временных шагах
- ✅ Отличная параллелизация на GPU
- ✅ Простота интеграции с другими PBD-объектами (ткань, верёвки)
- ✅ Контролируемое качество через число итераций

**Алгоритм:**
```
for each substep:
    apply_forces()
    predict_positions()

    for i in 0..solver_iterations:
        find_neighbors()
        calculate_lambda()  # множитель Лагранжа
        calculate_delta_p()  # коррекция позиции
        apply_delta_p()

    update_velocities()
    apply_viscosity()
    apply_vorticity_confinement()
```

**Производительность:**

| GPU | Частицы | FPS | Итерации |
|-----|---------|-----|----------|
| GTX 1050 | 32,000 | ~30 | 4 |
| Intel Iris Xe | 50,000 | ~50 | 4 |
| Intel Iris Xe | 100,000 | ~25 | 4 |
| RTX 3070 | 200,000 | ~60 | 4 |

**Расширения:**

#### Adaptive PBF (2016)
- Адаптивное число итераций на основе локальной ошибки
- Значительное ускорение при сохранении качества
- [Источник: arXiv](https://arxiv.org/abs/1608.04721)

#### XPBD (Extended Position Based Dynamics)
- Физически корректные параметры упругости
- Независимость от временного шага
- Лучшая сходимость
- [Источник: NVIDIA Research](https://matthias-research.github.io/pages/publications/PBDTutorial2017-slides-1.pdf)

**Реализации:**
- [PBF-CUDA](https://github.com/naeioi/PBF-CUDA) - CUDA реализация
- [WebGL PBF](https://github.com/xuxmin/pbf) - WebGL реализация
- [PositionBasedDynamics](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics) - C++ библиотека

---

### MLS-MPM (Material Point Method)

**Принцип работы:**
MPM комбинирует частицы (для переноса материала) и сетку (для решения уравнений). MLS-MPM (Moving Least Squares MPM) упрощает transfer между частицами и сеткой.

**Ключевое преимущество:** Не требует поиска соседей! Информация передаётся через сетку.

**Алгоритм:**
```
for each timestep:
    # Particle to Grid (P2G)
    scatter_particle_data_to_grid()

    # Grid operations
    apply_gravity()
    solve_grid_velocities()
    apply_boundary_conditions()

    # Grid to Particle (G2P)
    gather_grid_data_to_particles()
    advect_particles()
```

**Производительность (2024-2025):**

| Реализация | GPU | Частицы | FPS |
|------------|-----|---------|-----|
| WebGPU MLS-MPM | iGPU | 100,000 | 60 |
| WebGPU MLS-MPM | Discrete | 300,000 | 60 |
| Taichi MPM | RTX 3080 | 1,000,000 | ~20 |
| Multi-GPU MPM | 8x V100 | 100,000,000 | <1 |

**Современные улучшения:**

#### APS-MPM (Affine Projection Stabilizer, 2025)
- XPBD-стиль стабилизация
- 2.9-3.3x ускорение над традиционным MLS-MPM
- [Источник: Springer 2025](https://link.springer.com/article/10.1007/s00371-025-03953-2)

#### PB-MPM (Position-Based MPM, 2024)
- Формулировка как constraint solving
- Бо́льшие временные шаги
- Лучшая стабильность

**Реализации:**
- [Taichi MPM](https://github.com/yuanming-hu/taichi_mpm) - Python/Taichi (SIGGRAPH 2018)
- [GPUMPM](https://github.com/kuiwuchn/GPUMPM) - CUDA реализация

---

### FLIP/PIC/APIC

**FLIP (Fluid Implicit Particle):**
Гибридный метод, использующий частицы для переноса и сетку для решения давления.

**PIC (Particle-In-Cell):**
Предшественник FLIP, более диссипативный.

**APIC (Affine PIC, SIGGRAPH 2015):**
Сохраняет угловой момент, меньше диссипации.

**Производительность с GVDB (Sparse Volumes):**

| Сцена | Частицы | Сетка | Время/кадр |
|-------|---------|-------|------------|
| Dam Break | 8M | 300×200×200 | 0.8s |
| Column | 29M | 450×300×300 | 2.4s |

- До 10x ускорение по сравнению с CPU FLIP
- Виртуально неограниченный домен симуляции
- [Источник: CSAIL MIT](https://people.csail.mit.edu/kuiwu/gvdb_sim.html)

**Реализации:**
- [Blub](https://github.com/Wumpf/blub) - Rust/wgpu (APIC, работа в процессе)
- [GridFluidSim3D](https://github.com/rlguy/GridFluidSim3D) - C++ PIC/FLIP

---

### Lattice Boltzmann Method (LBM)

**Принцип работы:**
Вместо решения уравнений Навье-Стокса напрямую, LBM симулирует статистическое поведение частиц на решётке.

**Преимущества:**
- ✅ Идеально параллелизуется на GPU
- ✅ Простая обработка сложных границ
- ✅ Естественная работа с многофазными потоками
- ✅ Физически более точный (DNS)

**Недостатки:**
- ❌ Высокие требования к памяти
- ❌ Сложнее моделировать свободную поверхность
- ❌ Фиксированная сетка (проблемы с масштабом)

**Производительность FluidX3D (2024):**
- До 0.5 миллиарда обновлений узлов/сек
- Поддержка до 2^64 ячеек сетки
- OpenCL для кросс-платформенности
- [Источник: GitHub FluidX3D](https://github.com/ProjectPhysX/FluidX3D)

**Коммерческие решения:**
- **Pacefish® (SimScale)** - облачный LBM решатель
- Сокращение времени симуляции с дней до часов

---

## GPU-оптимизации

### Поиск соседей (Neighbor Search)

Критическое узкое место для SPH и PBF.

#### Методы:

**1. Uniform Grid / Spatial Hashing:**
```
hash = floor(position / cell_size)
cell_id = hash.x + hash.y * grid_width + hash.z * grid_width * grid_height
```
- O(n) сложность
- Эффективно для равномерных распределений

**2. Count Sort + Parallel Prefix Scan:**
- Фьюжн алгоритмов для O(n) поиска
- [Источник: MDPI 2024](https://www.mdpi.com/2076-3417/15/17/9706)

**3. Locally Perfect Hashing:**
- Избегание коллизий в окрестности частицы
- O(1) доступ без проверки коллизий
- [Источник: haug.codes](https://haug.codes/blog/locally-perfect-hashing/)

### Memory Layout

**Structure of Arrays (SoA) vs Array of Structures (AoS):**

```rust
// AoS (плохо для GPU)
struct Particle {
    position: Vec3,
    velocity: Vec3,
    density: f32,
}
let particles: Vec<Particle>;

// SoA (хорошо для GPU)
struct Particles {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    densities: Vec<f32>,
}
```

SoA даёт 30-45% улучшение производительности на GPU.

### Оптимизации compute shaders

1. **Workgroup size:** Оптимум обычно 64-256 (128 для Vulkan SPH)
2. **Shared memory:** Использовать для данных соседей
3. **Coalesced memory access:** Последовательный доступ к памяти
4. **Kernel fusion:** Объединение мелких ядер для уменьшения overhead

### Vulkan vs OpenGL vs CUDA

| API | Преимущества | Производительность |
|-----|-------------|-------------------|
| **CUDA** | Лучшая оптимизация, богатые библиотеки | ⭐⭐⭐⭐⭐ |
| **Vulkan** | Кросс-платформенность, низкий overhead | ⭐⭐⭐⭐ |
| **WebGPU/wgpu** | Rust-native, web + native | ⭐⭐⭐⭐ |
| **OpenGL Compute** | Простота, широкая поддержка | ⭐⭐⭐ |

Для Bevy рекомендуется **wgpu** (WebGPU) как нативное решение.

---

## Рендеринг жидкости

### Screen-Space Fluid Rendering

**Алгоритм (GDC 2010, Simon Green, NVIDIA):**

1. **Depth Pass:** Рендер частиц как сферы → depth buffer
2. **Thickness Pass:** Накопление толщины (no depth test, additive)
3. **Smoothing Pass:** Bilateral/Curvature Flow фильтрация depth
4. **Normal Reconstruction:** Из сглаженного depth buffer
5. **Shading:** Fresnel, refraction, reflection

**Оптимизации:**

#### Narrow-Band Rendering
- Фильтрация только у границы жидкости
- Значительное ускорение
- [Источник: ResearchGate 2022](https://www.researchgate.net/publication/359849957_Narrow-Band_Screen-Space_Fluid_Rendering)

#### Anisotropic Splatting
- Эллипсоидальные частицы вместо сфер
- Лучшее качество поверхности
- [Источник: ResearchGate 2022](https://www.researchgate.net/publication/366538410_Anisotropic_screen_space_rendering_for_particle-based_fluid_simulation)

### Марширующие кубы (Marching Cubes)

- Извлечение полигональной сетки из частиц
- Высокое качество, но дорого для realtime
- Использовать для финального рендера или pre-baked

---

## Существующие реализации для Rust/Bevy

### wgpu-based проекты

| Проект | Метод | Статус | GPU API |
|--------|-------|--------|---------|
| [Blub](https://github.com/Wumpf/blub) | APIC/FLIP | WIP | wgpu |
| [bevy-sph](https://github.com/AOS55/bevy-sph) | SPH | Experimental | Bevy |
| [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) | MLS-MPM | Demo | WebGPU |

### Физические движки для Bevy

| Движок | Тип | Fluids |
|--------|-----|--------|
| [Avian](https://github.com/Jondolf/avian) | XPBD | Нет (roadmap) |
| [Rapier](https://rapier.rs/) | TGS Soft | Нет |

### Bevy Hanabi (GPU Particles)

- Compute shader частицы
- Хорошая основа для fluid rendering
- [Источник: Bevy Hanabi](https://github.com/djeedai/bevy_hanabi)

---

## Нейросетевые подходы

### Physics-Informed Neural Networks (PINNs)

**Принцип:** Встраивание физических законов (Navier-Stokes) в loss function нейросети.

**Преимущества:**
- Не требуют большого датасета
- Могут обобщаться на новые граничные условия

**Недостатки для realtime:**
- Медленный inference (относительно традиционных методов)
- Сложность обучения для 3D
- Масштабируемость

**NVIDIA PhysicsNeMo:**
- GPU-ускоренный фреймворк для physics AI
- Поддержка PINNs, Neural Operators, GNNs
- [Источник: NVIDIA Modulus](https://developer.nvidia.com/modulus)

### Neural Operators

**Fourier Neural Operator (FNO):**
- Учится отображать начальное состояние в конечное
- Очень быстрый inference
- Ограничен обучающим распределением

### Гибридные подходы

Комбинация традиционной симуляции с ML для:
- Ускорения решателей
- Улучшения субсеточного моделирования
- Предсказания долгосрочной динамики

**Вывод:** Для realtime игр в 2024-2025 традиционные методы (PBF, MLS-MPM) остаются предпочтительными. Нейросети перспективны для предвычисленных эффектов.

---

## Сравнительная таблица методов

| Метод | Скорость | Качество | Сложность | GPU-пригодность | Память | Рекомендация |
|-------|----------|----------|-----------|-----------------|--------|--------------|
| **PBF** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Лучший выбор** |
| **DFSPH** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Если нужна точность |
| **MLS-MPM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Для вязких/гранулярных |
| **WCSPH** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Простой старт |
| **APIC/FLIP** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Крупномасштабные |
| **LBM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Если память не проблема |

---

## Рекомендации для Bevy

### Архитектура решения

```
┌─────────────────────────────────────────────────────────┐
│                    Bevy ECS                             │
├─────────────────────────────────────────────────────────┤
│  FluidSimulationPlugin                                  │
│  ├── SimulationSystem (compute shaders via wgpu)        │
│  ├── RenderingSystem (screen-space fluid)               │
│  └── InteractionSystem (Avian physics coupling)         │
├─────────────────────────────────────────────────────────┤
│                    wgpu Compute                         │
│  ├── neighbor_search.wgsl                               │
│  ├── density_constraint.wgsl (PBF)                      │
│  ├── velocity_update.wgsl                               │
│  └── rendering_passes.wgsl                              │
└─────────────────────────────────────────────────────────┘
```

### Рекомендуемый путь реализации

#### Фаза 1: Базовая PBF симуляция
1. Реализовать spatial hashing на GPU (wgpu compute)
2. Базовый PBF solver (3-4 итерации)
3. Простой рендеринг (точки/сферы)
4. **Target:** 50,000 частиц @ 60 FPS

#### Фаза 2: Оптимизации
1. Count Sort + Prefix Scan для neighbor search
2. SoA memory layout
3. Adaptive iteration count
4. **Target:** 200,000 частиц @ 60 FPS

#### Фаза 3: Качественный рендеринг
1. Screen-space fluid rendering
2. Curvature flow smoothing
3. Refraction/reflection
4. **Target:** Визуально привлекательная жидкость

#### Фаза 4: Интеграция
1. Coupling с Avian physics (rigid bodies)
2. Two-way interaction
3. Boundary handling

### Технический стек

| Компонент | Технология |
|-----------|------------|
| GPU API | wgpu (WebGPU) |
| Shaders | WGSL |
| ECS | Bevy |
| Physics coupling | Avian |
| Linear algebra | glam |

### Ключевые структуры данных

```rust
// GPU-side (SoA layout)
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct ParticleData {
    positions: [[f32; 4]; MAX_PARTICLES],   // vec4 for alignment
    velocities: [[f32; 4]; MAX_PARTICLES],
    predicted_positions: [[f32; 4]; MAX_PARTICLES],
    lambdas: [f32; MAX_PARTICLES],           // density constraint multipliers
}

// Spatial hash grid
#[repr(C)]
struct GridData {
    cell_start: [u32; NUM_CELLS],           // start index in sorted array
    cell_count: [u32; NUM_CELLS],           // particles per cell
    particle_indices: [u32; MAX_PARTICLES], // sorted by cell
}
```

### Параметры для настройки

```rust
pub struct FluidParams {
    pub rest_density: f32,        // 1000.0 kg/m³ для воды
    pub particle_radius: f32,     // h/4 обычно
    pub smoothing_radius: f32,    // h, влияет на качество/скорость
    pub viscosity: f32,           // XSPH viscosity coefficient
    pub surface_tension: f32,     // искусственное поверхностное натяжение
    pub vorticity_epsilon: f32,   // vorticity confinement
    pub solver_iterations: u32,   // 3-6 обычно достаточно
    pub substeps: u32,            // 1-4 substeps per frame
}
```

---

## Источники

### Академические работы

1. Macklin, M., Müller, M. (2013). **Position Based Fluids**. ACM TOG. [PDF](https://mmacklin.com/pbf_sig_preprint.pdf)

2. Bender, J., Koschier, D. (2017). **Divergence-Free SPH for Incompressible and Viscous Fluids**. [PDF](https://discovery.ucl.ac.uk/id/eprint/10056699/1/BK17.pdf)

3. Hu, Y., et al. (2018). **A Moving Least Squares Material Point Method**. ACM TOG (SIGGRAPH). [GitHub](https://github.com/yuanming-hu/taichi_mpm)

4. Wu, K., et al. (2018). **Fast Fluid Simulations with Sparse Volumes on the GPU**. [Project Page](https://people.csail.mit.edu/kuiwu/gvdb_sim.html)

5. Jiang, C., et al. (2015). **The Affine Particle-In-Cell Method**. ACM TOG (SIGGRAPH).

### Современные исследования (2024-2025)

6. **Enhancement of GPU-accelerated SPH with Dynamic Parallelism** (2025). [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590123025028634)

7. **GPU accelerated mixed-precision SPH framework** (2024). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0955799724000353)

8. **Moving Towards Large-Scale Particle Based Fluid Simulation in Unity 3D** (2024). [MDPI](https://www.mdpi.com/2076-3417/15/17/9706)

9. **Enhanced MPM with Affine Projection Stabilizer** (2025). [Springer](https://link.springer.com/article/10.1007/s00371-025-03953-2)

### Ресурсы и реализации

10. **SPlisHSPlasH** - Open-source SPH library. [Website](https://splishsplash.physics-simulation.org/) | [GitHub](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)

11. **FluidX3D** - Fast LBM solver. [GitHub](https://github.com/ProjectPhysX/FluidX3D)

12. **Blub** - Rust/wgpu fluid simulation. [GitHub](https://github.com/Wumpf/blub)

13. **bevy-sph** - Bevy SPH implementation. [GitHub](https://github.com/AOS55/bevy-sph)

14. **Avian Physics** - Bevy physics engine. [GitHub](https://github.com/Jondolf/avian)

### Туториалы и документация

15. **Position Based Simulation Tutorial** (SIGGRAPH 2017). [Slides](https://matthias-research.github.io/pages/publications/PBDTutorial2017-slides-1.pdf)

16. **Ten Minute Physics - Spatial Hashing**. [PDF](https://matthias-research.github.io/pages/tenMinutePhysics/11-hashing.pdf)

17. **Learn wgpu**. [Tutorial](https://sotrh.github.io/learn-wgpu/)

18. **Tutorial: Making a physics engine with Bevy**. [Blog](https://johanhelsing.studio/posts/bevy-xpbd/)

### SIGGRAPH 2024-2025

19. **SIGGRAPH 2024 Papers**. [List](https://kesen.realtimerendering.com/sig2024.html)

20. **SIGGRAPH 2025 Papers**. [List](https://www.realtimerendering.com/kesen/sig2025.html)

---

## Заключение

Для разработки библиотеки GPU-симуляции жидкости для Bevy рекомендуется:

1. **Начать с Position Based Fluids (PBF)** - лучший баланс скорости, качества и простоты реализации для realtime приложений.

2. **Использовать wgpu** как GPU API для кросс-платформенности и нативной интеграции с Bevy.

3. **Сфокусироваться на оптимизации neighbor search** - это основное узкое место производительности.

4. **Реализовать screen-space fluid rendering** для визуально привлекательного результата без дорогого mesh extraction.

5. **Рассмотреть MLS-MPM как альтернативу** для симуляции вязких жидкостей (мёд, лава) или гранулярных материалов (песок, снег).

6. **Планировать интеграцию с Avian** для взаимодействия с rigid bodies.

Ожидаемая производительность на современных GPU (RTX 3070+):
- **100,000-200,000 частиц @ 60 FPS** - достижимо с оптимизированным PBF
- **500,000+ частиц @ 30 FPS** - возможно с агрессивными оптимизациями

---

*Документ подготовлен для проекта Spira. Дата: Январь 2025.*
