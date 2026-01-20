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
9. [Симуляция в сложных геометриях](#симуляция-в-сложных-геометриях)
   - [Анатомические структуры и верхнечелюстная пазуха](#анатомические-структуры-и-верхнечелюстная-пазуха)
   - [Представление CT-данных: меш vs воксели](#представление-ct-данных-меш-vs-воксели)
   - [Рекомендации по выбору метода для медицинских симуляций](#рекомендации-по-выбору-метода-для-медицинских-симуляций)
10. [Взаимодействие с мягкими тканями](#взаимодействие-с-мягкими-тканями)
11. [Coupling с физическими движками (Avian/Rapier)](#coupling-с-физическими-движками-avianrapier)
12. [Рекомендации для Bevy](#рекомендации-для-bevy)
13. [Источники](#источники)

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

## Симуляция в сложных геометриях

### Анатомические структуры и верхнечелюстная пазуха

Симуляция жидкости в анатомических структурах (например, верхнечелюстной пазухе носа) представляет особые вызовы из-за сложной, неправильной геометрии. Рассмотрим применимость различных методов.

#### Исследования симуляции верхнечелюстной пазухи

Существующие CFD-исследования верхнечелюстной пазухи показывают:

1. **Особенности анатомии:**
   - В нормальном состоянии верхнечелюстная пазуха практически не вентилируется активно
   - Скорость воздушного потока в пазухах (в среднем 0.062 м/с) значительно ниже, чем в среднем носовом ходе (3.26 м/с)
   - Температура в пазухе остаётся практически постоянной (~34°C)
   - [Источник: ResearchGate - CFD-simulation of the air flows in the maxillary sinus](https://www.researchgate.net/publication/319139580_CFD-simulation_of_the_air_flows_in_the_maxillary_sinus)

2. **Клинические применения (2025):**
   - CFD используется для оптимизации доставки лекарств в носовую полость
   - Пред- и постоперационные сканы определяют давление, линии тока и скорость
   - Исследуется влияние хирургии на изменения в синоназальной полости
   - [Источник: SAGE Journals 2025](https://journals.sagepub.com/doi/10.1177/01455613251335109)

#### Сравнение методов для сложных геометрий

| Метод | Работа со сложной геометрией | Преимущества | Недостатки |
|-------|------------------------------|--------------|------------|
| **LBM** | ⭐⭐⭐⭐⭐ | Тривиальная обработка границ, воксельное представление | Высокие требования к памяти |
| **SPH/PBF** | ⭐⭐⭐⭐ | Безсеточный, хорошо для свободных поверхностей | Нужны граничные частицы или SDF |
| **MLS-MPM** | ⭐⭐⭐⭐ | Гибрид частиц и сетки | Сетка может быть проблемой для сложных форм |
| **FLIP/APIC** | ⭐⭐⭐ | Хорошо для больших объёмов | Требует адаптивной сетки |

#### LBM как оптимальный выбор для анатомических симуляций

**Lattice Boltzmann Method особенно хорошо подходит для верхнечелюстной пазухи:**

1. **Простота обработки сложных границ:**
   - Прямая работа с вокселизированной геометрией из CT
   - Не требует генерации качественного меша
   - [Источник: Springer - Numerical Simulation of Nasal Cavity Flow Based on LBM](https://link.springer.com/chapter/10.1007/978-3-642-14243-7_63)

2. **Преимущества над методами Навье-Стокса:**
   - Быстрая генерация сетки
   - Простой, гранулярный алгоритм для эффективной параллелизации
   - Высокая гибкость для сложных граничных условий
   - Эффективен для предсказания потоков в рамках computer-aided rhino-surgery
   - [Источник: ScienceDirect - Simulation of nasal flow by LBM](https://www.sciencedirect.com/science/article/abs/pii/S0010482506000965)

3. **Автоматизация с машинным обучением (2021-2025):**
   - Свёрточные нейросети для автоматической сегментации верхних дыхательных путей
   - Автоматическое определение входных/выходных областей для граничных условий
   - Точность до 99.5% по сравнению с ручной сегментацией
   - [Источник: Springer - Workflow for enhanced diagnostics in rhinology](https://link.springer.com/article/10.1007/s11517-021-02446-3)

4. **Гибридные подходы PINN-MRT (2025):**
   - Интеграция Multi-Relaxation-Time LBM с Physics-Informed Neural Networks
   - Улучшенная стабильность для сложных потоков при высоких числах Рейнольдса
   - [Источник: MDPI - Hybrid PINNs with MRT-LBM](https://www.mdpi.com/2227-7390/13/22/3712)

---

### Представление CT-данных: меш vs воксели

Для работы с CT-данными существует два основных подхода к представлению геометрии. Выбор зависит от метода симуляции и требований к точности.

#### Вокселизированное представление

**Преимущества:**
- ✅ Прямое использование CT-данных без предобработки
- ✅ Идеально для LBM (нативный формат)
- ✅ Быстрая настройка симуляции
- ✅ Подходит для гетерогенных объектов с нерегулярным распределением материалов
- ✅ Эффективен для GPU (прямое отображение вокселей на 3D-текстуры)

**Недостатки:**
- ❌ Ступенчатые артефакты (staircase artifacts) из-за анизотропных размеров вокселей
- ❌ Требует высокого разрешения для точных границ
- ❌ Больше памяти при высоком разрешении

**Применение в GPU-симуляции (NVIDIA GPU Gems 3):**
```
Препятствия представляются как inside-outside вокселизация.
Каждая ячейка сетки хранит скалярные (давление, температура)
и векторные (скорость) величины.

На границах solid-fluid применяется free-slip boundary condition:
скорости жидкости и твёрдого тела одинаковы в направлении
нормали к границе.
```
- [Источник: NVIDIA Developer - Real-Time 3D Fluids](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)

#### Меш-представление (полигональное)

**Преимущества:**
- ✅ Гладкие границы, высокая точность поверхности
- ✅ Меньше памяти при умеренной сложности
- ✅ Лучше для SPH/PBF с граничными частицами
- ✅ Стандартный формат для визуализации

**Недостатки:**
- ❌ Требует качественной генерации меша из CT (сложный процесс)
- ❌ Image-based meshing всё ещё активно исследуется
- ❌ Сложность для топологически сложных структур
- [Источник: arXiv - Image-To-Mesh Conversion for Biomedical Simulations](https://arxiv.org/html/2402.18596v1)

#### Сравнительное исследование (2025)

Недавнее исследование FE-моделей головы сравнило surface-based и voxel-based подходы:

| Аспект | Surface-based | Voxel-based |
|--------|---------------|-------------|
| Геометрическая точность | Высокая (conforming meshes) | Средняя (staircase) |
| Захват мелких деталей | Отличный | Зависит от разрешения |
| Время подготовки | Высокое | Низкое |
| Качество элементов | Контролируемое | Автоматическое |

- [Источник: Springer 2025 - Surface vs Voxel-based FE Head Models](https://link.springer.com/article/10.1007/s10237-025-01940-z)

#### Рекомендация для верхнечелюстной пазухи

**Для приоритетной задачи симуляции верхнечелюстной пазухи рекомендуется:**

1. **Начать с вокселей + LBM:**
   - Минимальная предобработка CT-данных
   - Быстрый старт и итерация
   - Отличная GPU-параллелизация
   - Естественная работа со сложной анатомией

2. **Использовать Signed Distance Field (SDF) для SPH/PBF:**
   - Если нужен Лагранжев подход
   - SDF генерируется из воксельных данных
   - Граничные условия через SPH boundary particles

3. **Меш — для финального рендеринга:**
   - Marching Cubes для извлечения поверхности из вокселей
   - Или при необходимости интеграции с другими системами

---

### Рекомендации по выбору метода для медицинских симуляций

#### Для симуляции жидкости в верхнечелюстной пазухе

| Критерий | LBM | SPH/PBF | Рекомендация |
|----------|-----|---------|--------------|
| Работа с CT-данными | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | LBM |
| Сложная геометрия | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | LBM |
| Realtime на GPU | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Оба хороши |
| Свободная поверхность | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | SPH/PBF |
| Физическая точность | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | LBM |
| Простота реализации | ⭐⭐⭐⭐ | ⭐⭐⭐ | LBM |

**Итоговая рекомендация для верхнечелюстной пазухи:**

**LBM (Lattice Boltzmann Method)** — лучший выбор для:
- Точной симуляции воздушных/жидкостных потоков в анатомических структурах
- Прямой работы с вокселизированными CT-данными
- Клинически значимых результатов (используется в medical CFD)

**PBF/SPH** — хороший выбор если:
- Нужна симуляция свободной поверхности жидкости (не заполненная полость)
- Важна визуальная привлекательность для игрового процесса
- Планируется унифицированная симуляция с другими PBD-объектами

---

## Взаимодействие с мягкими тканями

Для будущей интеграции с мягкими тканями (например, слизистой оболочкой) необходимо рассмотреть методы fluid-structure interaction (FSI).

### SPH-based FSI Framework

**Современные разработки (2024-2025):**

1. **SPHaptics (ноябрь 2025):**
   - Унифицированный фреймворк для bidirectional haptic interaction
   - Связь rigid bodies, deformable objects и Lagrangian fluids
   - Two-way force coupling с smoothing для стабильности
   - VR-сценарии: перемешивание жидкости, манипуляция мягкими тканями
   - [Источник: arXiv - SPHaptics](https://arxiv.org/html/2511.15908v1)

2. **SPH для FSI с разрушением (январь 2024):**
   - Weakly compressible SPH для жидкости
   - Pseudo-spring-based SPH для структуры
   - δ-SPH для улучшенных расчётов давления
   - Моделирование повреждений и разрушения под гидродинамической нагрузкой
   - [Источник: ScienceDirect - SPH framework for FSI](https://www.sciencedirect.com/science/article/abs/pii/S0029801824000593)

3. **SPH-BVF для биофизических систем:**
   - Boundary Volume Fraction (BVF) для сложных moving boundaries
   - Применение: динамика клеток, морфогенез, симуляция кровотока в тканях
   - [Источник: PMC - ALE-SPH with BVF](https://pmc.ncbi.nlm.nih.gov/articles/PMC8143034/)

### LBM vs SPH для FSI

Сравнение для биомедицинских применений (на примере сердечных клапанов):

| Аспект | LBM + FEM | SPH + FEM |
|--------|-----------|-----------|
| Простота реализации | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Скорость вычислений | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Большие деформации | ⭐⭐⭐ (mesh distortion) | ⭐⭐⭐⭐⭐ |
| Точность FSI | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

> "Хотя LBM легко реализовать и быстро считает, Лагранжево поведение SPH делает его более подходящим для robust и accurate симуляций FSI. Эйлеровы методы страдают от численных проблем при больших деформациях твёрдых тел."
> — [Источник: HAL - SPH vs LBM for trans-aortic FSI](https://hal.science/hal-05313398v1/document)

### XPBD и унифицированные солверы

**Unified Position-Based Simulation (2024-2025):**

1. **XPBD (Extended Position Based Dynamics):**
   - Единый солвер для fluids, elastic solids, stiff solids
   - Независимость от timestep
   - [Источник: NVIDIA - XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)

2. **XPBI — расширение для неэластичных материалов (2024):**
   - Поддержка elastoplastic, viscoplastic, granular материалов
   - High-resolution realtime: снег, песок, пластилин
   - Одновременная связь с XPBD водой и тканью
   - [Источник: arXiv - XPBI](https://arxiv.org/html/2405.11694v2)

3. **Multi-layer XPBD Solver (2024):**
   - Multigrid-подобное ускорение сходимости
   - Efficient propagation of impulses в contact-rich сценариях
   - [Источник: Wiley - Multi-layer XPBD](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15186?af=R)

### Рекомендация для мягких тканей

**Для симуляции жидкости + мягкие ткани в Bevy:**

1. **Унифицированный PBD/XPBD подход:**
   - Мягкие ткани как PBD constraints
   - Жидкость как PBF (Position Based Fluids)
   - Единый solver loop
   - Естественное coupling через shared constraints

2. **SPH + FEM гибрид:**
   - SPH для жидкости
   - FEM для мягких тканей с Fung-type constitutive law
   - Two-way coupling через immersed boundary

---

## Coupling с физическими движками (Avian/Rapier)

### Текущее состояние в экосистеме Bevy

| Движок | Тип солвера | Fluid support | Two-way FSI |
|--------|-------------|---------------|-------------|
| **Avian** | TGS Soft (XPBD-based) | ❌ Нет (roadmap) | ❌ |
| **Rapier** | TGS Soft | ❌ Нет | ❌ |
| **bevy-sph** | SPH | ✅ Experimental | ❌ |

> "На момент написания следующие возможности ещё не интегрированы: CCD, articulations, soft bodies (cloth), и fluid simulation."
> — [Источник: Tainted Coders - Bevy Physics](https://taintedcoders.com/bevy/physics/avian)

### Архитектурные различия

**Avian:**
- ECS-native: компоненты и расчёты напрямую в Bevy ECS
- Глубокая интеграция с Bevy
- Фокус на эргономике и модульности

**Rapier:**
- Отдельное представление физики, проецируемое на Bevy ECS
- Более зрелый, может использоваться вне Bevy
- [Источник: Tainted Coders - Rapier](https://taintedcoders.com/bevy/physics/rapier)

### Стратегии coupling жидкости с rigid bodies

#### 1. Weak Coupling (простой подход)

```rust
// Псевдокод weak coupling
fn fluid_rigid_coupling_system(
    mut fluid_query: Query<&mut FluidParticle>,
    rigid_query: Query<(&Transform, &Velocity, &Collider)>,
) {
    for mut particle in fluid_query.iter_mut() {
        for (transform, velocity, collider) in rigid_query.iter() {
            if particle.collides_with(collider) {
                // Частица отражается от rigid body
                particle.velocity = reflect(particle.velocity, collider.normal());
                // Добавляем скорость rigid body
                particle.velocity += velocity.linear;
            }
        }
    }
}
```

**Характеристики:**
- Rigid bodies имеют постоянные скорости во время расчёта жидкости
- Простая реализация
- Подходит для большинства игровых сценариев

#### 2. Strong Coupling (реалистичный подход)

```rust
// Псевдокод strong coupling с обратной связью
fn strong_coupling_system(
    mut fluid_query: Query<&mut FluidParticle>,
    mut rigid_query: Query<(&Transform, &mut ExternalForce, &Collider)>,
) {
    let mut accumulated_forces: HashMap<Entity, Vec3> = HashMap::new();

    for particle in fluid_query.iter() {
        for (entity, transform, _, collider) in rigid_query.iter() {
            if let Some(contact) = particle.contact_with(collider) {
                // Сила на rigid body от давления жидкости
                let force = particle.pressure * contact.normal * particle.volume;
                accumulated_forces.entry(entity)
                    .or_default()
                    .add(force);
            }
        }
    }

    // Применяем силы к rigid bodies
    for (entity, _, mut external_force, _) in rigid_query.iter_mut() {
        if let Some(force) = accumulated_forces.get(&entity) {
            external_force.force += *force;
        }
    }
}
```

**Характеристики:**
- Учитывает ускорение rigid bodies во время взаимодействия
- Более физически корректные результаты
- Сложнее в реализации, дороже вычислительно

#### 3. DiffFR — дифференцируемое coupling (исследование)

**Differentiable SPH-based Fluid-Rigid Coupling (2023):**
- Обратное проектирование движения rigid objects в two-way FSI
- Работа с discontinuous contacts
- Gradient formulation для оптимизации
- [Источник: ACM TOG - DiffFR](https://dl.acm.org/doi/10.1145/3618318)

### Практическая реализация для Bevy

#### Buoyancy (плавучесть)

```rust
// Пример расчёта плавучести
pub fn calculate_buoyancy(
    fluid_density: f32,
    submerged_volume: f32,
    gravity: Vec3,
) -> Vec3 {
    // Архимедова сила
    fluid_density * submerged_volume * -gravity
}

pub fn buoyancy_system(
    fluid: Res<FluidSimulation>,
    mut bodies: Query<(&Transform, &Collider, &mut ExternalForce)>,
) {
    for (transform, collider, mut force) in bodies.iter_mut() {
        let submerged_volume = fluid.calculate_submerged_volume(
            transform,
            collider
        );

        if submerged_volume > 0.0 {
            force.force += calculate_buoyancy(
                fluid.rest_density,
                submerged_volume,
                Vec3::new(0.0, -9.81, 0.0),
            );

            // Drag force
            let relative_velocity = body_velocity - fluid.local_velocity(transform.translation);
            force.force -= fluid.drag_coefficient * relative_velocity * submerged_volume;
        }
    }
}
```

#### Рекомендуемый подход для интеграции

1. **Фаза 1: Одностороннее взаимодействие (fluid → rigid)**
   - Жидкость применяет силы к rigid bodies (buoyancy, drag)
   - Rigid bodies не влияют на жидкость напрямую
   - Простая реализация, хорошо для начала

2. **Фаза 2: Двустороннее взаимодействие**
   - Rigid bodies создают displacement в жидкости
   - Граничные частицы или SDF для представления rigid bodies
   - Coupling через Avian's ExternalForce компонент

3. **Фаза 3: Полная интеграция**
   - Unified solver (если возможно перейти на XPBD для всего)
   - Или tight coupling между отдельными солверами

### Существующие реализации coupling

1. **SPH_Project (CUDA):**
   - Large scale simulation с rigid-fluid coupling
   - PyBullet для rigid body dynamics
   - [Источник: GitHub - SPH_Project](https://github.com/jason-huang03/SPH_Project)

2. **Fluids (CPU, educational):**
   - Two-way rigid body interaction с SPH
   - [Источник: GitHub - fluids](https://github.com/paivett/fluids)

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

#### Фаза 4: Интеграция с Avian
1. Coupling с Avian physics (rigid bodies)
2. Two-way interaction
3. Boundary handling

#### Фаза 5: Медицинские симуляции (верхнечелюстная пазуха)
1. Добавить LBM солвер для точных медицинских расчётов
2. Реализовать загрузку CT-данных (воксели)
3. Автоматическая сегментация границ (SDF или voxel-based)
4. **Target:** Реалистичная симуляция потоков в анатомических структурах

#### Фаза 6: Мягкие ткани (будущее)
1. Интеграция PBD/XPBD для деформируемых объектов
2. FSI coupling между жидкостью и мягкими тканями
3. Унифицированный solver для всех типов объектов

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

### Медицинские CFD и анатомические симуляции

21. **CFD-simulation of air flows in maxillary sinus**. [ResearchGate](https://www.researchgate.net/publication/319139580_CFD-simulation_of_the_air_flows_in_the_maxillary_sinus)

22. **Computational Modeling of Nasal Cavity Aerodynamics** (2025). [SAGE Journals](https://journals.sagepub.com/doi/10.1177/01455613251335109)

23. **Numerical Simulation of Nasal Cavity Flow Based on LBM**. [Springer](https://link.springer.com/chapter/10.1007/978-3-642-14243-7_63)

24. **Simulation of nasal flow by LBM**. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482506000965)

25. **Workflow for enhanced diagnostics in rhinology** (2021). [Springer](https://link.springer.com/article/10.1007/s11517-021-02446-3)

26. **Hybrid PINNs with MRT-LBM** (2025). [MDPI](https://www.mdpi.com/2227-7390/13/22/3712)

### Представление геометрии (Mesh vs Voxel)

27. **Real-Time 3D Fluids on GPU** (NVIDIA GPU Gems 3). [Developer](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)

28. **Surface vs Voxel-based FE Head Models** (2025). [Springer](https://link.springer.com/article/10.1007/s10237-025-01940-z)

29. **Image-To-Mesh Conversion for Biomedical Simulations** (2024). [arXiv](https://arxiv.org/html/2402.18596v1)

### Fluid-Structure Interaction

30. **SPHaptics: Bidirectional Haptic Interaction for FSI** (2025). [arXiv](https://arxiv.org/html/2511.15908v1)

31. **SPH framework for FSI with fracturing** (2024). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801824000593)

32. **ALE-SPH with Boundary Volume Fraction** (2021). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8143034/)

33. **SPH vs LBM for trans-aortic FSI**. [HAL](https://hal.science/hal-05313398v1/document)

### XPBD и унифицированные солверы

34. **XPBD: Position-Based Simulation of Compliant Dynamics**. [NVIDIA](https://matthias-research.github.io/pages/publications/XPBD.pdf)

35. **XPBI: PBD with Smoothing Kernels for Continuum Inelasticity** (2024). [arXiv](https://arxiv.org/html/2405.11694v2)

36. **Multi-layer XPBD Solver** (2024). [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15186?af=R)

### Rigid-Fluid Coupling

37. **DiffFR: Differentiable SPH-based Fluid-Rigid Coupling** (2023). [ACM TOG](https://dl.acm.org/doi/10.1145/3618318)

38. **SPH_Project** - Large-scale rigid-fluid coupling. [GitHub](https://github.com/jason-huang03/SPH_Project)

39. **Fluids** - Two-way rigid body interaction. [GitHub](https://github.com/paivett/fluids)

### Bevy и Rust Physics

40. **Bevy Physics: Avian**. [Tainted Coders](https://taintedcoders.com/bevy/physics/avian)

41. **Bevy Physics: Rapier**. [Tainted Coders](https://taintedcoders.com/bevy/physics/rapier)

---

## Заключение

### Для игровых приложений (realtime)

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

### Для медицинских симуляций (верхнечелюстная пазуха)

**Приоритетная рекомендация: LBM (Lattice Boltzmann Method)**

1. **LBM — лучший выбор** для симуляции в верхнечелюстной пазухе:
   - Прямая работа с вокселизированными CT-данными
   - Тривиальная обработка сложной анатомической геометрии
   - Физически точные результаты (используется в medical CFD)
   - Хорошо изучен для назальных симуляций

2. **Представление данных:**
   - **Рекомендуется начать с вокселей** — минимальная предобработка CT
   - Меш использовать только для визуализации (Marching Cubes)
   - SDF можно генерировать из вокселей при необходимости

3. **Альтернатива — SPH/PBF:**
   - Если нужна симуляция свободной поверхности жидкости
   - Если планируется унификация с игровыми механиками
   - Требуется конвертация CT в SDF или граничные частицы

### Для будущей интеграции с мягкими тканями

1. **Унифицированный XPBD подход:**
   - PBF для жидкости + PBD для мягких тканей
   - Естественное coupling через общие constraints
   - Совместимо с Avian (XPBD-based)

2. **SPH + FEM гибрид:**
   - Более физически точное поведение мягких тканей
   - Fung-type constitutive law для биологических тканей
   - Сложнее в реализации

### Для взаимодействия с rigid bodies (Avian/Rapier)

1. **Начать с weak coupling** — простая реализация, достаточно для большинства игр
2. **Переходить к strong coupling** при необходимости физически корректного поведения
3. **Использовать ExternalForce** компонент Avian для применения сил от жидкости

---

*Документ подготовлен для проекта Spira. Дата: Январь 2025. Обновлено: Январь 2026.*
