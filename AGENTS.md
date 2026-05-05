# F1Lab-AI / RegLab-F1

F1Lab-AI es una plataforma avanzada de laboratorio de pruebas de estrés regulatorio para la Fórmula 1.

El proyecto **no** busca clonar autos F1 reales ni predecir el rendimiento exacto de los equipos. Su objetivo es simular escenarios regulatorios, vehiculares, de carrera, energéticos, aerodinámicos, de neumáticos, estratégicos y ambientales plausibles para detectar cómo un conjunto dado de regulaciones F1 podría fallar antes de que esas fallas aparezcan en pista.

Objetivo de investigación primario:

> Dada una regulación候选ante de F1, generar familias plausibles de autos, simular vueltas, batallas, carreras y escenarios adversarials, y luego identificar modos de falla como dependencia excesiva de batería, adelantamientos artificiales, velocidades de cierre peligrosas, formación de trenes, colapso específico del circuito, comportamiento inestable de aero activo y riesgo de arquitectura dominante.

---

## Principios de Arquitectura

### Arquitectura de Dominios

La arquitectura debe incluir domains explícitos para:

- **regulation**: Lógica de regulación, versioning, constraints
- **conditions**: Clima, estado de pista, grip, evolución
- **strategy**: Estrategia de pits, neumáticos, energía, ataque/defensa
- **uncertainty**: Cuantificación de incertidumbre, análisis de sensibilidad
- **validation**: Calibración, backtesting, comparación con telemetría real
- **simulation facade**: Interfaz unificada para agentes y dashboards

### Módulos de Simulación Requeridos

```text
reglabsim/
├── interfaces.py          # Contratos y protocolos
├── facade.py              # Fachada de simulación
├── data/                  # Ingesta de datos (FastF1, OpenF1, Jolpica)
├── regulation/            # Modelo de regulaciones
├── conditions/            # Condiciones ambientales
├── circuits/              # Modelo de circuitos
├── vehicle/               # Modelo de vehículo
├── lap/                   # Simulación de vuelta
├── race/                  # Simulación de carrera
├── strategy/              # Estrategia de carrera
├── metrics/               # Métricas de regulación
├── optimization/          # Optimización y búsqueda adversarial
├── rl/                    # Entornos de RL
├── validation/            # Calibración y backtesting
└── uncertainty/           # Cuantificación de incertidumbre
```

### Interfaces y Contratos

El proyecto debe definir contratos en `reglabsim/interfaces.py`.

Protocolos requeridos:

```text
DataSource
DataSourceBase
CircuitBase
VehicleBase
LapSimulatorBase
RaceSimulatorBase
MetricBase
MetricRegistry
OptimizerBase
SimulationFacade
```

### Regla Crítica

Los agentes deben depender solo de `SimulationFacade`, no de concretos de simulación.

Mal:

```python
from reglabsim.race.race_simulator import RaceSimulator
from reglabsim.vehicle.ers import ERSModel
```

Bien:

```python
from reglabsim.interfaces import SimulationFacade
```

---

## Métricas de Regulación

### Battery Dependency Index

Mide cuánto el rendimiento depende del estado de energía eléctrica.

### Artificial Pass Index

Mide la proporción de adelantamientos causados principalmente por ventaja temporal de energía.

### Dangerous Closing Speed Index

Mide con qué frecuencia la velocidad de cierre excede un umbral de seguridad.

### Train Formation Index

Mide con qué frecuencia los autos permanecen dentro de distancia de ataque pero no pueden adelantar realistamente.

### Dominant Architecture Risk

Mide si una familia de autos sintética domina a través de circuitos y condiciones.

### Regulation Robustness Score

Mide qué tan a menudo una regulación permanece saludable a través de circuitos, familias de autos, estrategias y condiciones.

---

## Familias de Autos

Las familias de autos pertenecen en `configs/car_families.yaml`.

Familias ejemplo:

- `low_drag_missile`: Alta velocidad en recta, baja carga aerodinámica
- `high_downforce_stable`: Alta carga aerodinámica, estable en curvas
- `energy_efficient`: Optimizado para recuperación de energía
- `tyre_whisperer`: Gestor de neumáticos
- `dirty_air_resistant`: Resistente a dirty air
- `qualifying_specialist`: Especialista en clasificación
- `race_pace_concept`: Concepto de ritmo de carrera

---

## Reglas de Simulación

- Los agentes LLM **pueden**:
  - Proponer planes de experimentos
  - Clasificar modos de falla
  - Resumir resultados
  - Proponer sweeps de parámetros

- Los agentes LLM **no pueden**:
  - Fabricar telegetría
  - Alterar resultados de simulación
  - Reclamar conclusiones del mundo real sin evidencia
  - Reemplazar tests unitarios
  - Importar simuladores concretos directamente

---

## Definición de Terminado

Una feature no está completa hasta que:

- Está implementada en el módulo correcto
- Tiene tests
- Tiene interfaces tipadas donde es práctico
- Documenta asunciones
- Soporta ejecución determinista si es estocástica
- Es configurable si está relacionada con regulación
- No hace afirmaciones específicas de equipos hardcoded
- Tiene al menos un test de sanity
- No rompe experimentos existentes
- Puede ejecutarse a través de un comando o config documentada
- No hace bypass del facade cuando es usado por agentes

---

## Dependencias

Grupos de dependencias opcionales:

```toml
[project.optional-dependencies]
data = ["fastf1>=3.0", "requests>=2.31"]
dashboard = ["streamlit>=1.35", "plotly>=5.20"]
ml = ["scikit-learn>=1.4", "torch>=2.0"]
rl = ["gymnasium>=0.29", "stable-baselines3>=2.0"]
agents = ["langgraph>=0.2", "langchain-core>=0.3"]
optimization = ["optuna>=3.6", "pymoo>=0.6"]
dev = ["pytest>=8.0", "ruff>=0.3", "mypy>=1.9"]
```

---

## Comandos

```bash
make install      # Instalar dependencias
make test         # Ejecutar tests
make lint         # Linting con ruff
make format       # Formatear con black
make typecheck    # Type checking con mypy
make run-dashboard # Ejecutar dashboard
```

---

## Estructura de Configs

```text
configs/
├── regulations/        # YAML de regulaciones
├── car_families.yaml   # Familias de autos sintéticas
├── metric_thresholds.yaml #umbrales de métricas
├── duckdb/schema.sql   # Schema de base de datos
└── experiments/        # Configs de experimentos
```