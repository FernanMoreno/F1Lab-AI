# F1Lab-AI

**Advanced F1 Regulation Stress-Testing Laboratory**

F1Lab-AI es una plataforma experimental para analizar cómo los reglamentos actuales y alternativos de Fórmula 1 afectan al comportamiento de carrera: adelantamientos, dependencia energética, uso de Boost/Overtake Mode, formación de trenes, velocidades de cierre, dominancia de arquitecturas, robustez entre circuitos y estrategias de carrera.

El proyecto está alineado con la era técnica actual de la F1 2026, marcada por una nueva generación de monoplazas híbridos, mayor peso del sistema eléctrico, Active Aero, Recharge, Boost, Overtake Mode, coches más pequeños y una parrilla ampliada a 11 equipos y 22 coches.

> **Nota importante**: este proyecto simula familias plausibles de coches F1, no coches reales de equipos.  
> F1Lab-AI no replica datos internos de FIA, FOM, Pirelli, Red Bull, Mercedes, Ferrari, McLaren, Cadillac, Audi ni ningún otro equipo o fabricante.  
> El objetivo es identificar debilidades regulatorias y escenarios de fallo antes de que aparezcan en pista.

---

## Objetivo

F1Lab-AI busca responder preguntas como:

- ¿El reglamento 2026 genera demasiada dependencia de batería?
- ¿El Boost u Overtake Mode producen adelantamientos artificiales?
- ¿La Active Aero mejora las carreras o crea comportamientos difíciles de interpretar?
- ¿Existen circuitos donde el reglamento se rompe?
- ¿Una arquitectura de coche puede volverse estructuralmente dominante?
- ¿Las velocidades de cierre pueden volverse peligrosas en ciertos escenarios?
- ¿La estrategia energética puede pesar más que el ritmo real del coche?
- ¿Qué cambios regulatorios mínimos mejorarían el espectáculo y la seguridad?

El proyecto no intenta predecir ganadores reales de Grandes Premios. Su foco es el **stress testing regulatorio**.

---

## Características principales

- **Modelado de regulaciones F1**
  - Baseline 2025.
  - Baseline inicial 2026.
  - Aproximación actual/refinada 2026.
  - Escenarios experimentales para stress testing.

- **Familias sintéticas de coches**
  - Arquetipos como `low_drag_missile`, `high_downforce_stable`, `energy_efficient`, `tyre_whisperer` y otros.
  - No representan equipos reales.

- **Métricas de salud regulatoria**
  - Battery Dependency Index.
  - Artificial Pass Index.
  - Dangerous Closing Speed Index.
  - Train Formation Index.
  - Dominant Architecture Risk.
  - Regulation Robustness Score.

- **Simulación de escenarios**
  - Batallas entre dos coches.
  - Comparación entre circuitos.
  - Estrés energético.
  - Riesgo de trenes.
  - Velocidad de cierre.
  - Dominancia de arquitecturas.

- **Búsqueda adversarial**
  - Exploración automática de escenarios donde un reglamento puede fallar.

- **Soporte para agentes LLM**
  - Agentes para diseño de experimentos, análisis adversarial, validación y reporting.
  - Los agentes no sustituyen la simulación física ni las métricas deterministas.

- **Soporte para RL**
  - Entorno compatible con Gymnasium para futuras estrategias aprendidas.
  - El RL es una capa posterior; primero se validan los modelos deterministas.

---

## Instalación

```bash
git clone https://github.com/your-org/f1lab-ai.git
cd f1lab-ai

pip install -e ".[data,dashboard,ml,rl,agents,optimization,dev]"
````

O usando Make:

```bash
make install
```

---

## Uso rápido

```python
from reglabsim import create_facade

facade = create_facade()

print(facade.list_regulations())
# ['regulation_2025', 'regulation_2026_initial', 'regulation_2026_refined', 'regulation_experimental']

print(facade.list_car_families())
# ['low_drag_missile', 'high_downforce_stable', 'energy_efficient', ...]

print(facade.list_circuits())
# ['monza', 'monaco', 'baku', 'barcelona']

result = facade.run_battle_experiment(
    "configs/experiments/baku_closing_speed.yaml"
)

metrics = facade.compute_metrics(result)
print(metrics)
# {
#   'battery_dependency_index': 0.32,
#   'artificial_pass_index': 0.41,
#   'dangerous_closing_speed_index': 0.04,
#   ...
# }
```

---

## Estructura del proyecto

```text
f1lab-ai/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── Makefile
├── docker-compose.yml
│
├── configs/
│   ├── regulations/
│   │   ├── regulation_2025.yaml
│   │   ├── regulation_2026_initial.yaml
│   │   ├── regulation_2026_refined.yaml
│   │   └── regulation_experimental.yaml
│   │
│   ├── experiments/
│   │   ├── baku_closing_speed.yaml
│   │   ├── monza_energy_stress.yaml
│   │   ├── monaco_train_index.yaml
│   │   └── barcelona_balance.yaml
│   │
│   ├── car_families.yaml
│   ├── metric_thresholds.yaml
│   └── duckdb/
│       └── schema.sql
│
├── reglabsim/
│   ├── interfaces.py
│   ├── facade.py
│   │
│   ├── data/
│   ├── regulation/
│   ├── conditions/
│   ├── circuits/
│   ├── vehicle/
│   ├── lap/
│   ├── race/
│   ├── strategy/
│   ├── metrics/
│   ├── optimization/
│   ├── rl/
│   ├── validation/
│   └── uncertainty/
│
├── agents/
│   ├── graph.py
│   ├── regulation_agent.py
│   ├── experiment_agent.py
│   ├── adversarial_agent.py
│   ├── validation_agent.py
│   └── report_agent.py
│
├── dashboards/
│   ├── streamlit_app.py
│   └── pages/
│
├── experiments/
├── notebooks/
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── fixtures/
```

---

## Regulaciones disponibles

| ID                        | Descripción                                    | Uso recomendado                                       |
| ------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `regulation_2025`         | Baseline de comparación pre-2026               | Comparar el comportamiento anterior con la nueva era  |
| `regulation_2026_initial` | Modelo inicial del reglamento 2026             | Analizar riesgos del concepto 2026 de base            |
| `regulation_2026_refined` | Aproximación actual/refinada 2026 del proyecto | Usar como baseline principal para escenarios actuales |
| `regulation_experimental` | Escenarios extremos o alternativos             | Stress testing adversarial                            |

> Las configuraciones 2026 del proyecto son modelos computacionales basados en información pública y supuestos explícitos. No deben interpretarse como una copia completa de los documentos técnicos internos de FIA o de los equipos.

---

## Contexto técnico F1 2026 modelado

F1Lab-AI está diseñado para estudiar elementos clave de la era 2026:

* Mayor peso relativo de la potencia eléctrica.
* Recharge y gestión de energía.
* Boost para ataque o defensa.
* Overtake Mode.
* Active Aero con modos de recta y curva.
* Reducción de drag en recta.
* Mantenimiento de carga en curva.
* Coches más pequeños y ligeros.
* Menor anchura de neumáticos.
* Nueva aerodinámica orientada a reducir turbulencia.
* Parrilla de 11 equipos y 22 coches.

Estos elementos hacen que el reglamento 2026 sea especialmente interesante para simulación adversarial, porque pequeñas diferencias en energía, drag, downforce, aire sucio o estrategia pueden cambiar mucho la dinámica de carrera.

---

## Familias sintéticas de coches

Las familias de coches se definen en:

```text
configs/car_families.yaml
```

Ejemplos:

| Familia                   | Idea principal                                  |
| ------------------------- | ----------------------------------------------- |
| `low_drag_missile`        | Baja resistencia, fuerte en rectas              |
| `high_downforce_stable`   | Más carga y estabilidad en curva                |
| `energy_efficient`        | Mejor gestión y recuperación eléctrica          |
| `tyre_whisperer`          | Menor degradación de neumático                  |
| `dirty_air_resistant`     | Pierde menos rendimiento siguiendo a otro coche |
| `qualifying_specialist`   | Muy fuerte a una vuelta                         |
| `race_pace_concept`       | Consistencia en stint largo                     |
| `boost_dependent_concept` | Rendimiento muy ligado al uso de energía        |

Estas familias son arquetipos regulatorios. No representan equipos reales.

---

## Métricas de salud regulatoria

| Métrica                       | Descripción                                                                    | Umbral crítico inicial |
| ----------------------------- | ------------------------------------------------------------------------------ | ---------------------: |
| Battery Dependency Index      | Mide cuánto depende el rendimiento del estado de batería                       |                 > 0.40 |
| Artificial Pass Index         | Mide adelantamientos explicados principalmente por ventaja energética temporal |                 > 0.45 |
| Dangerous Closing Speed Index | Mide eventos con velocidades de cierre potencialmente peligrosas               |                 > 0.05 |
| Train Formation Index         | Mide trenes donde hay coches cerca pero sin adelantamiento viable              |                 > 0.35 |
| Dominant Architecture Risk    | Mide si una familia de coche domina demasiados escenarios                      |                 > 0.50 |
| Regulation Robustness Score   | Mide robustez global del reglamento entre escenarios                           |            bajo = peor |

Los umbrales se definen en:

```text
configs/metric_thresholds.yaml
```

Los valores iniciales son proxies de investigación, no valores oficiales FIA.

---

## Experimentos preconfigurados

| Experimento                | Objetivo                                                      |
| -------------------------- | ------------------------------------------------------------- |
| `baku_closing_speed.yaml`  | Evaluar velocidades de cierre en un circuito de rectas largas |
| `monza_energy_stress.yaml` | Analizar estrés energético y eficiencia en recta              |
| `monaco_train_index.yaml`  | Medir riesgo de trenes y baja capacidad de adelantamiento     |
| `barcelona_balance.yaml`   | Evaluar balance global entre arquitecturas                    |

Ejemplo:

```bash
python -m reglabsim.experiments.run --config configs/experiments/baku_closing_speed.yaml
```

Si el comando aún no está implementado, usar la fachada desde Python:

```python
from reglabsim import create_facade

facade = create_facade()
result = facade.run_battle_experiment(
    "configs/experiments/baku_closing_speed.yaml"
)
metrics = facade.compute_metrics(result)
print(metrics)
```

---

## Comandos disponibles

```bash
make install        # Instalar dependencias
make test           # Ejecutar tests
make lint           # Ejecutar linting
make format         # Formatear código
make typecheck      # Type checking
make run-dashboard  # Ejecutar dashboard Streamlit
make clean          # Limpiar artefactos
```

Verifica siempre el `Makefile` antes de asumir que un comando existe.

### CLI operativa actual

```bash
python -m reglabsim.cli run-multiagent-race configs/campaigns/suzuka_mini_multiagent.yaml
python -m reglabsim.cli run-redteam-campaign configs/campaigns/baku_redteam.yaml
python -m reglabsim.cli describe-track suzuka
python -m reglabsim.cli show-condition-profile windy_baku
python -m reglabsim.cli ingest-session-data 2024 suzuka race --drivers 1
python -m reglabsim.cli ingest-weekend-results 2024 1
python -m reglabsim.cli ingest-historical-weather suzuka 2024-04-07 2024-04-07
python -m reglabsim.cli build-weather-profile suzuka 2024-04-07 2024-04-07 --profile-id suzuka_openmeteo_2024
python -m reglabsim.cli validate-public-session configs/campaigns/suzuka_mini_multiagent.yaml 2024 suzuka race
```

Los comandos de ingestión guardan datasets reproducibles en `data/raw/` y `data/silver/` con manifiestos JSON y Parquet por partición.
`validate-public-session` compara ritmo medio, clima e incidentes del run contra la sesión pública ingerida y devuelve un score de credibilidad proxy.

---

## Tests

Ejecutar:

```bash
make test
```

O directamente:

```bash
pytest
```

El proyecto debe mantener tests para:

* conservación de energía,
* límites de SOC,
* monotonía de drag/downforce,
* restricciones regulatorias,
* carga de configs,
* métricas,
* estado de carrera,
* reproducibilidad con seed.

---

## Dashboard

Inicia el dashboard Streamlit:

```bash
make run-dashboard
```

O directamente:

```bash
streamlit run dashboards/streamlit_app.py
```

El dashboard debe etiquetar claramente:

* escenarios sintéticos,
* datos reales,
* datos calibrados,
* supuestos,
* incertidumbre,
* versión de regulación.

---

## Salida esperada de experimentos

Los experimentos deberían guardar resultados reproducibles en:

```text
outputs/
└── experiments/
    └── <experiment_name>/
        ├── config.yaml
        ├── metrics.json
        ├── metadata.json
        ├── summary.md
        └── simulation.parquet
```

La metadata debe incluir:

```json
{
  "experiment_name": "...",
  "created_at": "...",
  "config_hash": "...",
  "seed": 42,
  "regulation": "regulation_2026_refined",
  "simulator_version": "0.1.0",
  "data_version": "..."
}
```

---

## Roadmap

* [x] Estructura de paquetes y configuración inicial
* [x] `AGENTS.md` con reglas de arquitectura para agentes
* [x] Configs YAML de regulaciones, familias y experimentos
* [x] Stubs de módulos con type hints y docstrings
* [x] Tests unitarios e integración iniciales
* [ ] Pipeline funcional end-to-end de batalla entre dos coches
* [ ] Modelos físicos simplificados: drag, downforce, energía, tyre grip
* [ ] Integración completa de métricas
* [ ] Outputs reproducibles por experimento
* [ ] Validación básica contra telemetría pública
* [ ] Monte Carlo de escenarios
* [ ] Búsqueda adversarial completa
* [ ] Dashboard con visualización de resultados
* [ ] Ambiente RL funcional
* [ ] Integración robusta con FastF1/OpenF1/Jolpica

---

## Limitaciones conocidas

* Los modelos físicos iniciales son simplificados.
* Las familias de coches son sintéticas.
* El proyecto no predice resultados reales de F1.
* El proyecto no replica datos internos de FIA, Pirelli, FOM ni equipos.
* Las métricas de espectáculo/artificialidad son proxies investigativos.
* Los resultados deben interpretarse con incertidumbre y contexto.
* Cualquier comparación con F1 real requiere validación contra datos públicos.

---

## Buenas prácticas

* No mezclar datos reales y sintéticos sin etiquetarlos.
* No usar nombres de equipos reales para familias sintéticas.
* No sacar conclusiones regulatorias sin métricas y sensibilidad.
* No entrenar RL serio antes de tener un simulador determinista validado.
* No usar agentes LLM como fuente de verdad física.
* No modificar configs base sin versionado.
* No ocultar supuestos.

---

## Licencia

MIT

---

## Contribuir

Ver `CONTRIBUTING.md` para guidelines de desarrollo.

---

**F1Lab-AI**: laboratorio adversarial para identificar debilidades regulatorias antes de que aparezcan en pista.

```
::contentReference[oaicite:1]{index=1}
```

[1]: https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-2026-regulations.6j0tS0hrHG2T01tpmK6XYz "The beginner’s guide to the 2026 Formula 1 regulations"
