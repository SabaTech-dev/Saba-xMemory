# Diseño: port de patrones de projectmem a xMemory

**Fecha:** 2026-10-24 · **Rama:** `fase2-debt-benchmarks` · **Estado:** propuesta (entrada de F3)

## Objetivo

Aprovechar explícitamente el diseño de [projectmem](https://github.com/riponcm/projectmem)
(arXiv 2606.12329) en xMemory: lo mejor de projectmem (juicio determinista pre-acción,
event-sourcing auditable, cero coste de lectura) + lo mejor de Engram (memoria semántica
con embeddings, recall híbrido), en nuestro framework (Postgres + pgvector, paquete
`xmemory`, orientado a producto).

## Licencia y estrategia de port

**projectmem es MIT** (verificado en GitHub API, 2026-10-24: `spdx_id: MIT`).
Vendoring es legalmente viable, pero se recomienda **port limpio de conceptos**
(implementación propia dentro de `xmemory/`) por:

1. projectmem está acoplado a su CLI (typer) y a almacenamiento en archivos YAML/JSONL
   por proyecto; xMemory es librería + Postgres. Copiar código obligaría a arrastrar
   dependencias y estructuras ajenas.
2. Los patrones valiosos son de diseño (schema, invariantes, thresholds), no de código:
   el port es pequeño y limpio-room.
3. MIT nos deja la puerta abierta a copiar detalles concretos (p. ej. la tabla de
   sufijos de `location_to_file`, la lógica de snooze) citando atribución en el NOTICE.

## Mapa patrón → ubicación en xMemory

### P1 — Schema de eventos tipados → `xmemory/events.py` (módulo nuevo)

Patrón observado (`models.py`): `Event` con tipos `issue | hypothesis | attempt | fix |
decision | note`, `outcome in {worked, failed, partial}`, `issue_id` agrupando
intentos fallidos de un mismo problema, `supersedes` (puntero de retiro, resuelto
en lectura — el log nunca se reescribe), validación en `__post_init__`.

Adaptación xMemory:

- Dataclass `MemoryEvent` con tipos `{issue, attempt, fix, decision, note, gotcha}`
  (se añade `gotcha` cross-project, que en projectmem vive en global memory).
- `supersedes` resuelto en lectura (`superseded_ids(events)`): append-only estricto.
- Enlaces a unidades de memoria existentes: campo opcional `memory_ids: list[str]`
  para que un evento cite evidencia semántica (esto projectmem no lo tiene: su
  memória no tiene unidades).

**Qué se descarta:** `capture_source/capture_confidence` simplificado a
`source in {agent, git_post_commit, git_post_merge, manual}`; sin `hypothesis`
(aportaba poco frente a `attempt` + `notes`).

### P2 — Store event-sourced → tabla `memory_events` + export JSONL

Patrón observado (`storage.py`): un único `events.jsonl` append-only por proyecto,
`MEM_DIR = .projectmem`, descubrimiento ascendente del directorio, registro de
proyectos global. Cero vectores: lectura = replay determinista, coste 0.

Adaptación xMemory:

- Tabla Postgres `memory_events` append-only (PK `id`, `bank_id`, `type`, `summary`,
  `outcome`, `issue_id`, `supersedes`, `timestamp`, `files`, `source`, `git_commit`).
  **Nunca UPDATE/DELETE** — el retiro es `supersedes`, resuelto en lectura.
- Export opcional a `.xmemory/events.jsonl` para repos que quieran el log
  versionado en git (misma forma de datos que projectmem, sin ser el primario:
  Postgres sigue siendo la fuente).
- Los events NO se embeben por defecto (lectura 0-coste). Búsqueda = FTS sobre
  `summary` + filtros por tipo; lo semántico queda en las memory units.

**Qué se descarta:** un archivo por evento en YAML (projectmem migró a JSONL
internamente; YAML por evento no aporta en nuestro modelo).

### P3 — Precheck determinista (~80 ms) → `xmemory/judgment.py`

Patrón observado (`precheck.py`): compara archivos tocados (staged o working tree)
contra eventos recientes (30 días): intentos fallidos sobre los mismos archivos →
`warn`; 3+ fallos → `block` (nivel configurable); churn alto (4 cambios en ventana
de 7 commits) → aviso; snooze con TTL auditado en el propio log. Todo sin LLM.

Adaptación xMemory:

- `xmemory/judgment.py`: `precheck(files: list[str], level="warn", root=None) ->
  list[Warning]` con los mismos thresholds (`HIGH_CHURN_THRESHOLD=4`,
  `CHURN_WINDOW_COMMITS=7`, `FAILED_ATTEMPT_BLOCK_COUNT=3`, `RECENT_DAYS=30`) y
  snooze con TTL persistido como evento.
- Expuesto como tool MCP y como comando CLI. Determinista, medible (objetivo
  < 100 ms p95 en repos de miles de eventos: índice `(bank_id, timestamp)`).
- Diferenciador sobre projectmem: además del solape de archivos, el precheck puede
  usar el recall semántico como *segunda pasada opcional* (solo cuando hay
  candidatas ambiguas), manteniendo el caso común a coste 0.

**Qué se descarta:** el parseo de staging area de git dentro del paquete (xMemory
recibe la lista de archivos; el glue de git vive en los hooks, P4).

### P4 — Git hooks opt-in → `hooks/` + instalador

Patrón observado (`hooks.py`): snippets POSIX entre marcadores
`# >>> projectmem auto-capture >>>`, path absoluto del binario horneado en
instalación con fallback `command -v`, stdout/stderr a `/dev/null` para no
ensuciar el prompt; pre-commit (warn de approaches fallidos), post-commit
(auto-clasificación del mensaje), post-merge (herencia cross-project).

Adaptación xMemory:

- `hooks/install.py` genera los 3 hooks con marcadores análogos y path absoluto
  del entrypoint. `post-commit` captura `fix/decision` inferidos del mensaje
  (`feat:`→decision candidate, `fix:`→fix) con `capture_confidence`.
- **Restricción del entorno (no negociable):** `core.hooksPath` global de este
  server es default-deny (`~/.config/git/hooks/`, card 3a53a897). La instalación
  es **siempre opt-in por repo** tras revisión del hook
  (`git config --local core.hooksPath <repo>/.githooks`); el instalador lo
  documenta y nunca toca el global. Hooks de terceros siguen la política existente.
- `post-merge`: herencia de gotchas entre bancos (`bank_id` origen → destino)
  marcando eventos con `source=git_post_merge`.

### P5 — Proyección/compaction determinista → `xmemory/projection.py`

Patrón observado (`brief.py`): proyección del event log a resumen por secciones
(warnings, stale, open issues, decisions, gotchas, score de cobertura de archivos);
misma entrada → misma salida, sin LLM.

Adaptación xMemory:

- `project(bank_id) -> Projection` determinista: decisiones activas, issues abiertos
  (issue_id sin `fix` exitoso posterior), gotchas, approach map
  (issue → attempts fallidos → fix que funcionó), score de cobertura.
- Compaction: los eventos `superseded` se excluyen de la proyección pero permanecen
  en el log (auditable). Sin regenerate-no-determinista.

**Qué se descarta:** dashboard/visualize (fuera de scope del motor; el CLI puede
llegar después).

## No se porta (y por qué)

| Descartado | Razón |
|---|---|
| Stance "no vectores jamás" | El producto xMemory ES semántico; los eventos son la capa de juicio, no un reemplazo |
| CLI typer como superficie principal | xMemory es librería + MCP; CLI mínimo secundario |
| `.projectmem/` como único storage | Postgres ya es la fuente; JSONL queda como export git-friendly opcional |
| global_memory / multi-project registry | Cubierto por el concepto `bank_id` + Engram cross-project ya existente |

## Estimación por patrón

| Patrón | Módulo | Líneas estimadas (código+tests) | Riesgo |
|---|---|---|---|
| P1 events schema | `xmemory/events.py` | ~150 | bajo |
| P2 store | schema SQL + `events.py` (append/read/export) | ~200 | medio (migración schema) |
| P3 precheck | `xmemory/judgment.py` | ~250 | medio (thresholds a validar) |
| P4 hooks | `hooks/` + instalador | ~180 | medio (política hooks del server) |
| P5 proyección | `xmemory/projection.py` | ~200 | bajo |
| **Total** | | **~980** | entrega en 2-3 work units |

## Conclusión para F3

F3 se implementa con estos patrones **nativos en xMemory** (port, sin dependencia pip
de projectmem). La licencia MIT permite copiar fragmentos concretos si algún detalle
lo justifica (con atribución); no se prevé necesario salvo utilidades menores como
`location_to_file`. Importar projectmem como dependencia se descarta: traería su CLI,
su storage en archivos y su modelo de proyecto, los tres en conflicto con la
arquitectura Postgres-first de xMemory.

## Fuentes

- Código: riponcm/projectmem @ main (models.py, storage.py, precheck.py, hooks.py,
  brief.py), consultado 2026-10-24.
- Licencia: GitHub API `repos/riponcm/projectmem` → `spdx_id: MIT`.
- Resumen previo: `~/.openclaw/workspace/memory/research-projectmem.md` (card 4c983040).
- Paper: arXiv 2606.12329.
