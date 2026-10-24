# Informe Baseline — xMemory Fase 1

Fecha: 2026-10-24 · Card workboard: `70f9be91-4238-4453-b791-9f41acd1bba5` · Decisión D1: GO xMemory-first con lente de producto.

## 1. Tests y cobertura (estado real)

Comando: `pytest tests/ --cov=xmemory` (venv propia, Python 3.11, PostgreSQL real con schema de `sql/` aplicado).

| Escenario | Resultado |
|-----------|-----------|
| Con `XMEMORY_DB_URL` a PostgreSQL con schema aplicado | **22 passed, 0 failed — cobertura 92%** |
| Sin PostgreSQL / sin schema (default) | 8 passed, **14 skipped** (`test_bank` requiere DB) |

Cobertura por módulo (con DB): `bank.py` 86%, `models.py`/`recall.py`/`retain.py`/`__init__.py` 100%.

Sin cubrir: `bank.py` líneas 130-152 (ruta de búsqueda vectorial de `recall`) y 191 (fallback de `order_by` inválido).

Requisitos del entorno de test que costaron descubrir (deuda de DX):
- No hay `docker-compose` ni instrucciones para crear la DB de test: la base `xmemory_test`, el rol con password, las extensiones `vector`/`pg_trgm`, el schema `sql/schema.sql` + `sql/functions.sql` y los GRANTs deben montarse a mano.

## 2. Benchmarks — NO reproducibles

- `benchmarks/run_locomo.sh` y `run_evermembench.sh` son **stubs con TODO** (`# TODO: Replace with actual benchmark command`); no ejecutan ningún benchmark y además dependen de un directorio externo inexistente (`~/EverOS/benchmarks/`).
- No hay resultados guardados (`benchmarks/results/` no existe).
- Los claims del README (LoCoMo 62.1%, EverMemBench 45.4%) provienen de una evaluación interna previa con adaptador Hindsight/xMemory; **no verificables desde este repo**. El README ahora los marca como históricos y no reproducibles.

## 3. README — claims corregidos (commits d231150, 78fa5d3)

Corregido para que nada sea refutable por un cliente:
- "production-grade" y "full memory lifecycle: consolidate, evolve" → el código no implementa Consolidate ni Evolve; ahora marcados como roadmap.
- Stats de producción ("6 meses, 11.712 memorias") → marcadas como snapshot histórico de un despliegue privado, con fecha.
- Quick Start roto: referenciaba `docker-compose up -d`, `examples/basic_usage.py` y `XMemoryClient` — **ninguno existe**. Reescrito con el flujo real (`psql` schema + `pip install -e .` + `MemoryBank`).
- API Reference era un API HTTP inexistente (POST/GET, endpoint `graph`); ahora documenta el API Python real. `recall` documentado como requiring `query_embedding` (sin embedding devuelve vacío — verificado en código).
- Licencia: README ya prometía MIT pero LICENSE/pyproject eran Apache-2.0. Relicenciado a MIT (nota: se pierde el patent grant de Apache-2.0; si esto llega a venderse como producto con riesgo de patentes, reconsiderar).

## 4. Packaging

- `pip install -e .` verificado en venv limpia (Python 3.12): OK, metadata `License-Expression: MIT`.
- pyproject completo: nombre, versión, keywords, classifiers honestos (Beta), URLs corregidas a `SabaTech-dev/Saba-xMemory` (antes apuntaban a un repo inexistente `llllJokerllll/xMemory-framework`).
- Nota PEP 639: con `license = "MIT"` el classifier de licencia debe omitirse (setuptools moderno lo rechaza).

## 5. Deuda priorizada para MVP vendible

| # | Prioridad | Ítem |
|---|-----------|------|
| 1 | Alta | `recall` sin `query_embedding` devuelve vacío silenciosamente: implementar fallback (recencia o error explícito) |
| 2 | Alta | Extractor de embeddings: proveedor integrado (OpenAI/Ollama) o doc clara de que el caller lo aporta |
| 3 | Alta | Harness de benchmarks público y autocontenido que reproduzca (o actualice) LoCoMo/EverMemBench; scripts actuales son stubs |
| 4 | Alta | Setup de test reproducible: docker-compose de Postgres+pgvector, script que aplique schema y role de test, quitar los 14 skips |
| 5 | Media | Consolidate/Evolve: o implementarlos o separarlos claramente en roadmap (README ya lo hace; falta el código o la decisión) |
| 6 | Media | Tests de la ruta vectorial (`recall` con embeddings) y del fallback de `order_by` → cobertura 92% → ~100% |
| 7 | Media | `xmemory/retain.py` y `recall.py` son mocks muertos que no usa `MemoryBank`: eliminarlos o integrarlos |
| 8 | Baja | Ejemplo `examples/basic_usage.py` real (el README lo prometía y no existía) |
| 9 | Baja | Warning Pydantic deprecated (`class Config` → `ConfigDict`) en `models.py` |
| 10 | Baja | CI: pipeline que corra Postgres service y garantice 0 skips |

## 6. Entregables

- Commits: `78fa5d3` (licencia/packaging), `d231150` (README honesto), este informe.
- Tests: 22/22 verde con DB; sin DB, skips documentados (deuda #4).
- Benchmarks: baseline declarado NO reproducible; scripts stub documentados.
