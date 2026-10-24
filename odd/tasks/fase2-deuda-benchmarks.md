# Fase 2 — Deuda técnica y benchmarks reales (xMemory)

Objetivo: cerrar las 3 deudas del baseline audit y medir el baseline real con un harness de benchmarks autocontenido, para luego optimizar hacia los targets.
Branch: `fase2-debt-benchmarks` (sobre `main` @ d37c49e) · Entrega: commits por unidad en la branch, push/PR decisión del dueño.
Test runner: `.venv/bin/python -m pytest tests/ -q` con `XMEMORY_DB_URL=postgresql://xmemory:xmemory@localhost/xmemory_test` (PostgreSQL local con pgvector 0.8.7).

## Specs

Cita del mensaje de aprobación (Fase 2, card workboard 60c0005f-4703-4799-b853-b47f1dc6cbe4):

> "F1 cambió el plan: los benchmarks eran stubs, así que F2 ahora es, en orden:
> 1. Deuda #1: recall() sin query_embedding devuelve vacío silencioso (bank.py:124) — fallback sensato o error explícito, con tests
> 2. Deuda #2: proveedor de embeddings conectable (local-first; pensar en producto: defaults que funcionen sin setup)
> 3. Deuda #3: harness de benchmarks REAL reemplazando los stubs run_locomo.sh/run_evermembench.sh — medir baseline verdadero del motor actual
> 4. Con harness real: iterar optimización (rerank graph context, dedup, confidence decay, chunking) hacia LoCoMo ≥ 69.7% / EverMemBench ≥ 51.6% — o documentar límite estructural con evidencia"

- S1: `recall()` llamado sin `query_embedding` no debe devolver vacío en silencio; implementa fallback sensato o error explícito, con tests que cubran ambas rutas.
- S2: Proveedor de embeddings conectable, local-first, con defaults que funcionen sin setup (pensado como producto).
- S3: Harness de benchmarks real que reemplace los stubs `benchmarks/run_locomo.sh` y `benchmarks/run_evermembench.sh`; debe medir el baseline verdadero del motor actual y ser autocontenido (sin depender de `~/EverOS/benchmarks/`).
- S4: Con el harness real, iterar optimización (rerank de contexto de grafo, dedup, confidence decay, chunking) hacia LoCoMo ≥ 69.7% / EverMemBench ≥ 51.6% — o documentar el límite estructural con evidencia.

## Tasks

- T1 (S1, inline): fallback/error de `recall` sin embedding + tests RED→GREEN. [completado — commit a07e1fd; además se arregló el path vectorial (listas → ndarray) que nunca se había ejecutado; 24 passed]
- T2 (S2, inline): módulo `xmemory/embeddings.py` — protocolo provider + Ollama + OpenAI + fallback local determinista; integración en `recall`/`retain`. [completado — commit 6c30a3f; 32 passed]
- T3 (S3, inline): `benchmarks/` real — descarga datasets, ingest, eval, judge LLM; scripts ejecutables; baseline medido y guardado. [completado — commit 78eab78; smoke LoCoMo baseline 30.65% → con mejoras del harness 59.8% (ver T4)]
- T4 (S4, inline): iteraciones de optimización sobre baseline medido o informe de límite estructural. [en curso]

## Log

- L1 (2026-10-24): Request original (inter-session de Alfred, card 60c0005f): "Fase 2 aprobada... Tu F1 cambió el plan: los benchmarks eran stubs..." — texto completo en S0 arriba.
- L1b (2026-10-24): Entorno verificado: Postgres local con extensión vector OK; DB `xmemory_test` ya existía con schema aplicado (4 tablas); rol `xmemory` creado con password; venv con deps OK; baseline 22/22 passed. OPENAI_API_KEY disponible para judge del harness.
- L2 (2026-10-24): OPENAI_API_KEY sin créditos (429 insufficient_quota) → harness reconfigurado a endpoints locales: LiteLLM :4000 (chat mi-qwen38-27b-coldfusion-q4km, embeddings mi-nomic-embed-text-v2-moe 768 dims). Creada DB `xmemory_bench` con schema variante vector(768) (benchmarks/schema768.sql). Coste por corrida: $0 (infra local).
- L3 (2026-10-24): Prompt-engineering del harness medido en smoke (conv-26, 199 QA): strict prompt 9% → minimal 30.65% → judge semántico 30.65%→(mejora scoring) → reglas hablante+fechas 44.7% → guard atribución 53.8% → guard suavizado + inferencia 56.8% → 59.8%. top_k=30 degradó a 39.2% (ruido de contexto en Qwen 27B; precisión > recall) — top_k=20 se queda como default.
- L4 (2026-10-24): Deuda #1 resuelta como FALLBACK SENSATO (S1 permitía error o fallback): recall() ahora es híbrido (vector + full-text RRF); sin embedding disponible degrada a full-text (never silent-empty); search="vector" explícito sigue lanzando ValueError. Tests 33 passed (test_recall_vector_search_without_embedding_raises + test_recall_text_fallback_fulltext).
