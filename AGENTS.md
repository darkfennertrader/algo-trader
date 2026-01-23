# Repository Agent Instructions

- Maintain this file for coding workflow guidance only; keep other documentation in README.md.
- Package management and tooling: always use `uv` for Python tasks (installs, running scripts, dependency updates). Do not use `pip`, `pipenv`, `poetry`, or `venv` directly.
- Use `uv run` for executing project commands and `uv add` / `uv remove` for dependency changes to keep `pyproject.toml` in sync.
- Code contributions: ensure every Python snippet follows idiomatic patterns and aligns SOLID with the modern principles below (SRP -> cohesion, DIP/ISP -> depend on abstractions and keep interfaces lean, LSP -> prefer composition over inheritance, OCP -> low coupling, decouple creation from use).
- Type checking: write Python code with complete and correct type hints, and ensure it passes static type checking with both Pyright/Pylance and Pylint (with type checking enabled), fixing any type or lint issues they would report.
- For third-party library features, consult documentation via Context7 before implementing to ensure alignment with upstream APIs and usage.

## Coding Workflow
- Linting and typing: run `uv run pyright` and `uv run pylint algo_trader` before pushing changes.
- Tests: run `uv run pytest` at minimum; add focused tests for new behaviors. (No integration tests)
- Logging: prefer structured logging (no ad-hoc prints); include symbols/order ids and error context for operability.
- Error handling: fail fast on invalid input/config; catch only to add context and re-raise rather than swallowing exceptions.
- Error handling conventions: define typed errors in `algo_trader/domain/errors.py`, raise them in infrastructure/providers/application with context, and catch only at CLI boundaries to log a friendly message and exit non-zero.

## Coding Rules
- Load environment variables with `load_dotenv()` and raise an error when a required
  variable is missing; do not set defaults.
- Imports: expose a small, explicit public API in each package `__init__.py` and import from the package, not deep submodules.
- Imports: keep external imports to package level or at most two levels deep; avoid deeper paths unless necessary.
- Imports: use absolute imports across package boundaries; allow one-level relative imports only within the same package (e.g., `from .foo import bar`).
- Module layout: allow multiple submodules but keep packages cohesive; split by responsibility, not file size.
- Naming & intent: submodules should be named by domain responsibility (e.g., `models`, `protocols`, `service`), and `__init__.py` defines what is public.

## Principles of Modern Software Design
- Favor composition over inheritance.
- Keep modules highly cohesive (single responsibility principle).
- Keep modules loosely coupled; avoid content and global coupling.
- Depend on abstractions rather than concretions (use Protocol or type in Python for abstraction)
- Separate creation from usee: Factories or simple creator functions (wired via dependency
  injection) create concrete objects in one place, implementing DIP by returning
  abstractions (protocols/interfaces) instead of concrete types. Client/business code only uses those abstractions, not constructors, which satisfies OCP.
- Start with the data: treat data as the center of the design; keep data near the behavior
  that acts on it and enforce layered boundaries so modules only exchange the minimal, relevant fields required.
- Data-focused classes: prefer dataclasses to reduce boilerplate, and prefix private-like
  instance attributes with an underscore to signal they are internal.
- When working in a dirty tree, patch intentionally and be explicit: import all needed
  specifics from YAML config files so every fix is reproducible and self-contained.
- Function sizing: keep functions moderate in length (aim for under ~60 lines) to preserve
  readability and testability.
- Keep solutions simple and minimal: easy to understand, easy to change and easy to test.
  Take into consideration the following principles: KISS (keep it simple stupid), YAGNI
  (you ain't gonna need it). Apply these principles in the parts that the code is not
  gonna change likely (always ask the user before simplifying things)

## Function and class sizing
- Focus first on clarity and single responsibility; length is a symptom, not the goal.
- Functions: target small-to-moderate size (roughly 5–40 lines). Once past ~50–80 lines, look to split clear substeps into helpers and trim deep nesting; avoid trivial one-line wrappers that hurt readability.
- Classes: keep them cohesive with one clear concept; if methods feel unrelated or attributes balloon (≈10+ fields), consider splitting and using composition instead of a “god object.”
- Python habits: prefer small, well-named functions; group related logic into modules over huge classes; use dataclasses or lean classes for data containers; keep public APIs small and push complexity into underscore-prefixed helpers.
- Linters (pylint) with rules like too-many-locals/branches/statements/instance-attributes help flag overgrown code.
- Quick check: can you explain it in one sentence, view it on one screen, see a single reason to change, and make it clearer by extracting a helper? If not, refactor.

## Architecture & patterns for trading
- Strategy: define a strategy interface/protocol and pluggable concrete strategies (engine calls `strategy.on_bar(...)`) so you can swap mean reversion, momentum, etc. without touching the engine.
- Observer / Pub-Sub: event bus for ticks, fills, risk alerts; subjects publish (`market_data`, `order_filled`, etc.) and observers (strategies, risk, UI, logger, PnL) react.
- Adapter + Facade: wrap each broker/feed API to a unified interface (`send_order`, `cancel_order`, `positions`) and expose a simple Broker facade so strategies stay broker-agnostic.
- Factory / Abstract Factory: build brokers, feeds, strategies, storage from config (YAML/env); central composition keeps creation separate from use.
- State: model order lifecycle (NEW → SUBMITTED → PARTIALLY_FILLED → FILLED/CANCELED/REJECTED) with explicit states and transitions to avoid scattered status checks.
- Command: encapsulate actions (`PlaceOrderCommand`, `CancelOrderCommand`) so they can be queued, logged, replayed (backtests) and dispatched by an executor.
- Decorator: layer cross-cutting concerns around broker/strategy (risk checks, logging, throttling) without bloating core classes.
- Template Method: shared trading loop skeleton with hooks; `BacktestEngine` vs `LiveTradingEngine` override data/execution steps only.
- Pipeline: compose transformations from raw ticks → cleaned → bars → indicators → signals using chained functions/generators or async stages.
- Concurrency: use producer-consumer or async I/O (`queue.Queue`/`asyncio.Queue`) for feeds and order streams to keep processing non-blocking.
- Architecture style: layered + event-driven; domain (orders/positions/strategies), infrastructure (brokers/feeds/storage/bus), application (engine/orchestration/API), all reacting to events. SOLID + low coupling guide boundaries; abstractions allow swapping implementations.
