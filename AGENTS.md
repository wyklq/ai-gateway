# AGENTS.md - AI Gateway Development Guide

This document provides guidance for AI coding agents working in the LangDB AI Gateway codebase.

## Project Overview

LangDB AI Gateway is an open-source enterprise AI gateway built in Rust. It provides a unified interface to all LLMs using the OpenAI API format, with support for routing, guardrails, cost control, rate limiting, and MCP (Model Context Protocol).

**Workspace Structure:**
- `gateway/` - Main binary crate (`ai-gateway`). HTTP server (actix-web), CLI, TUI, OpenTelemetry gRPC collector (tonic, port 4317).
- `core/` - Core library (`langdb_core`). Business logic: model providers, handlers, routing, guardrails types, pricing, telemetry, MCP support.
- `guardrails/` - Guardrails library (`langdb_guardrails`). Content evaluation against user-defined guard rules.
- `udfs/` - User-defined functions (`langdb_udf`) for ClickHouse integration. Standalone binary.

**Note:** `default-members` in the workspace is `["gateway", "udfs"]`. Running bare `cargo build` or `cargo test` only targets these two (though `core` and `guardrails` are compiled transitively as dependencies of `gateway`). Use `--all` to target every member explicitly.

**Feature Flags:**
- `langdb_core` has a `database` feature (enabled by default) that pulls in `openssh`, `clickhouse`, and `tokio-util` for ClickHouse and SSH tunnel support.

## Environment Setup

Before running any build, lint, or test command, set the required Rust flags:

```bash
export RUSTFLAGS="--cfg tracing_unstable --cfg aws_sdk_unstable"
```

The gateway also supports `.env` files (`dotenv`). API keys and other secrets can be placed in a `.env` file at the project root.

## Build Commands

```bash
# Build all workspace members
cargo build --all

# Build in release mode
cargo build --release

# Build specific crate
cargo build -p ai-gateway
cargo build -p langdb_core
cargo build -p langdb_guardrails
```

## Lint Commands

```bash
# Run clippy (treats warnings as errors in CI)
# RUSTFLAGS must be set (see Environment Setup)
cargo clippy --all -- -D warnings

# Check formatting
cargo fmt -- --check

# Auto-fix formatting
cargo fmt
```

## Test Commands

```bash
# Run all tests (excludes udfs crate)
cargo test --all --exclude udfs

# Run tests for specific crate
cargo test -p langdb_core
cargo test -p langdb_guardrails

# Run a single test by name
cargo test test_name

# Run a single test in specific crate
cargo test -p langdb_core test_name

# Run tests with output
cargo test -- --nocapture
```

## Architecture Overview

Request flow from HTTP entry to LLM invocation:

```
HTTP Request (/v1/chat/completions)
  -> actix-web handler (core/src/handler/chat.rs)
    -> Rate limiting middleware (core/src/handler/middleware/rate_limit.rs)
    -> Cost / limit checks (gateway/src/limit.rs)
    -> Input guardrails evaluation (core/src/model/mod.rs :: apply_guardrails)
    -> Executor (core/src/executor/chat_completion/)
      -> Routing strategy (core/src/routing/) — selects provider/model
      -> ModelInstance::invoke / ::stream (core/src/model/)
        -> Provider-specific implementation (openai, anthropic, gemini, bedrock, ollama, proxy)
    -> Output guardrails evaluation
    -> TracedModel records span with tracing + OpenTelemetry
  -> HTTP Response
```

**Key subsystems:**
- **Routing** (`core/src/routing/`): Metric-based routing strategies for intelligent model selection across providers.
- **MCP** (`core/src/model/mcp.rs`, `core/src/model/mcp_server/`): Model Context Protocol support via the `rmcp` crate.
- **Pricing** (`core/src/pricing/`): Cost calculation per model/provider for usage tracking.
- **Telemetry** (`core/src/telemetry/`): OpenTelemetry span collection; gateway exposes a gRPC OTLP receiver on port 4317.
- **Responses** (`core/src/responses/`): OpenAI Responses API support (work in progress).

## Code Style Guidelines

### Imports Organization (Recommended)

```rust
// 1. Standard library imports first
use std::collections::HashMap;
use std::sync::Arc;

// 2. External crate imports
use actix_web::HttpResponse;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info_span, Instrument};

// 3. Workspace/crate imports (use crate:: for local)
use crate::error::GatewayError;
use crate::GatewayResult;
```

### Naming Conventions

- **Types/Structs/Enums**: PascalCase (`GatewayError`, `ChatCompletionRequest`)
- **Functions/Methods**: snake_case (`init_completion_model_instance`, `find_model_by_full_name`)
- **Constants**: SCREAMING_SNAKE_CASE (`DEFAULT_MAX_RETRIES`, `SPAN_MODEL_CALL`)
- **Module names**: snake_case (`chat_completion`, `llm_gateway`)

### Error Handling

The project uses a layered error architecture:

1. **`ModelError`** (`core/src/model/error.rs`) — Provider-level errors with 16+ variants including `CredentialsError`, `StreamError`, `RequestFailed`, `ModelNotFound`, `ConfigurationError`, etc.
2. **`GatewayError`** (`core/src/error.rs`) — Core-level errors. Wraps `ModelError` (boxed), `GuardError`, `McpServerError`, IO/parse errors, etc.
3. **`GatewayApiError`** (`core/src/lib.rs`) — API handler-level errors. Wraps `GatewayError` and adds `TokenUsageLimit`, `RouteError`, `CostCalculatorError`, etc. Implements `actix_web::ResponseError`.
4. **`GatewayResult<T>`** — Type alias: `Result<T, GatewayError>`.

```rust
// Provider-level error example (core/src/model/error.rs)
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model {0} not found")]
    ModelNotFound(String),

    #[error("Credentials for '{0}' are invalid or missing")]
    CredentialsError(String),

    #[error(transparent)]
    OpenAIApi(#[from] OpenAIError),

    #[error("Request failed: {0}")]
    RequestFailed(String),

    // ... 16+ variants total
}

// Core-level error (core/src/error.rs)
#[derive(Error, Debug)]
pub enum GatewayError {
    #[error(transparent)]
    ModelError(#[from] Box<ModelError>),  // Note: boxed to avoid large enum size

    #[error(transparent)]
    GuardError(#[from] GuardError),

    // ...
}

// Type alias
pub type GatewayResult<T> = Result<T, GatewayError>;
```

Key patterns:
- Use `#[error(transparent)]` for wrapping external errors
- Box large error types (e.g., `ModelError`, `McpServerError`) to avoid enum size issues
- Implement `From<T>` manually when boxing is needed

### Async Patterns

```rust
// Use async_trait for trait methods
#[async_trait]
pub trait ModelInstance: Sync + Send {
    async fn invoke(
        &self,
        input_vars: HashMap<String, Value>,
        tx: tokio::sync::mpsc::Sender<Option<ModelEvent>>,
        previous_messages: Vec<Message>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<ChatCompletionMessage>;

    async fn stream(
        &self,
        input_vars: HashMap<String, Value>,
        tx: mpsc::Sender<Option<ModelEvent>>,
        previous_messages: Vec<Message>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<()>;

    async fn embed(
        &self,
        _input: async_openai::types::EmbeddingInput,
    ) -> Result<async_openai::types::CreateEmbeddingResponse, ModelError> {
        unimplemented!("embed not implemented for this model");
    }
}

// Tests use tokio::test, not #[test]
#[tokio::test]
async fn test_guard_evaluation() {
    // test code
}
```

### Struct and Type Definitions

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GuardOrName {
    GuardId(String),
    GuardWithParameters(GuardWithParameters),
}
```

### Tracing and Logging

```rust
use tracing::{info_span, Instrument};

let span = info_span!(
    target: "langdb::user_tracing::models",
    SPAN_MODEL_CALL,
    input = &input_str,
    error = tracing::field::Empty,
);

async { /* code */ }.instrument(span.clone()).await;
```

## Important Notes

- **Rust Edition**: 2021
- **Required RUSTFLAGS**: `--cfg tracing_unstable --cfg aws_sdk_unstable` (must be set for build, test, and clippy)
- **Clippy**: Warnings are treated as errors in CI (`-D warnings`)
- **Formatting**: Use `cargo fmt` before committing
- **Tests**: Use `#[tokio::test]` for async tests
- **.env**: The gateway loads `.env` files via `dotenv` at startup

## Common Tasks

### Adding a new LLM provider

1. Create a new module in `core/src/model/` (e.g., `core/src/model/my_provider.rs`)
2. Implement the `ModelInstance` trait (must implement `invoke` and `stream`; optionally `embed`)
3. Add a new variant to `EngineType` enum in `core/src/types/engine.rs` and update its `Deserialize`, `Serialize`, `Display`, `FromStr` impls and `supports()` / `supported_features()` methods
4. Add provider-specific params struct (e.g., `MyProviderModelParams`) in `core/src/types/engine.rs`
5. Add a new variant to `CompletionEngineParams` enum in `core/src/types/engine.rs` and update its `engine_name()`, `provider_name()`, and `model_name()` methods
6. Update `init_completion_model_instance` in `core/src/model/mod.rs` to handle the new `CompletionEngineParams` variant
7. Update `credentials_identifier` in `core/src/model/mod.rs` to handle the new variant
8. Update `TraceModelDefinition::sanitize_json` in `core/src/model/mod.rs` to sanitize credentials for the new variant

### Adding a new API endpoint

1. Create handler in `core/src/handler/` (e.g., `core/src/handler/my_endpoint.rs`)
2. Register the module in `core/src/handler/mod.rs`
3. Add route in `gateway/src/http.rs` inside `ApiServer::attach_gateway_routes` (routes live under the `/v1` scope)
4. Update types in `core/src/types/` as needed

### Running the server locally

```bash
cp config.sample.yaml config.yaml
# Edit config.yaml to configure providers, cost limits, etc.
export RUSTFLAGS="--cfg tracing_unstable --cfg aws_sdk_unstable"
export RUST_LOG=debug
cargo run -- serve
```

### CLI Commands

The gateway binary supports the following subcommands:

```bash
cargo run -- serve              # Start the API server (default)
cargo run -- serve -i           # Start with interactive TUI mode
cargo run -- serve --host 0.0.0.0 --port 9090  # Custom host/port
cargo run -- list               # List all available models
cargo run -- update             # Update the models cache
cargo run -- update --force     # Force update
cargo run -- login              # Login to LangDB
```

### Configuration (`config.yaml`)

The configuration file supports the following top-level sections (see `config.sample.yaml` for reference):

```yaml
http:
  host: "0.0.0.0"
  port: 8080

# Optional: ClickHouse for telemetry persistence
# clickhouse:
#   url: http://localhost:8123

# Optional: Cost control limits (in dollars)
# cost_control:
#   daily: 10
#   monthly: 100
#   total: 1000

# Optional: Rate limiting (request counts)
# rate_limit:
#   hourly: 100
#   daily: 1000
#   monthly: 10000

# Optional: Provider API keys
# providers:
#   openai:
#     api_key: "{{ LANGDB_OPENAI_API_KEY }}"
#   anthropic:
#     api_key: "{{ LANGDB_ANTHROPIC_API_KEY }}"
#   gemini:
#     api_key: "{{ LANGDB_GEMINI_API_KEY }}"
#   bedrock:
#     api_key: "{{ LANGDB_BEDROCK_API_KEY }}"
#   deepseek:
#     api_key: "{{ LANGDB_DEEPSEEK_API_KEY }}"
```

Provider API keys can also be set via environment variables (loaded from `.env`).
