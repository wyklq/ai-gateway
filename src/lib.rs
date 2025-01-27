pub mod embed_mod;
pub mod error;
pub mod events;
pub mod executor;
pub mod handler;
pub mod llm_gateway;
pub mod model;
pub mod models;
pub mod otel;
pub mod types;

use crate::error::GatewayError;
use crate::types::gateway::CostCalculatorError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GatewayApiError {
    #[error("Failed to parse JSON")]
    JsonParseError(#[from] serde_json::Error),

    #[error(transparent)]
    GatewayError(#[from] GatewayError),

    #[error("{0}")]
    CustomError(String),

    #[error(transparent)]
    CostCalculatorError(#[from] CostCalculatorError),

    #[error(transparent)]
    ModelError(#[from] model::error::ModelError),
}
