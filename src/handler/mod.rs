pub mod chat;
pub mod embedding;
pub mod image;
pub mod models;

use crate::error::GatewayError;
use crate::model::types::ModelEvent;
use crate::models::LlmModelDefinition;
use crate::types::engine::Model;
use crate::GatewayApiError;
use actix_web::HttpRequest;
use std::collections::HashMap;

#[derive(Debug)]
pub struct AvailableModels(pub Vec<LlmModelDefinition>);

pub fn find_model_by_full_name(
    model_name: &str,
    provided_models: &AvailableModels,
) -> Result<LlmModelDefinition, GatewayApiError> {
    let model_parts = model_name.split('/').collect::<Vec<&str>>();

    let llm_model = if model_parts.len() == 1 {
        provided_models
            .0
            .iter()
            .find(|m| m.model.to_lowercase() == model_name.to_lowercase())
            .cloned()
    } else if model_parts.len() == 2 {
        let model_name = model_parts.last().expect("2 elements in model parts");
        let provided_by = model_parts.first().expect("2 elements in model parts");

        provided_models
            .0
            .iter()
            .find(|m| {
                m.model.to_lowercase() == model_name.to_lowercase()
                    && m.inference_provider.provider.to_string() == *provided_by
            })
            .cloned()
    } else {
        None
    };

    match llm_model {
        Some(model) => Ok(model),
        None => Err(GatewayApiError::CustomError(format!(
            "Model not found {model_name}"
        ))),
    }
}

// extract langdb-tags from headers, shoule be sth like this: tag1=value1&tag2=value2 => result should be a Map<String, String>
pub fn extract_tags(req: &HttpRequest) -> Result<HashMap<String, String>, GatewayError> {
    Ok(match req.headers().get("x-tags") {
        Some(value) => {
            let tags_str = value
                .to_str()
                .map_err(|e| GatewayError::CustomError(e.to_string()))?
                .to_string();
            let tags: HashMap<String, String> = tags_str
                .split('&')
                .map(|tag| {
                    let (k, v) = tag.split_once('=').unwrap();
                    (k.to_string(), v.to_string())
                })
                .collect();
            Some(tags)
        }
        None => None,
    }
    .unwrap_or_default())
}

pub fn record_map_err(
    e: impl Into<GatewayApiError> + ToString,
    span: tracing::Span,
) -> GatewayApiError {
    span.record("error", e.to_string());
    e.into()
}

#[derive(Clone, Default)]
pub struct CallbackHandlerFn(pub Option<tokio::sync::broadcast::Sender<ModelEventWithDetails>>);

impl CallbackHandlerFn {
    pub fn on_message(&self, message: ModelEventWithDetails) {
        if let Some(sender) = self.0.clone() {
            let _ = sender.send(message);
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelEventWithDetails {
    pub event: ModelEvent,
    pub model: Model,
}

impl ModelEventWithDetails {
    pub fn new(event: ModelEvent, model: Model) -> Self {
        Self { event, model }
    }
}
