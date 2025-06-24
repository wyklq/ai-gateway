use std::collections::HashMap;

use crate::embed_mod::Embed;
use crate::embed_mod::OpenAIEmbed;
use crate::embed_mod_ollama::OllamaEmbed;
use crate::error::GatewayError;
use crate::model::types::ModelEvent;
use crate::models::ModelMetadata;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::credentials::Credentials;
use actix_web::HttpRequest;
use tracing::Span;

use crate::types::embed::OpenAiEmbeddingParams;
use crate::types::gateway::{CreateEmbeddingRequest, CreateEmbeddingResponse};
use crate::types::{
    engine::{ExecutionOptions, Model, ModelTools, ModelType, OllamaModelParams},
};
use tracing_futures::Instrument;

use crate::handler::{CallbackHandlerFn, ModelEventWithDetails};

use super::get_key_credentials;
use super::ProvidersConfig;
use crate::types::provider::InferenceModelProvider;

pub async fn handle_embeddings_invoke(
    mut request: CreateEmbeddingRequest,
    callback_handler: &CallbackHandlerFn,
    llm_model: &ModelMetadata,
    key_credentials: Option<&Credentials>,
    req: HttpRequest,
    tags: HashMap<String, String>,
) -> Result<CreateEmbeddingResponse, GatewayError> {
    // 从 tags 获取 tenant_id
    let tenant_id = tags.get("tenant_id").cloned().unwrap_or_else(|| "unknown".to_string());
    let span = Span::current();
    span.record("tenant_id", &tenant_id);
    request.model = llm_model.inference_provider.model_name.clone();

    let params = OpenAiEmbeddingParams {
        model: Some(llm_model.model.clone()),
        dimensions: request.dimensions,
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Option<ModelEvent>>(1000);
    let model_name = llm_model.model.clone();

    let callback_handler = callback_handler.clone();
    let provider_name = llm_model.inference_provider.provider.to_string();
    let span = Span::current(); // 确保 span 被 clone
    
    tokio::spawn(async move {
        while let Some(Some(msg)) = rx.recv().await {
            callback_handler.on_message(ModelEventWithDetails::new(
                msg,
                Model {
                    name: model_name.clone(),
                    description: None,
                    provider_name: provider_name.clone(),
                    prompt_name: None,
                    model_params: HashMap::new(),
                    execution_options: ExecutionOptions::default(),
                    tools: ModelTools(vec![]),
                    model_type: ModelType::Embedding,
                    response_schema: None,
                    credentials: None,
                },
            ));
        }
    });

    let providers_config = req.app_data::<ProvidersConfig>().cloned();
    let mut custom_endpoint = llm_model.inference_provider.endpoint.clone();
    let key = match get_key_credentials(
        key_credentials,
        providers_config.as_ref(),
        &llm_model.inference_provider.provider.to_string(),
    ) {
        Some(Credentials::ApiKey(key)) => Some(key),
        Some(Credentials::ApiKeyWithEndpoint {
            api_key: key,
            endpoint,
        }) => {
            custom_endpoint = Some(endpoint);
            Some(ApiKeyCredentials { api_key: key })
        }
        _ => None,
    };

    let _provider_name = &llm_model.inference_provider.provider.to_string();
    // Provider selection: instantiate the correct Embed implementation
    let embed: Box<dyn Embed> = match llm_model.inference_provider.provider {
        InferenceModelProvider::Ollama => {
            // 直接用 OllamaModelParams 构造
            let params = OllamaModelParams {
                model: Some(llm_model.inference_provider.model_name.clone()),
                temperature: None,
                top_p: None,
                max_tokens: None,
                stop: None,
                response_format: None,
            };
            Box::new(OllamaEmbed::new(
                params,
                key.as_ref(),
                custom_endpoint.as_deref(),
            ))
        }
        _ => Box::new(OpenAIEmbed::new(params, key.as_ref(), custom_endpoint.as_deref())?)
    };
    
    // 调用 embedding API
    embed
        .invoke(request.input.clone(), Some(tx.clone()))
        .instrument(span.clone())
        .await
}
