use std::collections::HashMap;

use crate::embed_mod::Embed;
use crate::embed_mod::OpenAIEmbed;
use crate::error::GatewayError;
use crate::model::types::ModelEvent;
use crate::models::ModelMetadata;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::credentials::Credentials;
use actix_web::HttpRequest;
use async_openai::types::EmbeddingInput;
use tracing::Span;

use crate::types::embed::OpenAiEmbeddingParams;
use crate::types::{
    engine::{ExecutionOptions, Model, ModelTools, ModelType},
    gateway::{CreateEmbeddingRequest, Input},
};
use tracing_futures::Instrument;

use crate::handler::{CallbackHandlerFn, ModelEventWithDetails};

use super::get_key_credentials;
use super::ProvidersConfig;

pub async fn handle_embeddings_invoke(
    mut request: CreateEmbeddingRequest,
    callback_handler: &CallbackHandlerFn,
    llm_model: &ModelMetadata,
    key_credentials: Option<&Credentials>,
    req: HttpRequest,
) -> Result<async_openai::types::CreateEmbeddingResponse, GatewayError> {
    let span = Span::current();
    request.model = llm_model.inference_provider.model_name.clone();

    let params = OpenAiEmbeddingParams {
        model: Some(llm_model.model.clone()),
        dimensions: request.dimensions,
    };

    let input: EmbeddingInput = match &request.input {
        Input::String(s) => s.into(),
        Input::Array(vec) => vec.into(),
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Option<ModelEvent>>(1000);
    let model_name = llm_model.model.clone();

    let callback_handler = callback_handler.clone();
    let provider_name = llm_model.inference_provider.provider.to_string();
    
    tokio::spawn(async move {
        while let Some(Some(msg)) = rx.recv().await {
            callback_handler.on_message(ModelEventWithDetails::new(
                msg,
                Model {
                    name: model_name.clone(),
                    description: None,
                    provider_name,
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
    let mut custom_endpoint = None;
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

    let provider_name = &llm_model.inference_provider.provider.to_string();
    let embed: Box<dyn Embed> = match llm_model.inference_provider.provider {
        InferenceModelProvider::Ollama => {
            let api_key = key.as_ref().map(|k| k.api_key.clone());
            Box::new(crate::embed_mod::ollama::OllamaEmbed::new(
                llm_model.model.clone(),
                custom_endpoint,
                api_key,
            ))
        }
        _ => Box::new(OpenAIEmbed::new(params, key.as_ref(), custom_endpoint.as_deref())?)
    };
    
    embed
        .invoke(input, Some(tx.clone()))
        .instrument(span.clone())
        .await
}
