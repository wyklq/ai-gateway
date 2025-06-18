use crate::events::JsonValue;
use crate::model::error::ModelError;
use crate::model::types::{ModelEvent, ModelEventType};
use crate::GatewayResult;
use async_openai::types::{CreateEmbeddingResponse, Embedding, EmbeddingInput, Usage};
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::{Client, Url};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{error, info_span};
use tracing_futures::Instrument;

use super::Embed;

#[derive(Clone)]
pub struct OllamaEmbed {
    pub model: String,
    pub client: Client,
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
}

impl OllamaEmbed {
    pub fn new(
        model: String,
        endpoint: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            model,
            client: Client::new(),
            endpoint,
            api_key,
        }
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        
        // Add API key if provided
        if let Some(api_key) = &self.api_key {
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
            );
        }
        
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        
        headers
    }

    fn get_base_url(&self) -> Result<Url, ModelError> {
        let base_url = match &self.endpoint {
            Some(endpoint) => endpoint.clone(),
            None => "http://localhost:11434".to_string(),
        };

        Url::parse(&base_url).map_err(|e| {
            ModelError::ConfigurationError(format!("Failed to parse Ollama endpoint URL: {}", e))
        })
    }
}

#[async_trait::async_trait]
impl Embed for OllamaEmbed {
    async fn invoke(
        &self,
        input_text: EmbeddingInput,
        tx: Option<tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<CreateEmbeddingResponse> {
        let call_span = info_span!(
            target: "langdb::user_tracing::models::ollama::embeddings",
            "ollama_embed",
            input = tracing::field::Empty,
            output = tracing::field::Empty,
            error = tracing::field::Empty,
            usage = tracing::field::Empty,
            ttft = tracing::field::Empty,
        );

        async move {
            let base_url = self.get_base_url()?;
            let url = base_url.join("api/embeddings").map_err(|e| {
                ModelError::ConfigurationError(format!("Failed to construct Ollama API URL: {}", e))
            })?;
            
            let headers = self.build_headers();
            
            // Convert input to string
            let input_str = match input_text {
                EmbeddingInput::String(s) => s,
                EmbeddingInput::StringArray(arr) => arr.join(" "),
                EmbeddingInput::TokenArray(_) => {
                    return Err(ModelError::InvalidRequest("Token array not supported for Ollama embeddings".to_string()).into());
                }
            };
            
            // Build request payload
            let request_body = json!({
                "model": self.model,
                "prompt": input_str,
            });
            
            // Send the request
            let response = self
                .client
                .post(url)
                .headers(headers)
                .json(&request_body)
                .send()
                .await
                .map_err(|e| {
                    error!("Failed to send request: {}", e);
                    ModelError::RequestFailed(format!("Failed to send request: {}", e))
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                error!("Request failed with status {}: {}", status, error_text);
                return Err(ModelError::RequestFailed(format!("Request failed with status {}: {}", status, error_text)).into());
            }

            let json = response.json::<serde_json::Value>().await.map_err(|e| {
                error!("Failed to parse response: {}", e);
                ModelError::ParsingResponseFailed(format!("Failed to parse response: {}", e))
            })?;
            
            // Send the model event with the raw response if tx is provided
            if let Some(tx) = &tx {
                tx.send(Some(ModelEvent {
                    model_type: ModelEventType::Data,
                    data: JsonValue::Object(json.clone()),
                }))
                .await
                .map_err(|_| ModelError::SendingResultsFailed("Failed to send model event".to_string()))?;
            }

            // Parse the embedding response
            #[derive(Deserialize)]
            struct OllamaEmbeddingResponse {
                embedding: Vec<f32>,
            }

            let response_obj = serde_json::from_value::<OllamaEmbeddingResponse>(json)
                .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama embedding response: {}", e)))?;

            // Estimate token usage
            let input_len = input_str.len() as u32;
            let token_count = input_len / 4; // Very rough token estimation
            
            // Create a CreateEmbeddingResponse compatible with OpenAI's format
            let embedding_response = CreateEmbeddingResponse {
                data: vec![Embedding {
                    embedding: response_obj.embedding,
                    index: 0,
                    object: "embedding".to_string(),
                }],
                model: self.model.clone(),
                object: "list".to_string(),
                usage: Usage {
                    prompt_tokens: token_count,
                    total_tokens: token_count,
                    completion_tokens: 0,
                },
            };
            
            Ok(embedding_response)
        }
        .instrument(call_span)
        .await
    }

    async fn batched_invoke(
        &self,
        inputs: impl Stream<Item = GatewayResult<(String, Vec<serde_json::Value>)>>,
    ) -> impl Stream<Item = GatewayResult<Vec<(Vec<f32>, Vec<serde_json::Value>)>>> {
        inputs
            .map(|input_result| async {
                let (text, metadata) = input_result?;
                let embedding_input = EmbeddingInput::String(text);
                let response = self.invoke(embedding_input, None).await?;
                let embedding = response.data.first().map(|e| e.embedding.clone()).unwrap_or_default();
                Ok(vec![(embedding, metadata)])
            })
            .buffer_unordered(10) // Process up to 10 embeddings concurrently
    }
}
