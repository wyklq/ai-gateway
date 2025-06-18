use std::collections::HashMap;
use reqwest::{Client, Url};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize};
use serde_json::json;
use tokio::sync::mpsc::Sender;
use tracing::{error, info_span};
use tracing_futures::Instrument;

use crate::events::{JsonValue, SPAN_MODEL_CALL};
use crate::model::error::ModelError;
use crate::model::image_generation::ImageGenerationModelInstance;
use crate::model::types::{ModelEvent, ModelEventType};
use crate::types::credentials::ApiKeyCredentials;
use crate::types::gateway::{CreateImageRequest, ImageGenerationModelUsage};
use crate::types::image::ImagesResponse;
use crate::GatewayResult;

pub struct OllamaImageGeneration {
    model_name: String,
    client: Client,
    credentials: Option<ApiKeyCredentials>,
    endpoint: Option<String>,
}

impl OllamaImageGeneration {
    pub fn new(
        model_name: String, 
        credentials: Option<ApiKeyCredentials>,
        endpoint: Option<String>,
    ) -> Self {
        Self {
            model_name,
            client: Client::new(),
            credentials,
            endpoint,
        }
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        
        // Add API key if provided
        if let Some(creds) = &self.credentials {
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&format!("Bearer {}", creds.api_key)).unwrap(),
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
impl ImageGenerationModelInstance for OllamaImageGeneration {
    async fn create_new(
        &self,
        request: &CreateImageRequest,
        tx: Sender<Option<ModelEvent>>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<ImagesResponse> {
        let call_span = info_span!(
            target: "langdb::user_tracing::models::ollama::image_generation",
            SPAN_MODEL_CALL,
            input = request.prompt,
            output = tracing::field::Empty,
            error = tracing::field::Empty,
            usage = tracing::field::Empty,
            ttft = tracing::field::Empty,
            tags = JsonValue(&serde_json::to_value(tags).unwrap_or_default()).as_value()
        );

        let response = async {
            let base_url = self.get_base_url()?;
            let url = base_url.join("api/generate").map_err(|e| {
                ModelError::ConfigurationError(format!("Failed to construct Ollama API URL: {}", e))
            })?;
            
            let headers = self.build_headers();
            
            // Build request payload
            let request_body = json!({
                "model": self.model_name,
                "prompt": request.prompt,
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
                return Err(ModelError::RequestFailed(format!("Request failed with status {}: {}", status, error_text)));
            }

            let json = response.json::<serde_json::Value>().await.map_err(|e| {
                error!("Failed to parse response: {}", e);
                ModelError::ParsingResponseFailed(format!("Failed to parse response: {}", e))
            })?;
            
            // Send the model event with the raw response
            tx.send(Some(ModelEvent {
                model_type: ModelEventType::Data,
                data: JsonValue::Object(json.clone()),
            }))
            .await
            .map_err(|_| ModelError::SendingResultsFailed("Failed to send model event".to_string()))?;

            // Parse the image response
            #[derive(Deserialize)]
            struct OllamaImageResponse {
                images: Vec<String>,
            }

            let response_obj = serde_json::from_value::<OllamaImageResponse>(json)
                .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama image response: {}", e)))?;

            let images = ImagesResponse {
                created: chrono::Utc::now().timestamp(),
                // data: response_obj.images.into_iter()... TODO later
                model: self.model_name.clone(),
                usage: Some(ImageGenerationModelUsage {
                    prompt_tokens: request.prompt.len() as u32 / 4, // Very rough token estimation
                }),
            };

            // Send completion event
            tx.send(Some(ModelEvent {
                model_type: ModelEventType::ImageGenerationFinish(json!({
                    "model": self.model_name,
                    "created": images.created,
                    "usage": images.usage,
                })),
                data: Default::default(),
            }))
            .await
            .map_err(|_| ModelError::SendingResultsFailed("Failed to send completion event".to_string()))?;

            Ok(images)
        }
        .instrument(call_span.clone())
        .await;

        response.map_err(|e| e.into())
    }
}
