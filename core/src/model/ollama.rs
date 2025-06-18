use crate::events::JsonValue;
use crate::model::error::ModelError;
use crate::model::types::{ModelEvent, ModelEventType};
use crate::model::ModelInstance;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::engine::ExecutionOptions;
use crate::types::engine::{OllamaModelParams, OllamaResponseFormat};
use crate::types::gateway::{
    ChatCompletionContent, ChatCompletionMessage, ContentType, Usage,
};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::mpsc::Sender;
use tracing::{debug, error, info};

#[derive(Debug)]
pub struct OllamaModel {
    pub client: Client,
    pub credentials: Option<ApiKeyCredentials>,
    pub execution_options: ExecutionOptions,
    pub params: OllamaModelParams,
    pub endpoint: Option<String>,
}

impl OllamaModel {
    pub fn new(
        params: OllamaModelParams,
        execution_options: ExecutionOptions,
        credentials: Option<ApiKeyCredentials>,
        endpoint: Option<String>,
    ) -> Self {
        let client = Client::new();

        Self {
            client,
            credentials,
            execution_options,
            params,
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

    async fn send_request(
        &self, 
        url: Url, 
        body: serde_json::Value, 
        tx: &Sender<Option<ModelEvent>>
    ) -> Result<serde_json::Value, ModelError> {
        let headers = self.build_headers();
        
        let response = self
            .client
            .post(url)
            .headers(headers)
            .json(&body)
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

        Ok(json)
    }

    async fn parse_chat_completion_response(
        &self,
        response: serde_json::Value,
    ) -> Result<ChatCompletionContent, ModelError> {
        #[derive(Deserialize)]
        struct OllamaResponse {
            model: String,
            created_at: Option<String>,
            message: OllamaMessage,
            done: bool,
            total_duration: Option<u64>,
            load_duration: Option<u64>,
            prompt_eval_count: Option<u64>,
            prompt_eval_duration: Option<u64>,
            eval_count: Option<u64>,
            eval_duration: Option<u64>,
        }

        #[derive(Deserialize)]
        struct OllamaMessage {
            role: String,
            content: String,
        }

        let response_obj = serde_json::from_value::<OllamaResponse>(response.clone())
            .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama response: {}", e)))?;

        // Extract and convert the text content
        let content = ChatCompletionContent {
            content_type: ContentType::Text,
            text: Some(response_obj.message.content),
            ..Default::default()
        };

        Ok(content)
    }
    
    async fn parse_embedding_response(
        &self,
        response: serde_json::Value,
    ) -> Result<Vec<f32>, ModelError> {
        #[derive(Deserialize)]
        struct OllamaEmbeddingResponse {
            embedding: Vec<f32>,
        }

        let response_obj = serde_json::from_value::<OllamaEmbeddingResponse>(response)
            .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama embedding response: {}", e)))?;

        Ok(response_obj.embedding)
    }

    async fn parse_image_generation_response(
        &self,
        response: serde_json::Value,
    ) -> Result<Vec<String>, ModelError> {
        #[derive(Deserialize)]
        struct OllamaImageResponse {
            images: Vec<String>,
        }

        let response_obj = serde_json::from_value::<OllamaImageResponse>(response)
            .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama image response: {}", e)))?;

        Ok(response_obj.images)
    }

    fn calculate_usage(&self, prompt_tokens: Option<u32>, completion_tokens: Option<u32>) -> Usage {
        Usage {
            prompt_tokens: prompt_tokens.unwrap_or(0),
            completion_tokens: completion_tokens.unwrap_or(0),
            total_tokens: prompt_tokens.unwrap_or(0) + completion_tokens.unwrap_or(0),
        }
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

    fn build_chat_request(&self, messages: &[ChatCompletionMessage]) -> serde_json::Value {
        // Format messages for Ollama API format
        let formatted_messages: Vec<serde_json::Value> = messages.iter().map(|msg| {
            let content = match &msg.content {
                Some(ChatCompletionContent { text: Some(text), .. }) => text.clone(),
                _ => String::new(),
            };

            json!({
                "role": msg.role.to_string(),
                "content": content
            })
        }).collect();
        
        // Build request payload
        let mut request = json!({
            "model": self.params.model.clone().unwrap_or_default(),
            "messages": formatted_messages,
            "stream": false,
        });
        
        // Add optional parameters if specified
        if let Some(temp) = self.params.temperature {
            request["temperature"] = json!(temp);
        }
        
        if let Some(tokens) = self.params.max_tokens {
            request["max_tokens"] = json!(tokens);
        }

        if let Some(top_p) = self.params.top_p {
            request["top_p"] = json!(top_p);
        }
        
        if let Some(stop) = &self.params.stop {
            request["stop"] = json!(stop);
        }
        
        if let Some(format) = &self.params.response_format {
            match format {
                OllamaResponseFormat::Json => {
                    request["format"] = json!("json");
                }
            }
        }
        
        request
    }

    fn build_embedding_request(&self, input: &str) -> serde_json::Value {
        json!({
            "model": self.params.model.clone().unwrap_or_default(),
            "prompt": input,
        })
    }

    fn build_image_request(&self, prompt: &str) -> serde_json::Value {
        json!({
            "model": self.params.model.clone().unwrap_or_default(),
            "prompt": prompt,
        })
    }
}

#[async_trait]
impl ModelInstance for OllamaModel {
    async fn invoke(
        &self,
        messages: &[ChatCompletionMessage],
        tx: &Sender<Option<ModelEvent>>,
    ) -> Result<(ChatCompletionContent, Usage), ModelError> {
        let base_url = self.get_base_url()?;
        let url = base_url.join("api/chat").map_err(|e| {
            ModelError::ConfigurationError(format!("Failed to construct Ollama API URL: {}", e))
        })?;
        
        let request_body = self.build_chat_request(messages);
        
        // Send the request
        let response = self.send_request(url, request_body, tx).await?;
        
        // Parse the response
        let content = self.parse_chat_completion_response(response.clone()).await?;
        
        // Calculate usage (estimated)
        // Ollama doesn't provide token counts in its response, so we're making an estimation
        let prompt_length: u32 = messages.iter().map(|m| {
            match &m.content {
                Some(ChatCompletionContent { text: Some(t), .. }) => t.len() as u32,
                _ => 0,
            }
        }).sum();
        
        let completion_length = content.text.as_ref().map_or(0, |t| t.len() as u32);
        
        // Rough estimation: ~4 characters per token
        let prompt_tokens = Some(prompt_length / 4);
        let completion_tokens = Some(completion_length / 4);
        
        let usage = self.calculate_usage(prompt_tokens, completion_tokens);
        
        Ok((content, usage))
    }

    async fn embed(
        &self,
        text: &str,
        tx: &Sender<Option<ModelEvent>>,
    ) -> Result<(Vec<f32>, Usage), ModelError> {
        let base_url = self.get_base_url()?;
        let url = base_url.join("api/embeddings").map_err(|e| {
            ModelError::ConfigurationError(format!("Failed to construct Ollama embeddings API URL: {}", e))
        })?;
        
        let request_body = self.build_embedding_request(text);
        
        // Send the request
        let response = self.send_request(url, request_body, tx).await?;
        
        // Parse the response
        let embeddings = self.parse_embedding_response(response).await?;
        
        // Calculate usage (estimated)
        let text_length = text.len() as u32;
        // Rough estimation: ~4 characters per token
        let prompt_tokens = Some(text_length / 4);
        
        let usage = self.calculate_usage(prompt_tokens, None);
        
        Ok((embeddings, usage))
    }

    async fn generate_image(
        &self,
        prompt: &str,
        tx: &Sender<Option<ModelEvent>>,
    ) -> Result<(Vec<String>, Usage), ModelError> {
        let base_url = self.get_base_url()?;
        let url = base_url.join("api/generate").map_err(|e| {
            ModelError::ConfigurationError(format!("Failed to construct Ollama image generation API URL: {}", e))
        })?;
        
        let request_body = self.build_image_request(prompt);
        
        // Send the request
        let response = self.send_request(url, request_body, tx).await?;
        
        // Parse the response
        let images = self.parse_image_generation_response(response).await?;
        
        // Calculate usage (estimated)
        let prompt_length = prompt.len() as u32;
        // Rough estimation: ~4 characters per token
        let prompt_tokens = Some(prompt_length / 4);
        
        let usage = self.calculate_usage(prompt_tokens, None);
        
        Ok((images, usage))
    }
}
