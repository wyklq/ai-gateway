use crate::model::error::ModelError;
use crate::model::types::{ModelEvent, ModelEventType};
use crate::model::ModelInstance;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::engine::ExecutionOptions;
use crate::types::engine::{OllamaModelParams, OllamaResponseFormat};
use crate::types::gateway::{
    ChatCompletionContent, ChatCompletionMessage, Usage, CompletionModelUsage,
};
use async_openai::types::{EmbeddingInput, CreateEmbeddingResponse};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, Url};
use serde::{Deserialize};
use serde_json::json;
use tokio::sync::mpsc::Sender;
use tracing::{error, Span, field};
use std::collections::HashMap;
use serde_json::Value;
use crate::types::threads::Message;
use crate::GatewayResult;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use opentelemetry::trace::TraceContextExt;

#[derive(Debug, Clone)]
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

        tracing::debug!(target: "ollama_debug", "[OllamaModel::new] endpoint = {:?}", endpoint);

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
        // Use the current span which should be properly set up from invoke
        let span = Span::current();
        
        // Log the request body for tracing
        span.record("request_body", &body.to_string());
        
        let response = self
            .client
            .post(url.clone())
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                span.record("error", &format!("Failed to send request: {}", e));
                error!("Failed to send request: {}", e);
                ModelError::RequestFailed(format!("Failed to send request: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            span.record("error", &format!("Request failed with status {}: {}", status, error_text));
            error!("Request failed with status {}: {}", status, error_text);
            return Err(ModelError::RequestFailed(format!("Request failed with status {}: {}", status, error_text)));
        }

        let json = response.json::<serde_json::Value>().await.map_err(|e| {
            span.record("error", &format!("Failed to parse response: {}", e));
            error!("Failed to parse response: {}", e);
            ModelError::ParsingResponseFailed(format!("Failed to parse response: {}", e))
        })?;

        span.record("output", &json.to_string());
        // Send the model event with the raw response
        let _ = tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmContent(crate::model::types::LLMContentEvent {
                content: json.to_string(),
            })
        ))).await;

        Ok(json)
    }

    async fn parse_chat_completion_response(
        &self,
        response: serde_json::Value,
    ) -> Result<ChatCompletionMessage, ModelError> {
        // 适配 OpenAI 风格的返回格式
        let message = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .ok_or_else(|| ModelError::ParsingResponseFailed("Missing choices[0].message".to_string()))?;

        let role = message.get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("assistant")
            .to_string();
        let content = message.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(ChatCompletionMessage::new_text(role, content))
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
        Usage::CompletionModelUsage(CompletionModelUsage {
            input_tokens: prompt_tokens.unwrap_or(0),
            output_tokens: completion_tokens.unwrap_or(0),
            total_tokens: prompt_tokens.unwrap_or(0) + completion_tokens.unwrap_or(0),
            prompt_tokens_details: None,
            completion_tokens_details: None,
            is_cache_used: false,
        })
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
                Some(ChatCompletionContent::Text(text)) => text.clone(),
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
        input_vars: HashMap<String, Value>,
        tx: Sender<Option<ModelEvent>>,
        previous_messages: Vec<Message>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<ChatCompletionMessage> {
        // 仅支持文本消息，转换 Message 为 ChatCompletionMessage
        let messages: Vec<ChatCompletionMessage> = previous_messages.iter().map(|m| {
            ChatCompletionMessage::new_text(
                m.r#type.to_string(),
                m.content.clone().unwrap_or_default(),
            )
        }).collect();
        
        // Create a span specifically for this request
        let span = tracing::info_span!(
            "ollama_chat",
            provider = "ollama",
            model = self.params.model.clone().unwrap_or_default(),
            input = field::Empty,
            output = field::Empty,
            error = field::Empty,
            usage = field::Empty,
            tags = ?tags
        );
        let _enter = span.enter();
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] span trace_id = {:?}, span_id = {:?}", span.context().span().span_context().trace_id(), span.context().span().span_context().span_id());
        // Send LlmStart event to properly initialize trace context
        let _ = tx.send(Some(ModelEvent::new(
            &tracing::Span::current(),
            ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                provider_name: "ollama".to_string(),
                model_name: self.params.model.clone().unwrap_or_default(),
                input: serde_json::to_string(&messages).unwrap_or_default(),
            })
        ))).await;
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] Sent LlmStart event");
        // 打印 endpoint 相关 debug 信息
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] self.endpoint = {:?}, self.params.model = {:?}", self.endpoint, self.params.model);
        let base_url = self.get_base_url()?;
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] base_url = {}", base_url);
        // NOTE: Url::join 而非 String.join, chat 和 /chat 的处理不同，涉及到是否保留 /v1
        let url = base_url.join("/v1/chat/completions").map_err(|e| {
            ModelError::ConfigurationError(format!("Failed to construct Ollama API URL: {}", e))
        })?;
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] final url = {}", url);
        let request_body = self.build_chat_request(&messages);
        // Send request and record the span event with proper trace context
        let response = self.send_request(url, request_body, &tx).await?;
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] Got response: {:?}", response);
        let message = self.parse_chat_completion_response(response.clone()).await?;
        // Record the response in the span
        span.record("output", &format!("{:?}", message));
        let prompt_length: u32 = messages.iter().map(|m| {
            m.content.as_ref().and_then(|c| c.as_string()).map_or(0, |t| t.len() as u32)
        }).sum();
        let completion_length = message.content.as_ref().and_then(|c| c.as_string()).map_or(0, |t| t.len() as u32);
        let prompt_tokens = Some(prompt_length / 4);
        let completion_tokens = Some(completion_length / 4);
        let usage = self.calculate_usage(prompt_tokens, completion_tokens);
        // Update usage in span
        span.record("usage", &format!("{:?}", usage));
        let credentials_ident = if self.credentials.is_none() {
            crate::model::CredentialsIdent::Langdb
        } else {
            crate::model::CredentialsIdent::Own
        };
        // 发送 LlmStop 事件
        let _ = tx.send(Some(ModelEvent::new(
            &tracing::Span::current(),
            ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                provider_name: "ollama".to_string(),
                model_name: self.params.model.clone().unwrap_or_default(),
                output: message.content.as_ref().and_then(|c| c.as_string()),
                usage: Some(match usage {
                    Usage::CompletionModelUsage(u) => u,
                    _ => Default::default(),
                }),
                finish_reason: crate::model::types::ModelFinishReason::Stop,
                tool_calls: vec![],
                credentials_ident,
            })
        ))).await;
        tracing::debug!(target: "ollama_debug", "[OllamaModel::invoke] Sent LlmStop event");
        Ok(message)
    }

    async fn stream(
        &self,
        input_vars: HashMap<String, Value>,
        tx: Sender<Option<ModelEvent>>,
        previous_messages: Vec<Message>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<()> {
        // Create a span specifically for this stream request
        let span = tracing::info_span!(
            "ollama_chat_stream",
            provider = "ollama",
            model = self.params.model.clone().unwrap_or_default(),
            input = field::Empty,
            error = field::Empty,
            tags = ?tags
        );
        let _guard = span.enter();
        // Send LlmStart event to properly initialize trace context
        let _ = tx.send(Some(ModelEvent::new(
            &tracing::Span::current(),
            ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                provider_name: "ollama".to_string(),
                model_name: self.params.model.clone().unwrap_or_default(),
                input: serde_json::to_string(&previous_messages).unwrap_or_default(),
            })
        ))).await;
        // For now, streaming is not implemented
        span.record("error", &"Ollama streaming not implemented");
        Err(ModelError::RequestFailed("Ollama stream not implemented".to_string()).into())
    }

    async fn embed(
        &self,
        _input: EmbeddingInput,
    ) -> Result<CreateEmbeddingResponse, ModelError> {
        unimplemented!("embed not implemented for OllamaModel yet");
    }
}
