use crate::model::error::ModelError;
use crate::model::types::{LLMFirstToken, ModelEvent, ModelEventType};
use crate::model::ModelInstance;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::engine::ExecutionOptions;
use crate::types::engine::{OllamaModelParams, OllamaResponseFormat};
use crate::types::gateway::{
    ChatCompletionContent, ChatCompletionMessage, Usage, CompletionModelUsage,
};
use async_openai::types::{EmbeddingInput, CreateEmbeddingResponse};
use async_trait::async_trait;
use futures_util::stream::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, Url};
use serde_json::json;
use tokio::sync::mpsc::Sender;
use tracing::{error, Span, field};
use std::collections::HashMap;
use serde_json::Value;
use crate::types::threads::Message;
use crate::GatewayResult;
use tracing::Instrument;
use valuable::Valuable;

macro_rules! target {
    () => {
        "langdb::user_tracing::models::ollama"
    };
    ($subtgt:literal) => {
        concat!("langdb::user_tracing::models::ollama::", $subtgt)
    };
}

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

    // Add a helper method to validate model name
    fn validate_model(&self) -> Result<String, ModelError> {
        match &self.params.model {
            Some(model_name) if !model_name.trim().is_empty() => Ok(model_name.clone()),
            _ => Err(ModelError::ModelNotFound("Model name is not specified or empty".to_string())),
        }
    }

    pub fn get_model_name(&self) -> String {
        self.validate_model().unwrap_or_else(|_| "".to_string())
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
        // Use the current span which is already set up from the caller via .instrument()
        let span = Span::current();
        
        let response = self
            .client
            .post(url.clone())
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let error_msg = format!("Failed to send request: {}", e);
                span.record("error", &error_msg);
                error!("{}", error_msg);
                ModelError::RequestFailed(error_msg)
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            let error_msg = format!("Request failed with status {}: {}", status, error_text);
            span.record("error", &error_msg);

            tx.send(Some(ModelEvent::new(
                &span,
                ModelEventType::LlmContent(crate::model::types::LLMContentEvent {
                    content: error_msg.clone(),
                })
            ))).await
                .map_err(|e| ModelError::CustomError(e.to_string()))?;

            error!("{}", error_msg);
            return Err(ModelError::RequestFailed(error_msg));
        }

        let json = response.json::<serde_json::Value>().await.map_err(|e| {
            let error_msg = format!("Failed to parse response: {}", e);
            span.record("error", &error_msg);
            error!("{}", error_msg);
            ModelError::ParsingResponseFailed(error_msg)
        })?;

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
        // 兼容 OpenAI embedding 返回格式：{"object":"list","data":[{"object":"embedding","embedding":[...],"index":0}],...}
        let embedding = response
            .get("data")
            .and_then(|data| data.get(0))
            .and_then(|item| item.get("embedding"))
            .and_then(|emb| emb.as_array())
            .ok_or_else(|| ModelError::ParsingResponseFailed("Missing data[0].embedding in embedding response".to_string()))?;
        let embedding_vec: Result<Vec<f32>, _> = embedding.iter().map(|v| v.as_f64().map(|f| f as f32).ok_or(())).collect();
        embedding_vec.map_err(|_| ModelError::ParsingResponseFailed("Embedding array contains non-float values".to_string()))
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

    fn build_chat_request(&self, messages: &[ChatCompletionMessage], model_name: &str, stream: bool) -> serde_json::Value {
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
            "model": model_name,
            "messages": formatted_messages,
            "stream": stream,
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

    fn build_embedding_request(&self, input: &str, model_name: &str) -> serde_json::Value {
        // Format messages for OpenAI compatible Ollama API format
        json!({
            "input": input,
            "model": model_name,
        })
    }
}

#[async_trait]
impl ModelInstance for OllamaModel {
    async fn invoke(
        &self,
        _input_vars: HashMap<String, Value>,
        tx: Sender<Option<ModelEvent>>,
        previous_messages: Vec<Message>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<ChatCompletionMessage> {
        // 验证模型名称
        let model_name = match self.validate_model() {
            Ok(name) => name,
            Err(e) => {
                tracing::error!("Model validation failed: {:?}", e);
                return Err(e.into());
            }
        };

        // 仅支持文本消息，转换 Message 为 ChatCompletionMessage
        let messages: Vec<ChatCompletionMessage> = previous_messages.iter().map(|m| {
            ChatCompletionMessage::new_text(
                m.r#type.to_string(),
                m.content.clone().unwrap_or_default(),
            )
        }).collect();
        
        // Create a span specifically for this request - using target! pattern from openai.rs
        let input = serde_json::to_string(&messages).unwrap_or_default();
        let span = tracing::info_span!(
            target: target!("chat"),
            "model_call",
            provider = "ollama",
            model = model_name,
            input = input,
            output = field::Empty,
            error = field::Empty,
            usage = field::Empty,
            tags = crate::events::JsonValue(&serde_json::to_value(tags.clone()).unwrap_or_default()).as_value(),
        );
        
        // Send LlmStart event to properly initialize trace context - use span directly, not current()
        tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                provider_name: "ollama".to_string(),
                model_name: model_name.clone(),
                input,
            })
        ))).await
            .map_err(|e| crate::error::GatewayError::CustomError(e.to_string()))?;
        // 打印 endpoint 相关 debug 信息
        let base_url = self.get_base_url()?;
        // NOTE: Url::join 而非 String.join, chat 和 /chat 的处理不同，涉及到是否保留 /v1
        let url = base_url.join("/v1/chat/completions").map_err(|e| {
            let err_msg = format!("Failed to construct Ollama API URL: {}", e);
            span.record("error", &err_msg);
            ModelError::ConfigurationError(err_msg)
        })?;
        
        let request_body = self.build_chat_request(&messages, &model_name, false);

        // Send request and record the span event with proper trace context - Use .instrument()
        let response = async {
            self.send_request(url, request_body, &tx).await
        }
        .instrument(span.clone())
        .await?;
        let message = self.parse_chat_completion_response(response.clone()).await?;
        
        // Record the response in the span
        let output_str = serde_json::to_string(&message).unwrap_or_default();
        span.record("output", &output_str);
        
        let prompt_length: u32 = messages.iter().map(|m| {
            m.content.as_ref().and_then(|c| c.as_string()).map_or(0, |t| t.len() as u32)
        }).sum();
        let completion_length = message.content.as_ref().and_then(|c| c.as_string()).map_or(0, |t| t.len() as u32);
        let prompt_tokens = Some(prompt_length / 4);
        let completion_tokens = Some(completion_length / 4);
        let usage = self.calculate_usage(prompt_tokens, completion_tokens);
        
        // Update usage in span with proper format (matching openai.rs)
        if let Usage::CompletionModelUsage(ref u) = usage {
            span.record("usage", &format!("{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}", 
                u.input_tokens, u.output_tokens, u.total_tokens));
        }
        
        let credentials_ident = if self.credentials.is_none() {
            crate::model::CredentialsIdent::Langdb
        } else {
            crate::model::CredentialsIdent::Own
        };
        
        // 发送 LlmStop 事件 - use span directly, not current()
        tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                provider_name: "ollama".to_string(),
                model_name: model_name.clone(),
                output: message.content.as_ref().and_then(|c| c.as_string()),
                usage: Some(match usage {
                    Usage::CompletionModelUsage(u) => u,
                    _ => Default::default(),
                }),
                finish_reason: crate::model::types::ModelFinishReason::Stop,
                tool_calls: vec![],
                credentials_ident,
            })
        ))).await
            .map_err(|e| crate::error::GatewayError::CustomError(e.to_string()))?;
            
        Ok(message)
    }

    async fn stream(
        &self,
        _input_vars: HashMap<String, Value>,
        tx: Sender<Option<ModelEvent>>,
        previous_messages: Vec<Message>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<()> {
        let model_name = self.validate_model()?;
        let input = serde_json::to_string(&previous_messages).unwrap_or_default();
        let span = tracing::info_span!(
            target: target!("chat_stream"),
            "model_call_stream",
            provider = "ollama",
            model = model_name.clone(),
            input = input,
            error = field::Empty,
            tags = crate::events::JsonValue(&serde_json::to_value(tags.clone()).unwrap_or_default()).as_value()
        );

        let span_clone = span.clone();
        async move {
            tx.send(Some(ModelEvent::new(
                &span_clone,
                ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                    provider_name: "ollama".to_string(),
                    model_name: model_name.clone(),
                    input: serde_json::to_string(&previous_messages).unwrap_or_default(),
                }),
            )))
            .await
            .map_err(|e| ModelError::CustomError(e.to_string()))?;

            let base_url = self.get_base_url()?;
            let url = base_url.join("/v1/chat/completions").map_err(|e| {
                let err_msg = format!("Failed to construct Ollama API URL: {}", e);
                span_clone.record("error", &err_msg);
                ModelError::ConfigurationError(err_msg)
            })?;

            let messages: Vec<ChatCompletionMessage> = previous_messages
                .iter()
                .map(|m| {
                    ChatCompletionMessage::new_text(
                        m.r#type.to_string(),
                        m.content.clone().unwrap_or_default(),
                    )
                })
                .collect();

            let request_body = self.build_chat_request(&messages, &model_name, true);
            let headers = self.build_headers();

            let response = self
                .client
                .post(url)
                .headers(headers)
                .json(&request_body)
                .send()
                .await
                .map_err(|e| ModelError::RequestFailed(e.to_string()))?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let error_msg = format!("Request failed with status {}: {}", status, error_text);
                span_clone.record("error", &error_msg);

                let credentials_ident = if self.credentials.is_none() {
                    crate::model::CredentialsIdent::Langdb
                } else {
                    crate::model::CredentialsIdent::Own
                };

                tx.send(Some(ModelEvent::new(
                    &span_clone,
                    ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                        provider_name: "ollama".to_string(),
                        model_name: model_name.clone(),
                        output: Some(error_msg.clone()),
                        usage: None,
                        finish_reason: crate::model::types::ModelFinishReason::ContentFilter,
                        tool_calls: vec![],
                        credentials_ident,
                    }),
                )))
                .await
                .map_err(|e| ModelError::CustomError(e.to_string()))?;

                return Err(ModelError::RequestFailed(error_msg).into());
            }

            let mut stream = response.bytes_stream();
            let mut full_content = String::new();
            let mut finish_reason = crate::model::types::ModelFinishReason::Stop;
            let mut first_token_received = false;
            let mut done = false;

            while let Some(item) = stream.next().await {
                if done {
                    break;
                }
                let chunk = item.map_err(|e| ModelError::RequestFailed(format!("Stream error: {}", e)))?;
                let data = String::from_utf8_lossy(&chunk);
                for line in data.lines() {
                    if line.starts_with("data: ") {
                        let json_str = &line[6..];
                        if json_str.trim() == "[DONE]" {
                            done = true;
                            break;
                        }

                        let value: serde_json::Value = match serde_json::from_str(json_str) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };

                        if !first_token_received {
                            first_token_received = true;
                tx.send(Some(ModelEvent::new(
                &span_clone,
                ModelEventType::LlmFirstToken(
                                    LLMFirstToken {}
                                ),
                            )))
                            .await
                            .map_err(|e| ModelError::CustomError(e.to_string()))?;
                        }

                        if let Some(choices) = value.get("choices").and_then(|c| c.as_array()) {
                            if let Some(choice) = choices.get(0) {
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                        if !content.is_empty() {
                                            full_content.push_str(content);
                                            tx.send(Some(ModelEvent::new(
                                                &span_clone,
                                                ModelEventType::LlmContent(
                                                    crate::model::types::LLMContentEvent {
                                                        content: content.to_string(),
                                                    },
                                                ),
                                            )))
                                            .await
                                            .map_err(|e| ModelError::CustomError(e.to_string()))?;
                                        }
                                    }
                                }
                                if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                                    finish_reason = match reason {
                                        "stop" => crate::model::types::ModelFinishReason::Stop,
                                        "length" => crate::model::types::ModelFinishReason::Length,
                                        "content_filter" => crate::model::types::ModelFinishReason::ContentFilter,
                                        "tool_calls" => crate::model::types::ModelFinishReason::ToolCalls,
                                        _ => crate::model::types::ModelFinishReason::Stop,
                                    };
                                }
                            }
                        }
                    }
                }
            }

            let credentials_ident = if self.credentials.is_none() {
                crate::model::CredentialsIdent::Langdb
            } else {
                crate::model::CredentialsIdent::Own
            };

            let prompt_length: u32 = messages
                .iter()
                .map(|m| {
                    m.content
                        .as_ref()
                        .and_then(|c| c.as_string())
                        .map_or(0, |t| t.len() as u32)
                })
                .sum();
            let completion_length = full_content.len() as u32;
            let prompt_tokens = Some(prompt_length / 4);
            let completion_tokens = Some(completion_length / 4);
            let usage = self.calculate_usage(prompt_tokens, completion_tokens);

            tx.send(Some(ModelEvent::new(
                &span_clone,
                ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                    provider_name: "ollama".to_string(),
                    model_name: model_name.clone(),
                    output: Some(full_content),
                    usage: Some(match usage {
                        Usage::CompletionModelUsage(u) => u,
                        _ => Default::default(),
                    }),
                    finish_reason,
                    tool_calls: vec![],
                    credentials_ident,
                }),
            )))
            .await
            .map_err(|e| ModelError::CustomError(e.to_string()))?;

            Ok(())
        }
        .instrument(span.clone())
        .await
    }

    async fn embed(
        &self,
        input: EmbeddingInput,
    ) -> Result<CreateEmbeddingResponse, ModelError> {
        // 1. validate model
        let model_name = self.validate_model()?;
        // 2. 构造 tracing span
        let input_str = match &input {
            EmbeddingInput::String(s) => s.clone(),
            _ => {
                return Err(ModelError::ParsingResponseFailed(
                    "Ollama embedding only supports string input".to_string()
                ));
            }
        };
        let span = tracing::info_span!(
            target: target!("embed"),
            "model_embed",
            provider = "ollama",
            model = model_name,
            input = input_str,
            output = field::Empty,
            error = field::Empty,
        );
        // 3. base_url + /v1/embeddings
        let base_url = self.get_base_url()?;
        let url = base_url.join("/v1/embeddings").map_err(|e| {
            let err_msg = format!("Failed to construct Ollama embedding API URL: {}", e);
            span.record("error", &err_msg);
            ModelError::ConfigurationError(err_msg)
        })?;
        // 4. 构造 body
        let body = self.build_embedding_request(&input_str, &model_name);
        // 5. 发送请求
        let response = async {
            // embed 不需要 tx, 传一个 dummy channel
            let (dummy_tx, _rx) = tokio::sync::mpsc::channel(1);
            self.send_request(url, body, &dummy_tx).await
        }
        .instrument(span.clone())
        .await?;
        // 6. 解析 response
        let embedding = self.parse_embedding_response(response.clone()).await?;
        // 7. 构造 CreateEmbeddingResponse
        let data = vec![serde_json::json!({
            "object": "embedding",
            "index": 0,
            "embedding": embedding.clone(),
        })];
        // 8. 构造 usage，优先用后端真实 usage 字段
        let usage = if let Some(usage_val) = response.get("usage") {
            // 兼容 OpenAI usage 格式
            let prompt_tokens = usage_val.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let total_tokens = usage_val.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            async_openai::types::EmbeddingUsage {
                prompt_tokens,
                total_tokens,
            }
        } else {
            async_openai::types::EmbeddingUsage {
                prompt_tokens: embedding.len() as u32,
                total_tokens: embedding.len() as u32,
            }
        };
        // 9. 构造返回值，类型兼容
        let result = CreateEmbeddingResponse {
            object: "list".to_string(),
            data: serde_json::from_value(serde_json::json!(data)).unwrap_or_default(),
            model: model_name,
            usage,
        };
        // 8. 记录 output
        let output_str = serde_json::to_string(&result).unwrap_or_default();
        span.record("output", &output_str);
        Ok(result)
    }
}
