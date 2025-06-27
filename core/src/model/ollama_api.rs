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
use serde::{Deserialize, Serialize};
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
        "langdb::user_tracing::models::ollama_api"
    };
    ($subtgt:literal) => {
        concat!("langdb::user_tracing::models::ollama_api::", $subtgt)
    };
}

/// OllamaApiMessage represents the format required by Ollama's native /api/chat API
#[derive(Debug, Serialize, Deserialize)]
struct OllamaApiMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>, // Optional field for multimodal models
}

/// OllamaApiResponse represents the response format from Ollama's native /api/chat API
#[derive(Debug, Deserialize)]
struct OllamaApiResponse {
    model: String,
    created_at: String,
    message: OllamaApiMessage,
    done: bool,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
    done_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OllamaApiModel {
    pub client: Client,
    pub credentials: Option<ApiKeyCredentials>,
    pub execution_options: ExecutionOptions,
    pub params: OllamaModelParams,
    pub endpoint: Option<String>,
}

impl OllamaApiModel {
    pub fn new(
        params: OllamaModelParams,
        execution_options: ExecutionOptions,
        credentials: Option<ApiKeyCredentials>,
        endpoint: Option<String>,
    ) -> Self {
        let client = Client::new();

        tracing::debug!(target: "ollama_api_debug", "[OllamaApiModel::new] endpoint = {:?}", endpoint);

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
        // Parse the response in Ollama's native /api/chat format
        let api_response = serde_json::from_value::<OllamaApiResponse>(response.clone())
            .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama API response: {}", e)))?;

        // Extract the message from the response
        let role = api_response.message.role;
        let content = api_response.message.content;

        // Convert to ChatCompletionMessage format
        Ok(ChatCompletionMessage::new_text(role, content))
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
            ModelError::ConfigurationError(format!("Failed to parse Ollama API endpoint URL: {}", e))
        })
    }

    fn build_chat_request(&self, messages: &[ChatCompletionMessage], model_name: &str) -> serde_json::Value {
        // Convert ChatCompletionMessage format to Ollama API format
        let formatted_messages: Vec<serde_json::Value> = messages.iter().map(|msg| {
            match &msg.content {
                Some(ChatCompletionContent::Text(text)) => {
                    // 纯文本消息
                    json!({
                        "role": msg.role.to_string(),
                        "content": text.clone()
                    })
                },
                Some(ChatCompletionContent::Content(contents)) => {
                    // 可能包含图像的多模态内容
                    // 提取文本内容 - Ollama API 要求 content 字段只是一个纯文本字符串
                    let content = contents.iter().find(|c| c.r#type == crate::types::gateway::ContentType::Text)
                        .and_then(|c| c.text.clone())
                        .unwrap_or_default();
                                        
                    // 收集所有图像URL - 提取为纯 base64 数据（Ollama API 要求 images 数组中只包含 base64 数据，不需要任何前缀）
                    let image_urls: Vec<String> = contents.iter()
                        .filter(|c| c.r#type == crate::types::gateway::ContentType::ImageUrl)
                        .filter_map(|c| c.image_url.as_ref())
                        .filter_map(|img| {
                            // 检查URL是否已经是正确的data:格式并提取 base64 数据
                            let url = img.url.clone();                            
                            if url.starts_with("data:image/") && url.contains(";base64,") {
                                // URL是数据URL格式, 提取 base64 部分
                                match url.split(";base64,").nth(1) {
                                    Some(base64_data) => {
                                        if base64_data.is_empty() {
                                            None
                                        } else {
                                            Some(base64_data.to_string())
                                        }
                                    },
                                    None => {
                                        tracing::warn!(
                                            target: target!("image"),
                                            "build_chat_request: Failed to extract base64 data from data URL"
                                        );
                                        None
                                    },
                                }
                            } else if url.starts_with("http") {
                                // HTTP URL - Ollama 不支持，需要先下载并转换为base64
                                tracing::warn!(
                                    target: target!("image"),
                                    "build_chat_request: Found HTTP image URL but Ollama API requires base64 format: {}", url
                                );
                                None // 暂不支持 HTTP 网址
                            } else {
                                // 假设是纯base64数据
                                if url.trim().is_empty() {
                                    None
                                } else {
                                    Some(url)
                                }
                            }
                        })
                        .collect();
                    
                    // 根据 Ollama API 格式构建消息
                    // content 必须是纯文本，images 是可选的 base64 编码图像数组
                    if !image_urls.is_empty() {
                        json!({
                            "role": msg.role.to_string(),
                            "content": content,
                            "images": image_urls
                        })
                    } else {
                        // 没有图像，只发送文本
                        json!({
                            "role": msg.role.to_string(),
                            "content": content
                        })
                    }
                },
                None => {
                    // 没有内容的消息
                    json!({
                        "role": msg.role.to_string(),
                        "content": ""
                    })
                }
            }
        }).collect();
        
        // Build request payload for Ollama's native /api/chat endpoint
        let mut request = json!({
            "model": model_name,
            "messages": formatted_messages,
            "stream": false,
        });
        
        // Add optional parameters if specified
        if let Some(temp) = self.params.temperature {
            request["options"] = json!({
                "temperature": temp
            });
        }
        
        if let Some(tokens) = self.params.max_tokens {
            if request["options"].is_null() {
                request["options"] = json!({});
            }
            request["options"]["num_predict"] = json!(tokens);
        }

        if let Some(top_p) = self.params.top_p {
            if request["options"].is_null() {
                request["options"] = json!({});
            }
            request["options"]["top_p"] = json!(top_p);
        }
        
        if let Some(stop) = &self.params.stop {
            if request["options"].is_null() {
                request["options"] = json!({});
            }
            request["options"]["stop"] = json!(stop);
        }
        
        if let Some(format) = &self.params.response_format {
            match format {
                OllamaResponseFormat::Json => {
                    if request["options"].is_null() {
                        request["options"] = json!({});
                    }
                    request["format"] = json!("json");
                }
            }
        }
        
        request
    }
}

#[async_trait]
impl ModelInstance for OllamaApiModel {
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

        // 转换 Message 为 ChatCompletionMessage，支持文本和图像内容
        let messages: Vec<ChatCompletionMessage> = previous_messages.iter().map(|m| {
            // 检查是否有内容数组（可能包含图像）
            if !m.content_array.is_empty() {
                let mut contents = Vec::new();
                if let Some(text_content) = &m.content {
                    if !text_content.trim().is_empty() {
                        contents.push(crate::types::gateway::Content {
                            r#type: crate::types::gateway::ContentType::Text,
                            text: Some(text_content.clone()),
                            image_url: None,
                            audio: None,
                        });
                    }
                }
                for part in &m.content_array {
                    let type_str = part.r#type.to_string();
                    if type_str == "text" || type_str == "Text" {
                        let text = part.value.clone();
                        if !text.trim().is_empty() {
                            contents.push(crate::types::gateway::Content {
                                r#type: crate::types::gateway::ContentType::Text,
                                text: Some(text),
                                image_url: None,
                                audio: None,
                            });
                        }
                    } else if type_str == "image_url" || type_str == "ImageUrl" || type_str == "image" || type_str == "url" {
                        if part.value.trim().is_empty() {
                            continue;
                        }
                        let raw_value = part.value.clone();
                        let url: String;
                        if raw_value.starts_with("{") && raw_value.ends_with("}") {
                            match serde_json::from_str::<serde_json::Value>(&raw_value) {
                                Ok(json_value) => {
                                    let url_value = json_value.get("url")
                                        .or_else(|| json_value.get("image_url"))
                                        .and_then(|v| v.as_str());
                                    if let Some(image_url) = url_value {
                                        url = image_url.to_string();
                                    } else {
                                        continue;
                                    }
                                },
                                Err(_) => {
                                    url = raw_value;
                                }
                            }
                        } else {
                            url = raw_value;
                        }
                        if !url.trim().is_empty() {
                            let image_url = if !url.starts_with("data:") && !url.starts_with("http") {
                                format!("data:image/jpeg;base64,{}", url)
                            } else {
                                url
                            };
                            contents.push(crate::types::gateway::Content {
                                r#type: crate::types::gateway::ContentType::ImageUrl,
                                text: None,
                                image_url: Some(crate::types::gateway::ImageUrl {
                                    url: image_url,
                                }),
                                audio: None,
                            });
                        }
                    }
                }
                ChatCompletionMessage {
                    role: m.r#type.to_string(),
                    content: Some(ChatCompletionContent::Content(contents)),
                    tool_call_id: m.tool_call_id.clone(),
                    tool_calls: m.tool_calls.clone(),
                    refusal: None,
                }
            } else {
                let text = m.content.clone().unwrap_or_default();
                ChatCompletionMessage::new_text(
                    m.r#type.to_string(),
                    text
                )
            }
        }).collect();
        
        // Create a span specifically for this request
        let input = serde_json::to_string(&messages).unwrap_or_default();
        let span = tracing::info_span!(
            target: target!("chat"),
            "model_call",
            provider = "ollama_api",
            model = model_name,
            input = input,
            output = field::Empty,
            error = field::Empty,
            usage = field::Empty,
            tags = crate::events::JsonValue(&serde_json::to_value(tags.clone()).unwrap_or_default()).as_value(),
        );
        
        // Send LlmStart event to properly initialize trace context
        tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                provider_name: "ollama_api".to_string(),
                model_name: model_name.clone(),
                input,
            })
        ))).await
            .map_err(|e| crate::error::GatewayError::CustomError(e.to_string()))?;

        // 打印 endpoint 相关 debug 信息
        let base_url = self.get_base_url()?;
        // Use the native /api/chat endpoint instead of OpenAI compatible endpoint
        let url = base_url.join("/api/chat").map_err(|e| {
            let err_msg = format!("Failed to construct Ollama API URL: {}", e);
            span.record("error", &err_msg);
            ModelError::ConfigurationError(err_msg)
        })?;
        
        // Use the build_chat_request method to create the proper request for the /api/chat endpoint
        let request_body = self.build_chat_request(&messages, &model_name);
        
        // Send request and record the span event with proper trace context
        let response = async {
            self.send_request(url, request_body, &tx).await
        }
        .instrument(span.clone())
        .await?;
        
        // Parse the response using the native format parse method
        let message = self.parse_chat_completion_response(response.clone()).await?;
        
        // Record the response in the span
        let output_str = serde_json::to_string(&message).unwrap_or_default();
        span.record("output", &output_str);
        
        // Extract token counts from the response if available
        let api_response = serde_json::from_value::<OllamaApiResponse>(response.clone())
            .map_err(|e| ModelError::ParsingResponseFailed(format!("Failed to parse Ollama API response: {}", e)))?;
            
        let prompt_tokens = api_response.prompt_eval_count;
        let completion_tokens = api_response.eval_count;
        let usage = self.calculate_usage(prompt_tokens, completion_tokens);
        
        // Update usage in span with proper format
        if let Usage::CompletionModelUsage(ref u) = usage {
            span.record("usage", &format!("{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}", 
                u.input_tokens, u.output_tokens, u.total_tokens));
        }
        
        let credentials_ident = if self.credentials.is_none() {
            crate::model::CredentialsIdent::Langdb
        } else {
            crate::model::CredentialsIdent::Own
        };
        
        // 发送 LlmStop 事件
        tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                provider_name: "ollama_api".to_string(),
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
        // 验证模型名称
        let model_name = self.validate_model()?;
        
        // Create a span specifically for this stream request
        let input = serde_json::to_string(&previous_messages).unwrap_or_default();
        let span = tracing::info_span!(
            target: target!("chat_stream"),
            "model_call_stream",
            provider = "ollama_api",
            model = model_name,
            input = input,
            error = field::Empty,
            tags = crate::events::JsonValue(&serde_json::to_value(tags.clone()).unwrap_or_default()).as_value()
        );
        
        // Send LlmStart event to properly initialize trace context
        tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                provider_name: "ollama_api".to_string(),
                model_name: model_name.clone(),
                input,
            })
        ))).await
            .map_err(|e| crate::error::GatewayError::CustomError(e.to_string()))?;
            
        // For now, streaming is not implemented
        let error_msg = "Ollama API streaming not implemented yet";
        span.record("error", &error_msg);
        
        // Send LlmStop event with error
        tx.send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                provider_name: "ollama_api".to_string(),
                model_name: model_name,
                output: Some(error_msg.to_string()),
                usage: None,
                finish_reason: crate::model::types::ModelFinishReason::ContentFilter,
                tool_calls: vec![],
                credentials_ident: if self.credentials.is_none() {
                    crate::model::CredentialsIdent::Langdb
                } else {
                    crate::model::CredentialsIdent::Own
                },
            })
        ))).await
            .map_err(|e| crate::error::GatewayError::CustomError(e.to_string()))?;
            
        Err(ModelError::RequestFailed(error_msg.to_string()).into())
    }

    async fn embed(
        &self,
        _input: EmbeddingInput,
    ) -> Result<CreateEmbeddingResponse, ModelError> {
        // Embedding API is the same in both versions, so we'll just return an error
        Err(ModelError::ParsingResponseFailed(
            "Ollama API embedding feature not implemented. Use the regular Ollama provider for embeddings.".to_string()
        ))
    }
}
