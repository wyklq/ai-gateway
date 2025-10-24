use crate::model::error::ModelError;
use crate::model::types::{LLMFirstToken, ModelEvent, ModelEventType};
use crate::model::ModelInstance;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::engine::ExecutionOptions;
use crate::types::engine::{OllamaModelParams, OllamaResponseFormat};
use crate::types::gateway::{
    ChatCompletionContent, ChatCompletionMessage, Usage, CompletionModelUsage, ToolCall, FunctionCall,
};
use async_openai::types::{EmbeddingInput, CreateEmbeddingResponse};
use async_trait::async_trait;
use futures_util::stream::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, Url};
use serde_json::json;
use serde::{Serialize, Deserialize};
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
    pub max_retries: Option<u32>,
}

impl OllamaModel {
    // 添加 map_tool_call 方法，将 ToolCall 转换为 ModelToolCall
    pub fn map_tool_call(tool_call: &ToolCall) -> crate::model::types::ModelToolCall {
        crate::model::types::ModelToolCall {
            tool_id: tool_call.id.clone(),
            tool_name: tool_call.function.name.clone(),
            input: tool_call.function.arguments.clone(),
        }
    }

    pub fn new(
        params: OllamaModelParams,
        execution_options: ExecutionOptions,
        credentials: Option<ApiKeyCredentials>,
        endpoint: Option<String>,
        max_retries: Option<u32>,
    ) -> Self {
        let client = Client::new();
        tracing::debug!(target: "ollama_debug", "[OllamaModel::new] endpoint = {:?}", endpoint);
        Self {
            client,
            credentials,
            execution_options,
            params,
            endpoint,
            max_retries,
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

            let _ = tx.try_send(Some(ModelEvent::new(
                &span,
                ModelEventType::LlmContent(crate::model::types::LLMContentEvent {
                    content: error_msg.clone(),
                })
            ))); 

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
    ) -> Result<(ChatCompletionMessage, Option<CompletionModelUsage>), ModelError> {
        // Compatible with OpenAI style; support string or array content
        let message = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .ok_or_else(|| ModelError::ParsingResponseFailed("Missing choices[0].message".to_string()))?;

        let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("assistant").to_string();
        let raw_content = message.get("content").ok_or_else(|| ModelError::ParsingResponseFailed("Missing message.content".to_string()))?;

        // Parse tool calls if present
        let tool_calls = message.get("tool_calls").and_then(|tool_calls_value| {
            tool_calls_value.as_array().map(|tool_calls_array| {
                tool_calls_array
                    .iter()
                    .enumerate()
                    .filter_map(|(index, tool_call)| {
                        let id = tool_call.get("id")?.as_str()?.to_string();
                        let tool_type = tool_call.get("type")?.as_str()?.to_string();
                        let function = tool_call.get("function")?;
                        let name = function.get("name")?.as_str()?.to_string();
                        let arguments = function.get("arguments")?.as_str()?.to_string();
                        Some(ToolCall {
                            index: Some(index),
                            id,
                            r#type: tool_type,
                            function: FunctionCall { name, arguments },
                        })
                    })
                    .collect::<Vec<ToolCall>>()
            })
        });

        // usage field parsing
        let usage = response.get("usage").and_then(|usage_val| {
            let prompt_tokens = usage_val.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let completion_tokens = usage_val.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let total_tokens = usage_val.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            Some(CompletionModelUsage {
                input_tokens: prompt_tokens,
                output_tokens: completion_tokens,
                total_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: None,
                is_cache_used: false,
            })
        });

        // Content may be string or array of objects
        let mut chat_msg = if let Some(s) = raw_content.as_str() {
            ChatCompletionMessage::new_text(role, s.to_string())
        } else if let Some(arr) = raw_content.as_array() {
            let mut parts: Vec<crate::types::gateway::Content> = Vec::new();
            for item in arr {
                if let Some(t) = item.get("type").and_then(|v| v.as_str()) {
                    match t {
                        "text" => {
                            let txt = item.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            parts.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::Text, text: Some(txt), image_url: None, audio: None });
                        }
                        "image_url" => {
                            let url = item.get("image_url")
                                .and_then(|iu| iu.get("url"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            parts.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::ImageUrl, text: None, image_url: Some(crate::types::gateway::ImageUrl { url, detail: None }), audio: None });
                        }
                        "input_audio" => {
                                if let Some(audio_obj) = item.get("audio") {
                                let data = audio_obj.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let format = audio_obj.get("format").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                parts.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::InputAudio, text: None, image_url: None, audio: Some(crate::types::gateway::InputAudio { data, format }) });
                            } else {
                                parts.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::InputAudio, text: None, image_url: None, audio: None });
                            }
                        }
                        other => {
                            tracing::debug!(target: "ollama_debug", "Unknown content part type in response: {}", other);
                        }
                    }
                }
            }
            ChatCompletionMessage { role, content: Some(ChatCompletionContent::Content(parts)), tool_call_id: None, tool_calls: None, refusal: None }
        } else {
            ChatCompletionMessage::new_text(role, "".to_string())
        };

        if let Some(tools) = tool_calls { chat_msg.tool_calls = Some(tools); }
        Ok((chat_msg, usage))
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
        #[derive(Debug, Serialize, Deserialize)]
        struct OllamaImageUrl { url: String }
        #[derive(Debug, Serialize, Deserialize)]
        struct OllamaAudio { data: String, format: String }
        #[derive(Debug, Serialize, Deserialize)]
        struct OllamaContentPart {
            #[serde(rename = "type")]
            part_type: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            text: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            image_url: Option<OllamaImageUrl>,
            #[serde(skip_serializing_if = "Option::is_none")]
            audio: Option<OllamaAudio>,
        }
        #[derive(Debug, Serialize, Deserialize)]
        struct OllamaRequestMessage {
            role: String,
            content: serde_json::Value,
        }
        #[derive(Debug, Serialize, Deserialize)]
        struct OllamaChatRequest {
            model: String,
            messages: Vec<OllamaRequestMessage>,
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            max_tokens: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_p: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop: Option<Vec<String>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            format: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            frequency_penalty: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            presence_penalty: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            seed: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tools: Option<serde_json::Value>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_choice: Option<serde_json::Value>,
            #[serde(skip_serializing_if = "Option::is_none")]
            logit_bias: Option<serde_json::Value>,
            #[serde(skip_serializing_if = "Option::is_none")]
            user: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            n: Option<u32>,
        }

        let mut req_messages: Vec<OllamaRequestMessage> = Vec::with_capacity(messages.len());
        for msg in messages {
            let role = msg.role.to_string();
            let value = match &msg.content {
                Some(ChatCompletionContent::Text(t)) => serde_json::Value::String(t.clone()),
                Some(ChatCompletionContent::Content(contents)) => {
                    let mut parts: Vec<OllamaContentPart> = Vec::with_capacity(contents.len());
                    for c in contents {
                        match c.r#type {
                            crate::types::gateway::ContentType::Text => {
                                parts.push(OllamaContentPart { part_type: "text".to_string(), text: c.text.clone(), image_url: None, audio: None });
                            }
                            crate::types::gateway::ContentType::ImageUrl => {
                                let url = c.image_url.as_ref().map(|i| i.url.clone()).unwrap_or_default();
                                parts.push(OllamaContentPart { part_type: "image_url".to_string(), text: None, image_url: Some(OllamaImageUrl { url }), audio: None });
                            }
                            crate::types::gateway::ContentType::InputAudio => {
                                if let Some(audio) = &c.audio {
                                    parts.push(OllamaContentPart { part_type: "input_audio".to_string(), text: None, image_url: None, audio: Some(OllamaAudio { data: audio.data.clone(), format: audio.format.clone() }) });
                                } else {
                                    parts.push(OllamaContentPart { part_type: "input_audio".to_string(), text: None, image_url: None, audio: None });
                                }
                            }
                        }
                    }
                    serde_json::to_value(parts).unwrap_or_else(|_| serde_json::Value::Array(vec![]))
                }
                None => serde_json::Value::String(String::new()),
            };
            let rm = OllamaRequestMessage { role, content: value };
            tracing::debug!(target: "ollama_debug", formatted_message = %serde_json::to_string(&rm).unwrap_or_default());
            req_messages.push(rm);
        }

        let request = OllamaChatRequest {
            model: model_name.to_string(),
            messages: req_messages,
            stream,
            temperature: self.params.temperature,
            max_tokens: self.params.max_tokens.map(|v| v as u32),
            top_p: self.params.top_p,
            stop: self.params.stop.clone(),
            format: self.params.response_format.as_ref().map(|f| match f { OllamaResponseFormat::Json => "json".to_string() }),
            frequency_penalty: self.params.frequency_penalty,
            presence_penalty: self.params.presence_penalty,
            seed: self.params.seed.map(|v| v as u32),
            tools: self.params.tools.clone().map(|t| json!(t)),
            tool_choice: self.params.tool_choice.clone(),
            logit_bias: self.params.logit_bias.clone().map(|lb| json!(lb)),
            user: self.params.user.clone(),
            n: self.params.n,
        };

        let value = serde_json::to_value(&request).unwrap_or_else(|_| json!({"model": model_name, "messages": [], "stream": stream}));
        tracing::debug!(target: "ollama_debug", final_request_body = %value);
        value
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

        // 支持多模态：如果存在 content_array 则重建为 ChatCompletionContent::Content
        let messages: Vec<ChatCompletionMessage> = previous_messages.iter().map(|m| {
            if !m.content_array.is_empty() {
                let mut contents_vec: Vec<crate::types::gateway::Content> = Vec::new();
                // 若原始有单独的纯文本 content 字段，保留为一个 text part
                if let Some(txt) = &m.content {
                    if !txt.trim().is_empty() {
                        contents_vec.push(crate::types::gateway::Content {
                            r#type: crate::types::gateway::ContentType::Text,
                            text: Some(txt.clone()),
                            image_url: None,
                            audio: None,
                        });
                    }
                }
                for part in &m.content_array {
                    let kind = part.r#type.to_string();
                    if kind.eq_ignore_ascii_case("text") {
                        if !part.value.trim().is_empty() {
                            contents_vec.push(crate::types::gateway::Content {
                                r#type: crate::types::gateway::ContentType::Text,
                                text: Some(part.value.clone()),
                                image_url: None,
                                audio: None,
                            });
                        }
                    } else if kind.eq_ignore_ascii_case("image_url") || kind.eq_ignore_ascii_case("image") || kind.eq_ignore_ascii_case("url") {
                        if part.value.trim().is_empty() { continue; }
                        let raw = part.value.clone();
                        // 如果是 JSON 包住的 {"url":"..."} 形式，解析取 url
                        let mut final_url = raw.clone();
                        if raw.starts_with('{') && raw.ends_with('}') {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
                                if let Some(u) = v.get("url").and_then(|x| x.as_str())
                                    .or_else(|| v.get("image_url").and_then(|x| x.as_str()) ) {
                                    final_url = u.to_string();
                                }
                            }
                        }
                        contents_vec.push(crate::types::gateway::Content {
                            r#type: crate::types::gateway::ContentType::ImageUrl,
                            text: None,
                            image_url: Some(crate::types::gateway::ImageUrl { url: final_url, detail: None }),
                            audio: None,
                        });
                    } else if kind.eq_ignore_ascii_case("input_audio") {
                        // 保留音频（如果需要完整支持，可在 part.additional_options 中扩展）暂时仅当作占位
                        // 由于 threads::MessageContentPart 里音频数据结构可能不同，这里简单跳过，避免破坏现有逻辑
                        continue;
                    }
                }
                ChatCompletionMessage {
                    role: m.r#type.to_string(),
                    content: Some(ChatCompletionContent::Content(contents_vec)),
                    tool_call_id: m.tool_call_id.clone(),
                    tool_calls: m.tool_calls.clone(),
                    refusal: None,
                }
            } else {
                ChatCompletionMessage::new_text(
                    m.r#type.to_string(),
                    m.content.clone().unwrap_or_default(),
                )
            }
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
        let _ = tx.try_send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                provider_name: "ollama".to_string(),
                model_name: model_name.clone(),
                input,
            }))));
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
        let (message, usage_from_response) = self.parse_chat_completion_response(response.clone()).await?;
        
        // Record the response in the span
        let output_str = serde_json::to_string(&message).unwrap_or_default();
        span.record("output", &output_str);
        
        // 优先用 response usage 字段，没有则 fallback 到估算
        let usage = if let Some(u) = usage_from_response {
            Usage::CompletionModelUsage(u)
        } else {
            let prompt_length: u32 = messages.iter().map(|m| {
                m.content.as_ref().and_then(|c| c.as_string()).map_or(0, |t| t.len() as u32)
            }).sum();
            let completion_length = message.content.as_ref().and_then(|c| c.as_string()).map_or(0, |t| t.len() as u32);
            let prompt_tokens = Some(prompt_length / 4);
            let completion_tokens = Some(completion_length / 4);
            self.calculate_usage(prompt_tokens, completion_tokens)
        };
        
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
        let tool_calls_for_event = message.tool_calls.clone().unwrap_or_default();
        // 类型转换，将 ToolCall 转换为 ModelToolCall
        let tool_calls_for_event: Vec<crate::model::types::ModelToolCall> = tool_calls_for_event.iter().map(Self::map_tool_call).collect();
        
        let _ = tx.try_send(Some(ModelEvent::new(
            &span,
            ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                provider_name: "ollama".to_string(),
                model_name: model_name.clone(),
                output: message.content.as_ref().and_then(|c| c.as_string()),
                usage: Some(match usage {
                    Usage::CompletionModelUsage(u) => u,
                    _ => Default::default(),
                }),
                finish_reason: if message.tool_calls.is_some() {
                    crate::model::types::ModelFinishReason::ToolCalls
                } else {
                    crate::model::types::ModelFinishReason::Stop
                },
                tool_calls: tool_calls_for_event,
                credentials_ident,
            })
        )));

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
            let _ = tx.try_send(Some(ModelEvent::new(
                &span_clone,
                ModelEventType::LlmStart(crate::model::types::LLMStartEvent {
                    provider_name: "ollama".to_string(),
                    model_name: model_name.clone(),
                    input: serde_json::to_string(&previous_messages).unwrap_or_default(),
                }),
            )));

            let base_url = self.get_base_url()?;
            let url = base_url.join("/v1/chat/completions").map_err(|e| {
                let err_msg = format!("Failed to construct Ollama API URL: {}", e);
                span_clone.record("error", &err_msg);
                ModelError::ConfigurationError(err_msg)
            })?;

            let messages: Vec<ChatCompletionMessage> = previous_messages.iter().map(|m| {
                if !m.content_array.is_empty() {
                    let mut contents_vec: Vec<crate::types::gateway::Content> = Vec::new();
                    if let Some(txt) = &m.content { if !txt.trim().is_empty() { contents_vec.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::Text, text: Some(txt.clone()), image_url: None, audio: None }); } }
                    for part in &m.content_array {
                        let kind = part.r#type.to_string();
                        if kind.eq_ignore_ascii_case("text") {
                            if !part.value.trim().is_empty() {
                                contents_vec.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::Text, text: Some(part.value.clone()), image_url: None, audio: None });
                            }
                        } else if kind.eq_ignore_ascii_case("image_url") || kind.eq_ignore_ascii_case("image") || kind.eq_ignore_ascii_case("url") {
                            if part.value.trim().is_empty() { continue; }
                            let raw = part.value.clone();
                            let mut final_url = raw.clone();
                            if raw.starts_with('{') && raw.ends_with('}') { if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) { if let Some(u) = v.get("url").and_then(|x| x.as_str()).or_else(|| v.get("image_url").and_then(|x| x.as_str()) ) { final_url = u.to_string(); } } }
                            contents_vec.push(crate::types::gateway::Content { r#type: crate::types::gateway::ContentType::ImageUrl, text: None, image_url: Some(crate::types::gateway::ImageUrl { url: final_url, detail: None }), audio: None });
                        } else if kind.eq_ignore_ascii_case("input_audio") {
                            continue; // 暂不处理音频
                        }
                    }
                    ChatCompletionMessage { role: m.r#type.to_string(), content: Some(ChatCompletionContent::Content(contents_vec)), tool_call_id: m.tool_call_id.clone(), tool_calls: m.tool_calls.clone(), refusal: None }
                } else {
                    ChatCompletionMessage::new_text(m.r#type.to_string(), m.content.clone().unwrap_or_default())
                }
            }).collect();

            let request_body = self.build_chat_request(&messages, &model_name, true);
            let headers = self.build_headers();

            let mut retries_left = self.max_retries.unwrap_or(crate::model::DEFAULT_MAX_RETRIES);
            let response = loop {
                let resp_result = self
                    .client
                    .post(url.clone())
                    .headers(headers.clone())
                    .json(&request_body)
                    .send()
                    .await;
                match resp_result {
                    Ok(response) => {
                        if response.status().is_success() {
                            break response;
                        } else {
                            let status = response.status();
                            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                            let error_msg = format!("Request failed with status {}: {}", status, error_text);
                            span_clone.record("error", &error_msg);
                            let should_retry = status.is_server_error() || status == 429;
                            if !should_retry || retries_left == 0 {
                                let credentials_ident = if self.credentials.is_none() {
                                    crate::model::CredentialsIdent::Langdb
                                } else {
                                    crate::model::CredentialsIdent::Own
                                };
                                let _ = tx.try_send(Some(ModelEvent::new(
                                    &span_clone,
                                    ModelEventType::LlmStop(crate::model::types::LLMFinishEvent {
                                        provider_name: "ollama".to_string(),
                                        model_name: model_name.clone(),
                                        output: Some(error_msg.clone()),
                                        usage: None,
                                        finish_reason: crate::model::types::ModelFinishReason::ContentFilter,
                                        tool_calls: vec![], // 空的 ModelToolCall 向量
                                        credentials_ident,
                                    }),
                                )));
                                return Err(ModelError::RequestFailed(error_msg).into());
                            }
                            retries_left -= 1;
                            tokio::time::sleep(tokio::time::Duration::from_millis(
                                1000 * (self.max_retries.unwrap_or(crate::model::DEFAULT_MAX_RETRIES) - retries_left) as u64
                            )).await;
                            continue;
                        }
                    },
                    Err(e) => {
                        let error_msg = format!("Failed to send streaming request: {}", e);
                        span_clone.record("error", &error_msg);
                        if retries_left == 0 {
                            return Err(ModelError::RequestFailed(error_msg).into());
                        }
                        retries_left -= 1;
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            1000 * (self.max_retries.unwrap_or(crate::model::DEFAULT_MAX_RETRIES) - retries_left) as u64
                        )).await;
                    }
                }
            };

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
                            let _ = tx.try_send(Some(ModelEvent::new(
                                &span_clone,
                                ModelEventType::LlmFirstToken(
                                                LLMFirstToken {}
                                            ),
                                        )));
                        }

                        if let Some(choices) = value.get("choices").and_then(|c| c.as_array()) {
                            if let Some(choice) = choices.get(0) {
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                        if !content.is_empty() {
                                            full_content.push_str(content);
                                            let _ = tx.try_send(Some(ModelEvent::new(
                                                &span_clone,
                                                ModelEventType::LlmContent(
                                                    crate::model::types::LLMContentEvent {
                                                        content: content.to_string(),
                                                    },
                                                ),
                                            )));
                                        }
                                    }
                                    // Extract and process tool calls
                                    if let Some(delta_tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                                        for tool_call in delta_tool_calls {
                                            // 只要存在 tool_calls 就设置完成原因为 ToolCalls
                                            if tool_call.get("id").is_some() {
                                                finish_reason = crate::model::types::ModelFinishReason::ToolCalls;
                                                // 直接忽略 LlmToolCall 相关事件
                                            }
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

            // We need to collect any tool calls from the final response
            let collected_tool_calls: Vec<crate::model::types::ModelToolCall> = if finish_reason == crate::model::types::ModelFinishReason::ToolCalls {
                // In a real implementation, you would collect tool calls from stream chunks
                // For simplicity, we're creating a placeholder implementation that relies on the finish reason
                // In production, you should maintain a list of tool calls seen in the stream
                let tool_calls = Vec::new();
                
                // If we know we had tool calls (from finish_reason), but don't have the details,
                // we can add a debug note about it
                tracing::debug!(target: "ollama_debug", "Stream finished with tool calls but details weren't collected");
                
                tool_calls
            } else {
                vec![]
            };
            
            let _ = tx.try_send(Some(ModelEvent::new(
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
                    tool_calls: collected_tool_calls,
                    credentials_ident,
                }),
            )));

            Ok(())
        }
        .instrument(span.clone().or_current())
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
