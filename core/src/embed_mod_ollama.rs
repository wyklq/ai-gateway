use crate::model::ollama::OllamaModel;
use crate::types::engine::OllamaModelParams;
use crate::model::CredentialsIdent;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::gateway::{EmbeddingUsage, Input, CreateEmbeddingResponse as GatewayEmbeddingResponse, EmbeddingData};
use crate::GatewayError;
use crate::GatewayResult;
use crate::model::types::{ModelEvent, ModelEventType, ModelFinishReason, LLMFinishEvent};
use futures::stream::TryReadyChunksError;
use futures::{Stream, TryStreamExt};
use serde_json::Value;
use tracing::{Span};
use async_trait::async_trait;
use crate::embed_mod::Embed;
use crate::model::ModelInstance;
use std::pin::Pin;

#[derive(Clone)]
pub struct OllamaEmbed {
    params: OllamaModelParams,
    model: OllamaModel,
    credentials_ident: CredentialsIdent,
}

impl OllamaEmbed {
    pub fn new(
        params: OllamaModelParams,
        credentials: Option<&ApiKeyCredentials>,
        endpoint: Option<&str>,
    ) -> Self {
        let model = OllamaModel::new(
            params.clone(),
            Default::default(),
            credentials.cloned(),
            endpoint.map(|s| s.to_string()),
        );
        let credentials_ident = credentials
            .map(|_c| CredentialsIdent::Own)
            .unwrap_or(CredentialsIdent::Langdb);
        Self {
            params,
            model,
            credentials_ident,
        }
    }

    async fn execute(
        &self,
        input: String,
        span: Span,
        tx: Option<&tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<Vec<f32>> {
        let embedding_input = async_openai::types::EmbeddingInput::String(input);
        let response = self.model.embed(embedding_input).await.map_err(GatewayError::from)?;
        // 这里只取第一个 embedding
        let embedding = response.data.get(0).map(|e| e.embedding.clone()).unwrap_or_default();
        let model_name = self.model.get_model_name();
        if let Some(tx) = tx {
            let _guard = span.enter();
            tx.send(Some(ModelEvent::new(
                &span,
                ModelEventType::LlmStop(LLMFinishEvent {
                    provider_name: "ollama".to_string(),
                    model_name,
                    output: None,
                    usage: None,
                    finish_reason: ModelFinishReason::Stop,
                    tool_calls: vec![],
                    credentials_ident: self.credentials_ident.clone(),
                }),
            )))
            .await
            .unwrap();
        }
        Ok(embedding)
    }
}

#[async_trait]
impl Embed for OllamaEmbed {
    async fn invoke(
        &self,
        input_text: Input,
        tx: Option<tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<GatewayEmbeddingResponse> {
        let input = match input_text {
            Input::String(s) => s,
            Input::Array(arr) => {
                // Ollama 暂不支持批量，取第一个
                arr.into_iter().next().ok_or_else(|| GatewayError::CustomError("Ollama embedding only supports String input".to_string()))?
            }
        };
        let call_span = tracing::info_span!("embedding_ollama", input = &input);
        let embedding = self.execute(input, call_span.clone(), tx.as_ref()).await?;
        let model_name = self.model.get_model_name();
        // Ollama 只返回一个 embedding
        let data = vec![EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: 0,
        }];
        Ok(GatewayEmbeddingResponse {
            object: "list".to_string(),
            data,
            model: model_name,
            usage: EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        })
    }

    async fn batched_invoke(
        &self,
        inputs: Box<dyn Stream<Item = GatewayResult<(String, Vec<Value>)>> + Send + Unpin>,
    ) -> Pin<Box<dyn Stream<Item = GatewayResult<Vec<(Vec<f32>, Vec<Value>)>>> + Send + '_>> {
        Box::pin(inputs
            .try_ready_chunks(2048)
            .map_err(|TryReadyChunksError(_, e)| e)
            .map_ok(|chunk| {
                let chunk_text: Vec<String> =
                    chunk.iter().map(|(text, _)| text.to_string()).collect();
                let values: Vec<Vec<Value>> =
                    chunk.iter().map(|(_, values)| values.clone()).collect();
                async {
                    let span = Span::current();
                    // Ollama 暂不支持批量，逐个处理
                    let mut result = Vec::new();
                    for text in chunk_text {
                        let embedding = self.execute(text, span.clone(), None).await?;
                        result.push(embedding);
                    }
                    Ok((result, values))
                }
            })
            .try_buffered(10)
            .map_ok(|(embeddings, values)| {
                embeddings.into_iter().zip(values.into_iter()).collect()
            }))
    }
}
