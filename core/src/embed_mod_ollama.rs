use crate::model::ollama::OllamaModel;
use crate::types::engine::OllamaModelParams;
use crate::model::CredentialsIdent;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::gateway::CompletionModelUsage;
use crate::GatewayError;
use crate::GatewayResult;
use crate::model::types::{ModelEvent, ModelEventType, ModelFinishReason, LLMFinishEvent};
use futures::stream::TryReadyChunksError;
use futures::{Stream, TryStreamExt};
use serde_json::Value;
use tracing::Instrument;
use tracing::{field, Span};
use async_trait::async_trait;
use crate::embed_mod::Embed;
use async_openai::types::{EmbeddingInput, CreateEmbeddingResponse};

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
        let (embedding, usage) = self.model.embed(&input, &tokio::sync::mpsc::channel(1).0).await.map_err(GatewayError::from)?;
        if let Some(tx) = tx {
            tx.send(Some(ModelEvent::new(
                &span,
                ModelEventType::LlmStop(LLMFinishEvent {
                    provider_name: "ollama".to_string(),
                    model_name: self.params.model.clone().unwrap_or_default(),
                    output: None,
                    usage: Some(CompletionModelUsage {
                        input_tokens: usage.prompt_tokens,
                        output_tokens: 0,
                        total_tokens: usage.total_tokens,
                        ..Default::default()
                    }),
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
        input_text: EmbeddingInput,
        tx: Option<tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<CreateEmbeddingResponse> {
        let mut data = Vec::new();
        let mut usage = CompletionModelUsage::default();
        let texts: Vec<String> = match input_text {
            EmbeddingInput::String(s) => vec![s],
            EmbeddingInput::Array(arr) => arr,
        };
        for (i, input_str) in texts.into_iter().enumerate() {
            let call_span = tracing::info_span!(target: "embedding", "ollama", input = input_str, output = field::Empty, ttft = field::Empty, error = field::Empty, usage = field::Empty);
            let embedding = self.execute(input_str.clone(), call_span.clone(), tx.as_ref()).instrument(call_span.clone()).await?;
            data.push(async_openai::types::Embedding {
                index: i as i32,
                embedding,
                object: "embedding".to_string(),
            });
            // usage 统计可累加
        }
        Ok(CreateEmbeddingResponse {
            object: "list".to_string(),
            data,
            model: self.params.model.clone().unwrap_or_default(),
            usage,
        })
    }

    async fn batched_invoke(
        &self,
        inputs: Box<dyn Stream<Item = GatewayResult<(String, Vec<Value>)>> + Send + Unpin>,
    ) -> Box<dyn Stream<Item = GatewayResult<Vec<(Vec<f32>, Vec<Value>)>>> + Send + Unpin> {
        Box::new(inputs
            .try_ready_chunks(2048)
            .map_err(|TryReadyChunksError(_, e)| e)
            .map_ok(|chunk| {
                let chunk_text: Vec<String> = chunk.iter().map(|(text, _)| text.to_string()).collect();
                let values: Vec<Vec<Value>> = chunk.iter().map(|(_, values)| values.clone()).collect();
                async {
                    let span = Span::current();
                    let mut results = Vec::new();
                    for text in chunk_text {
                        let embedding = self.execute(text, span.clone(), None).await?;
                        results.push(embedding);
                    }
                    Ok((results, values))
                }
            })
            .try_buffered(10)
            .map_ok(|(embeddings, values)| {
                embeddings.into_iter().zip(values.into_iter()).collect()
            }))
    }
}
