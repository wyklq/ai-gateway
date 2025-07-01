use crate::events::{JsonValue, RecordResult, SPAN_OPENAI};
use crate::model::error::ModelError;
use crate::model::openai::openai_client;
use crate::model::types::LLMFinishEvent;
use crate::model::types::ModelEvent;
use crate::model::types::ModelEventType;
use crate::model::types::ModelFinishReason;
use crate::model::CredentialsIdent;
use crate::types::credentials::ApiKeyCredentials;
use crate::types::embed::OpenAiEmbeddingParams;
use crate::types::gateway::CompletionModelUsage;
use crate::GatewayError;
use crate::GatewayResult;
use async_openai::config::OpenAIConfig;
use async_openai::types::{CreateEmbeddingRequestArgs, CreateEmbeddingResponse, EmbeddingInput};
use async_openai::Client;
use futures::stream::TryReadyChunksError;
use futures::{Stream, TryStreamExt};
use serde_json::Value;
use tracing::Instrument;
use tracing::{field, Span};
use valuable::Valuable;

pub mod ollama;

macro_rules! target {
    () => {
        "langdb::user_tracing::models::openai"
    };
    ($subtgt:literal) => {
        concat!("langdb::user_tracing::models::openai::", $subtgt)
    };
}

#[allow(async_fn_in_trait)]
pub trait Embed: Sync + Send {
    async fn invoke(
        &self,
        input_text: EmbeddingInput,
        tx: Option<tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<CreateEmbeddingResponse>;
    async fn batched_invoke(
        &self,
        inputs: impl Stream<Item = GatewayResult<(String, Vec<Value>)>>,
    ) -> impl Stream<Item = GatewayResult<Vec<(Vec<f32>, Vec<Value>)>>>;
}

#[derive(Clone)]
pub struct OpenAIEmbed {
    params: OpenAiEmbeddingParams,
    client: Client<OpenAIConfig>,
    credentials_ident: CredentialsIdent,
}

impl OpenAIEmbed {
    pub fn new(
        params: OpenAiEmbeddingParams,
        credentials: Option<&ApiKeyCredentials>,
        endpoint: Option<&str>,
    ) -> Result<Self, ModelError> {
        let client = openai_client(credentials, endpoint)?;

        let credentials_ident = credentials
            .map(|_c| CredentialsIdent::Own)
            .unwrap_or(CredentialsIdent::Langdb);

        Ok(Self {
            params,
            client,
            credentials_ident,
        })
    }

    async fn execute(
        &self,
        input: EmbeddingInput,
        span: Span,
        tx: Option<&tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<CreateEmbeddingResponse> {
        let embedding_model = self.params.model.as_ref().unwrap();

        // Start building the request
        let mut request_builder = CreateEmbeddingRequestArgs::default();

        let request_builder = request_builder.model(embedding_model.clone()).input(input);

        if let Some(dimensions) = self.params.dimensions {
            request_builder.dimensions(dimensions);
        }
        // Finalize the request
        let request = request_builder.build().map_err(ModelError::OpenAIApi)?;

        // Send the request and handle the response
        let mut response = async move {
            let result = self.client.embeddings().create(request).await;

            let _ = result
                .as_ref()
                .map(|response| serde_json::to_value(response).unwrap())
                .as_ref()
                .map(JsonValue)
                .record();

            let response = result.map_err(|e| ModelError::CustomError(e.to_string()))?;

            let span = Span::current();
            let usage = response.usage.clone();
            span.record(
                "usage",
                JsonValue(&serde_json::to_value(usage).unwrap()).as_value(),
            );
            Ok::<_, GatewayError>(response)
        }
        .instrument(span.clone().or_current())
        .await?;

        if let Some(tx) = tx {
            tx.send(Some(ModelEvent::new(
                &span,
                ModelEventType::LlmStop(LLMFinishEvent {
                    provider_name: SPAN_OPENAI.to_string(),
                    model_name: self.params.model.clone().unwrap_or_default(),
                    output: None,
                    usage: Some(CompletionModelUsage {
                        input_tokens: response.usage.prompt_tokens,
                        output_tokens: 0,
                        total_tokens: response.usage.total_tokens,
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

        let mut embeddings = response.data;
        embeddings.sort_by_key(|e| e.index);

        response.data = embeddings;

        Ok(response)
    }
}

impl Embed for OpenAIEmbed {
    async fn invoke(
        &self,
        input_text: EmbeddingInput,
        tx: Option<tokio::sync::mpsc::Sender<Option<ModelEvent>>>,
    ) -> GatewayResult<CreateEmbeddingResponse> {
        let input = serde_json::to_string(&input_text)?;
        let call_span = tracing::info_span!(target: target!("embedding"), SPAN_OPENAI, input = input, output = field::Empty, ttft = field::Empty, error = field::Empty, usage = field::Empty);

        self.execute(input_text, call_span.clone(), tx.as_ref())
            .instrument(call_span.clone())
            .await
    }

    async fn batched_invoke(
        &self,
        inputs: impl Stream<Item = GatewayResult<(String, Vec<Value>)>>,
    ) -> impl Stream<Item = GatewayResult<Vec<(Vec<f32>, Vec<Value>)>>> {
        inputs
            .try_ready_chunks(2048)
            .map_err(|TryReadyChunksError(_, e)| e)
            .map_ok(|chunk| {
                let chunk_text: Vec<String> =
                    chunk.iter().map(|(text, _)| text.to_string()).collect();
                let values: Vec<Vec<Value>> =
                    chunk.iter().map(|(_, values)| values.clone()).collect();
                async {
                    let span = Span::current();
                    let embeddings = self.execute(chunk_text.into(), span, None).await?;

                    Ok((embeddings, values))
                }
            })
            .try_buffered(10)
            .map_ok(|(embeddings, values)| {
                let x: Vec<Vec<f32>> = embeddings.data.into_iter().map(|e| e.embedding).collect();
                x.into_iter().zip(values.into_iter()).collect()
            })
    }
}
