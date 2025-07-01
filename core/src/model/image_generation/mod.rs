use std::collections::HashMap;
use std::sync::Arc;

use crate::model::error::ModelError;
use langdb_open::OpenAISpecModel;
use ollama::OllamaImageGeneration;
use openai::OpenAIImageGeneration;
use serde::Serialize;
use serde_json::Value;
use tracing::info_span;
use tracing_futures::Instrument;
use valuable::Valuable;

use crate::events::{JsonValue, RecordResult, SPAN_MODEL_CALL};
use crate::model::types::ModelEventType;
use crate::types::engine::{ImageGenerationEngineParams, ImageGenerationModelDefinition};
use crate::types::gateway::{CostCalculator, CreateImageRequest, ImageGenerationModelUsage, Usage};
use crate::types::image::ImagesResponse;
use crate::GatewayResult;

use super::types::ModelEvent;
use super::CredentialsIdent;
use tokio::sync::mpsc::channel;

pub mod langdb_open;
pub mod ollama;
pub mod openai;

#[async_trait::async_trait]
pub trait ImageGenerationModelInstance: Sync + Send {
    async fn create_new(
        &self,
        request: &CreateImageRequest,
        tx: tokio::sync::mpsc::Sender<Option<ModelEvent>>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<ImagesResponse>;
}

fn initialize_image_generation_model_instance(
    definition: ImageGenerationModelDefinition,
    cost_calculator: Option<Arc<Box<dyn CostCalculator>>>,
    endpoint: Option<&str>,
    provider_name: Option<&str>,
) -> Result<Box<dyn ImageGenerationModelInstance>, ModelError> {
    match &definition.engine {
        ImageGenerationEngineParams::OpenAi {
            credentials,
            endpoint,
            model_name,
            ..
        } => Ok(Box::new(TracedImageGenerationModel {
            inner: OpenAIImageGeneration::new(
                credentials.clone().as_ref(),
                None,
                endpoint.as_ref().map(|s| s.as_str()),
            )?,
            definition: definition.clone(),
            cost_calculator: cost_calculator.clone(),
        })),
        ImageGenerationEngineParams::Ollama {
            credentials,
            endpoint,
            model_name,
            ..
        } => Ok(Box::new(TracedImageGenerationModel {
            inner: OllamaImageGeneration::new(
                model_name.clone(),
                credentials.clone(),
                endpoint.clone(),
            ),
            definition: definition.clone(),
            cost_calculator: cost_calculator.clone(),
        })),
        ImageGenerationEngineParams::LangdbOpen { credentials, .. } => {
            Ok(Box::new(TracedImageGenerationModel {
                inner: OpenAISpecModel::new(
                    credentials.clone().as_ref(),
                    endpoint,
                    provider_name.unwrap(),
                )?,
                definition: definition.clone(),
                cost_calculator: cost_calculator.clone(),
            }))
        }
    }
}

pub async fn initialize_image_generation(
    definition: ImageGenerationModelDefinition,
    cost_calculator: Option<Arc<Box<dyn CostCalculator>>>,
    endpoint: Option<&str>,
    provider_name: Option<&str>,
) -> Result<Box<dyn ImageGenerationModelInstance>, ModelError> {
    initialize_image_generation_model_instance(definition, cost_calculator, endpoint, provider_name)
}
pub struct TracedImageGenerationModel<Inner: ImageGenerationModelInstance> {
    inner: Inner,
    definition: ImageGenerationModelDefinition,
    cost_calculator: Option<Arc<Box<dyn CostCalculator>>>,
}

#[derive(Clone, Serialize)]
struct TracedImageGenerationModelDefinition {
    pub name: String,
    pub provider_name: String,
    pub engine_name: String,
    pub prompt_name: Option<String>,
    pub model_params: ImageGenerationModelDefinition,
    pub model_name: String,
}

impl TracedImageGenerationModelDefinition {
    pub fn sanitize_json(&self) -> GatewayResult<Value> {
        let mut model = self.clone();

        match &mut model.model_params.engine {
            ImageGenerationEngineParams::OpenAi {
                ref mut credentials,
                ..
            } => {
                credentials.take();
            }
            ImageGenerationEngineParams::LangdbOpen {
                ref mut credentials,
                ..
            } => {
                credentials.take();
            }
            ImageGenerationEngineParams::Ollama { .. } => {
                // Ollama 暂不支持 image generation
            }
        }
        let model = serde_json::to_value(&model)?;
        Ok(model)
    }

    pub fn get_credentials_owner(&self) -> CredentialsIdent {
        match &self.model_params.engine {
            ImageGenerationEngineParams::OpenAi { credentials, .. } => match &credentials {
                Some(_) => CredentialsIdent::Own,
                None => CredentialsIdent::Langdb,
            },
            ImageGenerationEngineParams::LangdbOpen { credentials, .. } => match &credentials {
                Some(_) => CredentialsIdent::Own,
                None => CredentialsIdent::Langdb,
            },
            ImageGenerationEngineParams::Ollama { .. } => CredentialsIdent::Langdb, // 默认
        }
    }
}

impl From<ImageGenerationModelDefinition> for TracedImageGenerationModelDefinition {
    fn from(value: ImageGenerationModelDefinition) -> Self {
        Self {
            model_name: value.db_model.name.clone(),
            name: value.name.clone(),
            provider_name: value.db_model.provider_name.clone(),
            engine_name: value.engine.engine_name().to_string(),
            prompt_name: None,
            model_params: value.clone(),
        }
    }
}

#[async_trait::async_trait]
impl<Inner: ImageGenerationModelInstance> ImageGenerationModelInstance
    for TracedImageGenerationModel<Inner>
{
    async fn create_new(
        &self,
        request: &CreateImageRequest,
        outer_tx: tokio::sync::mpsc::Sender<Option<ModelEvent>>,
        tags: HashMap<String, String>,
    ) -> GatewayResult<ImagesResponse> {
        let traced_model: TracedImageGenerationModelDefinition = self.definition.clone().into();
        let credentials_ident = traced_model.get_credentials_owner();
        let model = traced_model.sanitize_json()?;
        let model_str = serde_json::to_string(&model)?;
        let model_name = self.definition.name.clone();
        let provider_name = self.definition.db_model.provider_name.clone();
        let request_str = serde_json::to_string(request)?;

        let (tx, mut rx) = channel::<Option<ModelEvent>>(outer_tx.max_capacity());
        let span = info_span!(
            target: "langdb::user_tracing::models", SPAN_MODEL_CALL,
            input = &request_str,
            model = model_str,
            provider_name = provider_name,
            output = tracing::field::Empty,
            error = tracing::field::Empty,
            credentials_identifier = credentials_ident.to_string(),
            cost = tracing::field::Empty,
            usage = tracing::field::Empty,
            tags = JsonValue(&serde_json::to_value(tags.clone())?).as_value(),
        );

        let cost_calculator = self.cost_calculator.clone();
        tokio::spawn(
            async move {
                while let Some(Some(msg)) = rx.recv().await {
                    if let Some(cost_calculator) = cost_calculator.as_ref() {
                        if let ModelEventType::ImageGenerationFinish(generation_finish_event) =
                            &msg.event
                        {
                            let s = tracing::Span::current();
                            let u = ImageGenerationModelUsage {
                                quality: generation_finish_event.quality.clone(),
                                size: generation_finish_event.size.clone().into(),
                                images_count: generation_finish_event.count_of_images,
                                steps_count: generation_finish_event.steps,
                            };
                            match cost_calculator
                                .calculate_cost(
                                    &model_name,
                                    &provider_name,
                                    &Usage::ImageGenerationModelUsage(u.clone()),
                                )
                                .await
                            {
                                Ok(c) => {
                                    s.record("cost", serde_json::to_string(&c).unwrap());
                                }
                                Err(e) => {
                                    tracing::error!("Error calculating cost: {:?}", e);
                                }
                            };

                            s.record("usage", serde_json::to_string(&u).unwrap());
                        }
                    }

                    outer_tx.send(Some(msg)).await.unwrap();
                }
            }
            .instrument(span.clone()),
        );

        async {
            let result = self.inner.create_new(request, tx, tags).await;
            let _ = result.as_ref().map(|r| r.data.len()).record();

            result
        }
        .instrument(span)
        .await
    }
}
