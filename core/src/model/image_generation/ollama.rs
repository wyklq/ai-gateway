use std::collections::HashMap;
use tokio::sync::mpsc::Sender;
use crate::model::image_generation::ImageGenerationModelInstance;
use crate::types::gateway::CreateImageRequest;
use crate::types::image::ImagesResponse;
use crate::model::types::ModelEvent;
use crate::GatewayResult;

/// Ollama 暂不支持 image generation，所有实现均为 todo
pub struct OllamaImageGeneration {
    // ...existing fields...
}

impl OllamaImageGeneration {
    pub fn new(
        _model_name: String,
        _credentials: Option<crate::types::credentials::ApiKeyCredentials>,
        _endpoint: Option<String>,
    ) -> Self {
        todo!("Ollama image generation is not supported yet")
    }
}

#[async_trait::async_trait]
impl ImageGenerationModelInstance for OllamaImageGeneration {
    async fn create_new(
        &self,
        _request: &CreateImageRequest,
        _tx: Sender<Option<ModelEvent>>,
        _tags: HashMap<String, String>,
    ) -> GatewayResult<ImagesResponse> {
        // Ollama 暂不支持 image generation
        unimplemented!("Ollama image generation is not supported yet")
    }
}
