use directories::BaseDirs;
use langdb_core::models::ModelMetadata;
use reqwest;
use serde_yaml;
use std::fs;

#[derive(Debug, thiserror::Error)]
pub enum ModelsLoadError {
    #[error("Failed to fetch models: {0}")]
    FetchError(#[from] reqwest::Error),
    #[error("Failed to store models: {0}")]
    StoreError(#[from] std::io::Error),
    #[error("Failed to parse models config: {0}")]
    ParseError(#[from] serde_yaml::Error),
    #[error("Could not determine home directory")]
    NoHomeDir,
}

/// Load models configuration from the filesystem, fetching it first if it doesn't exist
pub async fn load_models(force_update: bool) -> Result<Vec<ModelMetadata>, ModelsLoadError> {
    let models_yaml = if force_update {
        // Force fetch and store new models
        fetch_and_store_models().await?
    } else {
        get_models_path()?
    };
    let models: Vec<ModelMetadata> = serde_yaml::from_str(&models_yaml)?;
    Ok(models)
}

pub async fn fetch_and_store_models() -> Result<String, ModelsLoadError> {
    // Create .langdb directory in home folder
    let base_dirs = BaseDirs::new().ok_or(ModelsLoadError::NoHomeDir)?;
    let langdb_dir = base_dirs.home_dir().join(".langdb");
    fs::create_dir_all(&langdb_dir)?;

    // Fetch models from API
    // let client = reqwest::Client::new();
    // let response = client
    //    .get("https://api.us-east-1.langdb.ai/pricing")
    //    .send()
    //    .await?
    //    .json::<serde_json::Value>()
    //    .await?;
    // Convert to YAML
    //    let yaml = serde_yaml::to_string(&response)?;
    let yaml = r#"
- model: o1-mini
  model_provider: openai
  inference_provider:
    provider: openai
    model_name: o1-mini
    endpoint: null
  price:
    per_input_token: 3.0
    per_output_token: 12.0
    valid_from: null
  input_formats:
  - text
  output_formats:
  - text
  capabilities: []
  type: completions
  limits:
    max_context_size: 128000
  description: The o1 series of large language models are trained with reinforcement learning to perform complex reasoning. o1 models think before they answer, producing a long internal chain of thought before responding to the user. Faster and cheaper reasoning model particularly good at coding, math, and science
  parameters:
    max_tokens:
      default: 1000
      description: The maximum number of tokens that can be generated in the completion. The token count of your prompt plus max_tokens cannot exceed the model's context length.
      max: null
      min: null
      required: false
      type: int
    seed:
      default: null
      description: If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend.
      max: null
      min: null
      required: false
      step: 1
      type: int
    "#.to_string(); // Placeholder for actual YAML content

    // Store in models.yaml
    let models_path = langdb_dir.join("models.yaml");
    fs::write(&models_path, &yaml)?;

    Ok(yaml)
}

pub fn get_models_path() -> Result<String, std::io::Error> {
    if let Some(base_dirs) = BaseDirs::new() {
        let user_models = base_dirs.home_dir().join(".langdb").join("models.yaml");
        if user_models.exists() {
            return std::fs::read_to_string(user_models);
        }
    }
    Ok(include_str!("../../models.yaml").to_string())
}
