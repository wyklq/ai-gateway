use serde::{Deserialize, Serialize};
use validator::Validate;
use validator::ValidationError;

#[derive(Serialize, Deserialize, Validate, Clone, Debug)]
#[validate(schema(function = "validate_openai_embedding_params"))]
pub struct OpenAiEmbeddingParams {
    pub model: Option<String>,
    // This can be up to 3072 for text-embedding-3-larg and up to 1536 for text-embedding-3-small
    // For older models, this parameter is not supported
    pub dimensions: Option<u16>,
}

fn validate_openai_embedding_params(it: &OpenAiEmbeddingParams) -> Result<(), ValidationError> {
    if let Some(dimensions) = it.dimensions {
        if let Some(ref model) = it.model {
            if model == "text-embedding-3-large" {
                if dimensions > 3072 {
                    let mut err = ValidationError::new("range");
                    err.add_param("value".into(), &dimensions);
                    return Err(err);
                }
            } else if model == "text-embedding-3-small" {
                if dimensions > 1536 {
                    let mut err = ValidationError::new("range");
                    err.add_param("value".into(), &dimensions);
                    return Err(err);
                }
            } else {
                return Err(ValidationError::new("invalid_param")
                    .with_message(std::borrow::Cow::Borrowed("Invalid parameter `dimensions`")));
            }
        }
    }
    Ok(())
}
