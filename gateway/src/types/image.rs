use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Image {
    pub b64_json: Option<String>,
    pub url: Option<String>,
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImagesResponse {
    pub created: Option<u32>,
    pub data: Vec<Image>,
}
