use std::fmt::Display;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use serde_with::serde_as;

use super::{gateway::ToolCall, message::MessageType};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MessageThread {
    pub id: String,         // UUID
    pub model_name: String, // Corresponding LangDB model
    pub user_id: String,    // UUID
    pub project_id: String, // Project identifier
    pub is_public: bool,
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PublicMessageThread {
    pub id: String, // UUID
    pub is_public: bool,
    pub tenant_id: String,
}

#[serde_as]
#[derive(Serialize, Debug, Clone)]
pub struct Message {
    pub model_name: String,        // Corresponding LangDB model
    pub thread_id: Option<String>, // Identifier of the thread to which this message belongs
    pub user_id: String,           // UUID
    pub content_type: MessageContentType,
    pub content: Option<String>,
    pub content_array: Vec<MessageContentPart>,
    pub r#type: MessageType, // Human / AI Message
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            model_name: String,
            thread_id: Option<String>,
            user_id: String,
            content_type: MessageContentType,
            content: Option<String>,
            content_array: Vec<MessageContentPart>,
            r#type: MessageType,
            tool_call_id: Option<String>,
            tool_calls: Option<serde_json::Value>,
        }

        let helper = Helper::deserialize(deserializer)?;

        let tool_calls = match helper.tool_calls {
            Some(Value::String(s)) => serde_json::from_str(&s).map_err(serde::de::Error::custom)?,
            Some(Value::Array(_)) => helper.tool_calls,
            _ => None,
        };

        Ok(Message {
            model_name: helper.model_name,
            thread_id: helper.thread_id,
            user_id: helper.user_id,
            content_type: helper.content_type,
            content: helper.content,
            content_array: helper.content_array,
            r#type: helper.r#type,
            tool_call_id: helper.tool_call_id,
            tool_calls: tool_calls.and_then(|v| serde_json::from_value(v).ok()),
        })
    }
}

// Value is deserialized into this object selectively
// by a prompt
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum InnerMessage {
    Text(String),
    Array(Vec<MessageContentPart>),
}

impl From<Message> for InnerMessage {
    fn from(val: Message) -> Self {
        match val.content_array.len() {
            0 => InnerMessage::Text(val.content.unwrap_or_default()),
            _ => InnerMessage::Array(val.content_array),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MessageContentPart {
    pub r#type: MessageContentType,
    pub value: String,
    pub additional_options: Option<MessageContentPartOptions>,
}

// Custom serialization to a stable, self-descriptive JSON object preserving multimodal data
impl serde::Serialize for MessageContentPart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        // Normalize type to lowercase (OpenAI style) except we keep existing names for backward clarity
        let type_str = match self.r#type {
            MessageContentType::Text => "text",
            MessageContentType::ImageUrl => "image_url",
            MessageContentType::InputAudio => "input_audio",
        };
        map.serialize_entry("type", type_str)?;
        match self.r#type {
            MessageContentType::Text => {
                map.serialize_entry("text", &self.value)?;
            }
            MessageContentType::ImageUrl => {
                // Provide nested object with url, optionally detail from additional_options
                #[derive(serde::Serialize)]
                struct Img<'a> { url: &'a str, #[serde(skip_serializing_if = "Option::is_none")] detail: Option<&'a str> }
                let detail = self.additional_options.as_ref().and_then(|opt| match opt {
                    MessageContentPartOptions::Image(d) => Some(match d { ImageDetail::Auto => "auto", ImageDetail::Low => "low", ImageDetail::High => "high" }),
                    _ => None,
                });
                let img = Img { url: &self.value, detail };
                map.serialize_entry("image_url", &img)?;
            }
            MessageContentType::InputAudio => {
                // Provide audio object with data and format if available
                #[derive(serde::Serialize)]
                struct Aud<'a> { data: &'a str, #[serde(skip_serializing_if = "Option::is_none")] format: Option<&'a str> }
                let format = self.additional_options.as_ref().and_then(|opt| match opt {
                    MessageContentPartOptions::Audio(AudioDetail { r#type: AudioFormat::Mp3 }) => Some("mp3"),
                    MessageContentPartOptions::Audio(AudioDetail { r#type: AudioFormat::Wav }) => Some("wav"),
                    _ => None,
                });
                let aud = Aud { data: &self.value, format };
                map.serialize_entry("audio", &aud)?;
            }
        }
        // Raw value (for backward or debugging usage)
        map.serialize_entry("raw", &self.value)?;
        if let Some(opts) = &self.additional_options {
            map.serialize_entry("options", opts)?;
        }
        map.end()
    }
}

// Custom deserialization matching the above representation
impl<'de> serde::Deserialize<'de> for MessageContentPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, MapAccess, Visitor};
        use std::fmt;
        struct PartVisitor;
        impl<'de> Visitor<'de> for PartVisitor {
            type Value = MessageContentPart;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a message content part object")
            }
            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut type_str: Option<String> = None;
                let mut text: Option<String> = None;
                let mut image_url: Option<serde_json::Value> = None;
                let mut audio: Option<serde_json::Value> = None;
                let mut raw: Option<String> = None;
                let mut options: Option<MessageContentPartOptions> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "type" => type_str = map.next_value()?,
                        "text" => text = map.next_value()?,
                        "image_url" => image_url = map.next_value()?,
                        "audio" => audio = map.next_value()?,
                        "raw" => raw = map.next_value()?,
                        "options" => options = map.next_value()?,
                        _ => {
                            let _ignored: serde_json::Value = map.next_value()?; // ignore unknown
                        }
                    }
                }
                let t = type_str.ok_or_else(|| M::Error::missing_field("type"))?;
                let part_type = match t.as_str() {
                    "text" => MessageContentType::Text,
                    "image_url" => MessageContentType::ImageUrl,
                    "input_audio" => MessageContentType::InputAudio,
                    other => return Err(M::Error::custom(format!("unsupported part type {other}"))),
                };
                let value = match part_type {
                    MessageContentType::Text => text.or(raw).ok_or_else(|| M::Error::custom("missing text value"))?,
                    MessageContentType::ImageUrl => {
                        // Expect image_url: { url: ... }
                        if let Some(v) = image_url {
                            v.get("url").and_then(|u| u.as_str()).map(|s| s.to_string()).ok_or_else(|| M::Error::custom("image_url.url missing"))?
                        } else {
                            raw.ok_or_else(|| M::Error::custom("missing image_url value"))?
                        }
                    }
                    MessageContentType::InputAudio => {
                        if let Some(v) = audio {
                            v.get("data").and_then(|u| u.as_str()).map(|s| s.to_string()).ok_or_else(|| M::Error::custom("audio.data missing"))?
                        } else {
                            raw.ok_or_else(|| M::Error::custom("missing audio value"))?
                        }
                    }
                };
                Ok(MessageContentPart { r#type: part_type, value, additional_options: options })
            }
        }
        deserializer.deserialize_map(PartVisitor)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub enum MessageContentType {
    #[default]
    Text,
    ImageUrl,
    InputAudio,
}

impl Display for MessageContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContentType::Text => f.write_str("Text"),
            MessageContentType::ImageUrl => f.write_str("ImageUrl"),
            MessageContentType::InputAudio => f.write_str("InputAudio"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MessageContentValue {
    Text(String),
    ImageUrl(Vec<MessageContentPart>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MessageContentPartOptions {
    Image(ImageDetail),
    Audio(AudioDetail),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioDetail {
    pub r#type: AudioFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AudioFormat {
    Mp3,
    Wav,
}
impl MessageContentPartOptions {
    pub fn as_image(&self) -> Option<ImageDetail> {
        match self {
            MessageContentPartOptions::Image(image) => Some(image.to_owned()),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageRequest {
    pub model_name: String,
    pub thread_id: Option<String>,
    pub user_id: String,
    pub parameters: IndexMap<String, serde_json::Value>,
    pub message: InnerMessage,
    #[serde(default = "default_include_history")]
    pub include_history: bool,
    #[serde(default)]
    pub history_length: Option<u32>,
}

pub fn default_include_history() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use crate::types::threads::MessageContentPart;

    #[test]
    fn message_serialization() {
        let test = vec![
            MessageContentPart {
                r#type: super::MessageContentType::ImageUrl,
                value: "https://example.com/image.png".to_string(),
                additional_options: Some(super::MessageContentPartOptions::Image(super::ImageDetail::High)),
            },
            MessageContentPart {
                r#type: super::MessageContentType::Text,
                value: "Caption for image".to_string(),
                additional_options: None,
            },
        ];
        let val = serde_json::to_value(&test).unwrap();
        // Ensure object-based serialization with preserved fields
        assert_eq!(val, json!([
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "high"},
                "raw": "https://example.com/image.png",
                "options": {"Image":"High"}
            },
            {
                "type": "text",
                "text": "Caption for image",
                "raw": "Caption for image"
            }
        ]));
    }
}
