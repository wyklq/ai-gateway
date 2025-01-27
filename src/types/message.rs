use std::{collections::HashSet, fmt::Display};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum MessageType {
    #[serde(rename = "system")]
    SystemMessage,
    #[serde(rename = "ai")]
    AIMessage,
    #[serde(rename = "human")]
    HumanMessage,
    #[serde(rename = "tool")]
    ToolResult,
}

impl Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageType::SystemMessage => f.write_str("system"),
            MessageType::AIMessage => f.write_str("ai"),
            MessageType::HumanMessage => f.write_str("human"),
            MessageType::ToolResult => f.write_str("tool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PromptMessage {
    pub r#type: MessageType,
    pub msg: String,
    #[serde(default)]
    pub wired: bool,
    pub parameters: HashSet<String>,
}

impl Display for PromptMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.r#type, self.msg)
    }
}
