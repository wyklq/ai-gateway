// This module provides adapters for aws_smithy_types::Document to be used with serde_json
use aws_smithy_types::Document;
use serde::Deserialize;
use serde::{Deserializer, Serializer};
use serde_json::Value;

// Convert Document to string for JSON serialization
pub fn serialize_document<S>(doc: &Document, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let json_str = match doc {
        Document::String(s) => s.clone(),
        _ => {
            let value = document_to_json_value(doc)
                .map_err(|e| serde::ser::Error::custom(format!("Document conversion error: {}", e)))?;
            serde_json::to_string(&value)
                .map_err(|e| serde::ser::Error::custom(format!("JSON serialization error: {}", e)))?
        }
    };
    serializer.serialize_str(&json_str)
}

// Convert from string to Document for JSON deserialization
pub fn deserialize_document<'de, D>(deserializer: D) -> Result<Document, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = String::deserialize(deserializer)?;
    let value: Value = serde_json::from_str(&s)
        .map_err(|e| serde::de::Error::custom(format!("JSON parse error: {}", e)))?;
    
    json_value_to_document(&value)
        .map_err(|e| serde::de::Error::custom(format!("Document conversion error: {}", e)))
}

// Helper function to convert Document to serde_json Value
pub fn document_to_json_value(doc: &Document) -> Result<Value, String> {
    match doc {
        Document::String(s) => {
            // If the string is valid JSON, parse it; otherwise, treat it as a regular string
            match serde_json::from_str(s) {
                Ok(value) => Ok(value),
                Err(_) => Ok(Value::String(s.clone())),
            }
        },
        Document::Bool(b) => Ok(Value::Bool(*b)),
        Document::Number(n) => {
            // Since aws_smithy_types::Number doesn't implement Display,
            // we'll serialize it to a JSON value and then extract that
            match serde_json::to_value(n) {
                Ok(value) => Ok(value),
                Err(e) => Err(format!("Failed to convert number: {}", e))
            }
        },
        Document::Array(arr) => {
            let mut values = Vec::new();
            for item in arr {
                values.push(document_to_json_value(item)?);
            }
            Ok(Value::Array(values))
        },
        Document::Object(m) => {
            let mut map = serde_json::Map::new();
            for (k, v) in m {
                map.insert(k.clone(), document_to_json_value(v)?);
            }
            Ok(Value::Object(map))
        },
        Document::Null => Ok(Value::Null),
    }
}

// Helper function to convert serde_json Value to Document
pub fn json_value_to_document(value: &Value) -> Result<Document, String> {
    match value {
        Value::String(s) => Ok(Document::String(s.clone())),
        Value::Bool(b) => Ok(Document::Bool(*b)),
        Value::Number(_) => {
            // Convert to string and store as string to avoid Number conversion issues
            Ok(Document::String(value.to_string()))
        },
        Value::Array(arr) => {
            let mut docs = Vec::new();
            for item in arr {
                docs.push(json_value_to_document(item)?);
            }
            Ok(Document::Array(docs))
        },
        Value::Object(obj) => {
            let mut map = std::collections::HashMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), json_value_to_document(v)?);
            }
            Ok(Document::Object(map))
        },
        Value::Null => Ok(Document::Null),
    }
}

// Helper function to safely convert a JSON string to Document
pub fn parse_json_to_document(json_str: &str) -> Result<Document, String> {
    let value: Value = serde_json::from_str(json_str)
        .map_err(|e| format!("JSON parse error: {}", e))?;
    json_value_to_document(&value)
}
