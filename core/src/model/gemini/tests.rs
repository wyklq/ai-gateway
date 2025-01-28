use serde_json::Value;
use std::collections::HashMap;
use tokio_stream::StreamExt;

use super::{
    client::Client,
    types::{Content, GenerationConfig, Part, PartFunctionResponse},
};
use crate::model::gemini::types::{GenerateContentRequest, Role};
use base64::{engine::general_purpose::STANDARD, Engine};

fn get_config() -> GenerationConfig {
    GenerationConfig {
        max_output_tokens: Some(2048),
        temperature: Some(0.4),
        top_p: Some(1.0),
        top_k: Some(32),
        ..Default::default()
    }
}

fn sample_request() -> GenerateContentRequest {
    let json = include_str!("./tools.json");
    serde_json::from_str(json).unwrap()
}

#[tokio::test]
async fn test_models() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());
    let models = client.models().await.unwrap();
    println!("{models:?}");
}

#[tokio::test]
async fn test_invoke() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());

    let response = client
        .invoke(
            "gemini-1.5-flash",
            GenerateContentRequest {
                contents: vec![Content {
                    role: Role::User,
                    parts: vec![Part::Text("Tell me a large poem".to_string())],
                }],
                generation_config: Some(get_config()),
                tools: None,
            },
        )
        .await
        .unwrap();

    let mut text = String::new();
    response.candidates.iter().for_each(|candidate| {
        candidate.content.parts.iter().for_each(|part| {
            if let Part::Text(t) = part {
                text.push_str(t);
            }
        });
    });
    println!("{text}");
    assert!(!text.is_empty());
}

#[tokio::test]
async fn test_invoke_image() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());

    let image_data: &[u8] = include_bytes!("./pasta.jpg");
    let image_data = STANDARD.encode(image_data);

    let payload = GenerateContentRequest {
            contents: vec![Content {
                role: Role::User,
                parts: vec![
                    Part::Text("Guess the calories count and breakdown based on the image. GIve a json format back".to_string()),
                    Part::InlineData {
                        mime_type: "image/jpeg".to_string(),
                        data: image_data,
                    },
                ],
            }],
            tools: None,
            generation_config: Some(get_config()),
        };
    let response = client.invoke("gemini-1.5-flash", payload).await.unwrap();

    let mut text = String::new();
    response.candidates.iter().for_each(|candidate| {
        candidate.content.parts.iter().for_each(|part| {
            if let Part::Text(t) = part {
                text.push_str(t);
            }
        });
    });
    println!("{text}");
    assert!(!text.is_empty());
}

#[tokio::test]
async fn test_stream() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());

    let stream = client
        .stream(
            "gemini-1.5-flash",
            GenerateContentRequest {
                contents: vec![Content {
                    role: Role::User,
                    parts: vec![Part::Text("Tell me a large poem".to_string())],
                }],
                generation_config: Some(get_config()),
                tools: None,
            },
        )
        .await
        .unwrap();

    tokio::pin!(stream);
    let mut text = String::new();
    while let Some(Ok(res)) = stream.next().await {
        if let Some(res) = res {
            let res = res.clone();
            res.candidates.iter().for_each(|candidate| {
                candidate.content.parts.iter().for_each(|part| {
                    if let Part::Text(t) = part {
                        text.push_str(t);
                        print!("{t}");
                    }
                });
            });
        }
    }
    assert!(!text.is_empty());
}

#[tokio::test]
async fn test_tools() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());
    let response = client
        .invoke("gemini-1.5-flash", sample_request())
        .await
        .unwrap();
    let mut calls = vec![];
    println!("{response:?}");
    response.candidates.iter().for_each(|candidate| {
        candidate.content.parts.iter().for_each(|part| {
            if let Part::FunctionCall { name, args } = part {
                calls.push((name, args));
            }
        });
    });
    println!("{calls:?}");
    assert!(!calls.is_empty());
}

#[tokio::test]
async fn test_tools_stream() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());

    let stream = client
        .stream("gemini-1.5-flash", sample_request())
        .await
        .unwrap();
    let mut calls: Vec<String> = vec![];
    let mut text = String::new();
    tokio::pin!(stream);
    while let Some(Ok(res)) = stream.next().await {
        if let Some(res) = res {
            res.candidates.iter().for_each(|candidate| {
                println!("{:#?}", candidate);
                candidate.content.parts.iter().for_each(|part| {
                    if let Part::FunctionCall { name, args: _ } = part {
                        calls.push(name.to_string());
                    } else if let Part::Text(t) = part {
                        text.push_str(t);
                        print!("{t}");
                    }
                });
            });
        }
    }
    println!("{calls:?}");
    assert!(!calls.is_empty() || !text.is_empty());
}

#[tokio::test]
async fn test_tools_stream_response() {
    let client = Client::new(std::env::var("GOOGLE_API_KEY").unwrap());

    let model_response = Content {
        role: Role::Model,
        parts: vec![Part::FunctionCall {
            name: "describe_tables".to_string(),
            args: HashMap::new(),
        }],
    };
    let tool_response = Content {
        role: Role::User,
        parts: vec![Part::FunctionResponse {
            name: "describe_tables".to_string(),
            response: Some(PartFunctionResponse {
                fields: vec![(
                    "tables".to_string(),
                    Value::Array(vec!["a".to_string().into(), "b".to_string().into()]),
                )]
                .into_iter()
                .collect::<HashMap<_, Value>>(),
            }),
        }],
    };
    let mut request = sample_request();
    let mut contents = request.contents;
    contents.extend([model_response, tool_response]);
    request.contents = contents;
    let stream = client.stream("gemini-1.5-flash", request).await.unwrap();
    tokio::pin!(stream);
    let mut text = String::new();
    while let Some(Ok(res)) = stream.next().await {
        if let Some(res) = res {
            res.candidates.iter().for_each(|candidate| {
                println!("{:?}", candidate.content);
                candidate.content.parts.iter().for_each(|part| {
                    if let Part::Text(t) = part {
                        text.push_str(t);
                        print!("{t}");
                    }
                });
            });
        }
    }
    println!("{text}");
    assert!(!text.is_empty());
}
