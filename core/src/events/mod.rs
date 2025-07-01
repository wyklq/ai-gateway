use bytemuck::TransparentWrapper;
use opentelemetry::{baggage::BaggageExt as _, global::ObjectSafeSpan as _};
use opentelemetry_sdk::error::OTelSdkError;
use opentelemetry_sdk::trace::SpanProcessor;
use serde_json::Value;
use valuable::{Listable, Mappable, Valuable, Visit};
use std::fmt;

mod layer;
pub use layer::{config, layer, RecordResult, UuidIdGenerator};

pub const SPAN_QUERY: &str = "query";

pub const SPAN_QUERY_ENTITIES: &str = "query_entities";

pub const SPAN_API_STREAM: &str = "api_stream";

pub const SPAN_API_INVOKE: &str = "api_invoke";

pub const SPAN_OPENAI: &str = "openai";

pub const SPAN_GEMINI: &str = "gemini";

pub const SPAN_ANTHROPIC: &str = "anthropic";

pub const SPAN_BEDROCK: &str = "bedrock";

pub const SPAN_CACHE: &str = "cache";

pub const SPAN_TOOLS: &str = "tools";

pub const SPAN_TOOL: &str = "tool";

pub const SPAN_MODEL_CALL: &str = "model_call";

pub const SPAN_OPENAI_SPEC: &str = "openai_spec";

pub const SPAN_REQUEST_ROUTING: &str = "request_routing";

pub const SPAN_GUARD_EVAULATION: &str = "guard_evaluation";

pub const SPAN_VIRTUAL_MODEL: &str = "virtual_model";

#[derive(Debug)]
pub struct BaggageSpanProcessor<const N: usize> {
    keys: [&'static str; N],
}

impl<const N: usize> BaggageSpanProcessor<N> {
    pub const fn new(keys: [&'static str; N]) -> Self {
        Self { keys }
    }
}

impl<const N: usize> SpanProcessor for BaggageSpanProcessor<N> {
    fn on_start(&self, span: &mut opentelemetry_sdk::trace::Span, cx: &opentelemetry::Context) {
        for key in self.keys {
            let value = cx.baggage().get(key);
            if let Some(value) = value {
                span.set_attribute(opentelemetry::KeyValue::new(key, value.clone()));
            }
        }
    }

    fn on_end(&self, _span: opentelemetry_sdk::trace::SpanData) {}

    fn force_flush(&self) -> std::result::Result<(), OTelSdkError> {
        Ok(())
    }

    fn shutdown(&self) -> std::result::Result<(), OTelSdkError> {
        Ok(())
    }

    fn shutdown_with_timeout(
        &self,
        _timeout: std::time::Duration,
    ) -> std::result::Result<(), OTelSdkError> {
        Ok(())
    }
}

// Simple wrapper that converts JsonValue to a value usable by tracing
pub struct TracingValue<'a>(pub &'a Value);

impl<'a> fmt::Display for TracingValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self.0).unwrap_or_default())
    }
}

impl<'a> fmt::Debug for TracingValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string_pretty(self.0).unwrap_or_default())
    }
}

// Updated JsonValue struct that implements field::Value
pub struct JsonValue<'a>(pub &'a Value);

impl<'a> JsonValue<'a> {
    // Convert to a tracing-compatible value
    pub fn tracing_value(&self) -> TracingValue<'a> {
        TracingValue(self.0)
    }
}

// Implement field::Visit for TracingValue so it can be used with Span::record
impl<'a> valuable::Visit for TracingValue<'a> {
    fn visit_value(&mut self, _: valuable::Value<'_>) {
        // Implementation needed
    }
}

// Instead of implementing field::Value directly, let's use Display to debug record
impl<'a> TracingValue<'a> {
    pub fn record_to_span(&self, span: &tracing::Span, field_name: &str) {
        span.record(field_name, &self.to_string());
    }
}

// Implementing wrapper for owned JsonValue to make it compatible with tracing::Value
pub struct JsonValueOwned(pub Value);

#[derive(TransparentWrapper)]
#[repr(transparent)]
struct JsonMap(serde_json::Map<String, Value>);

#[derive(TransparentWrapper)]
#[repr(transparent)]
struct JsonArray(Vec<Value>);

impl Valuable for JsonValueOwned {
    fn as_value(&self) -> valuable::Value<'_> {
        match self.0 {
            Value::Array(ref array) => JsonArray::wrap_ref(array).as_value(),
            Value::Bool(ref value) => value.as_value(),
            Value::Number(ref num) => {
                if num.is_f64() {
                    valuable::Value::F64(num.as_f64().unwrap())
                } else if num.is_i64() {
                    valuable::Value::I64(num.as_i64().unwrap())
                } else {
                    unreachable!()
                }
            }
            Value::Null => valuable::Value::Unit,
            Value::String(ref s) => s.as_value(),
            Value::Object(ref object) => JsonMap::wrap_ref(object).as_value(),
        }
    }

    fn visit(&self, visit: &mut dyn Visit) {
        match self.0 {
            Value::Array(ref array) => JsonArray::wrap_ref(array).visit(visit),
            Value::Bool(ref value) => value.visit(visit),
            Value::Number(ref num) => {
                if num.is_f64() {
                    num.as_f64().unwrap().visit(visit)
                } else if num.is_i64() {
                    num.as_i64().unwrap().visit(visit)
                } else {
                    unreachable!()
                }
            }
            Value::Null => valuable::Value::Unit.visit(visit),
            Value::String(ref s) => s.visit(visit),
            Value::Object(ref object) => JsonMap::wrap_ref(object).visit(visit),
        }
    }
}
impl Valuable for JsonValue<'_> {
    fn as_value(&self) -> valuable::Value<'_> {
        match self.0 {
            Value::Array(ref array) => JsonArray::wrap_ref(array).as_value(),
            Value::Bool(ref value) => value.as_value(),
            Value::Number(ref num) => {
                if num.is_f64() {
                    valuable::Value::F64(num.as_f64().unwrap())
                } else if num.is_i64() {
                    valuable::Value::I64(num.as_i64().unwrap())
                } else {
                    unreachable!()
                }
            }
            Value::Null => valuable::Value::Unit,
            Value::String(ref s) => s.as_value(),
            Value::Object(ref object) => JsonMap::wrap_ref(object).as_value(),
        }
    }

    fn visit(&self, visit: &mut dyn Visit) {
        match self.0 {
            Value::Array(ref array) => JsonArray::wrap_ref(array).visit(visit),
            Value::Bool(ref value) => value.visit(visit),
            Value::Number(ref num) => {
                if num.is_f64() {
                    num.as_f64().unwrap().visit(visit)
                } else if num.is_i64() {
                    num.as_i64().unwrap().visit(visit)
                } else {
                    unreachable!()
                }
            }
            Value::Null => valuable::Value::Unit.visit(visit),
            Value::String(ref s) => s.visit(visit),
            Value::Object(ref object) => JsonMap::wrap_ref(object).visit(visit),
        }
    }
}

impl Valuable for JsonMap {
    fn as_value(&self) -> valuable::Value<'_> {
        valuable::Value::Mappable(self)
    }

    fn visit(&self, visit: &mut dyn Visit) {
        for (k, v) in self.0.iter() {
            visit.visit_entry(k.as_value(), JsonValue(v).as_value());
        }
    }
}

impl Mappable for JsonMap {
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len();
        (len, Some(len))
    }
}

impl Valuable for JsonArray {
    fn as_value(&self) -> valuable::Value<'_> {
        valuable::Value::Listable(self)
    }

    fn visit(&self, visit: &mut dyn Visit) {
        for v in self.0.iter() {
            visit.visit_value(JsonValue(v).as_value())
        }
    }
}

impl Listable for JsonArray {
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len();
        (len, Some(len))
    }
}
