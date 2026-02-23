//! Z.AI API client implementation for chat and completion functionality.
//!
//! This module provides integration with Z.AI's GLM models through their API.
//! Z.AI uses an OpenAI-compatible API, so we leverage the OpenAICompatibleProvider.
//!
//! # Features
//!
//! - **Structured Output**: Supports `json_object` response format
//! - **Thinking/Reasoning**: GLM-5 returns `reasoning_content` accessible via `response.thinking()`
//! - **Streaming**: Full support for streaming responses
//! - **Tool Calling**: Supports function calling with tools
//!
//! # Endpoints
//!
//! - Default (Coding Plan): `https://api.z.ai/api/coding/paas/v4/`
//! - General: `https://api.z.ai/api/paas/v4/`
//!
//! # Example
//!
//! ```text
//! use autoagents_llm::backends::zai::Zai;
//! use autoagents_llm::builder::LLMBuilder;
//! use autoagents_llm::chat::{ChatMessage, ChatProvider};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let llm: Arc<Zai> = LLMBuilder::<Zai>::new()
//!         .api_key(std::env::var("ZAI_API_KEY").unwrap())
//!         .model("glm-5")
//!         .build()
//!         .unwrap();
//!
//!     let messages = vec![ChatMessage::user().content("Hello!").build()];
//!
//!     // Access reasoning content (thinking) from GLM-5
//!     let response = llm.chat(&messages, None).await.unwrap();
//!     if let Some(thinking) = response.thinking() {
//!         println!("Model reasoning: {}", thinking);
//!     }
//!     println!("Response: {}", response.text().unwrap());
//! }
//! ```

use crate::chat::{
    ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse, StructuredOutputFormat,
    Tool,
};
use crate::providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig};
use crate::{
    LLMProvider,
    builder::LLMBuilder,
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;

/// Z.AI configuration for the OpenAI-compatible provider
struct ZaiConfig;

impl OpenAIProviderConfig for ZaiConfig {
    const PROVIDER_NAME: &'static str = "Z.AI";
    const DEFAULT_BASE_URL: &'static str = "https://api.z.ai/api/coding/paas/v4/";
    const DEFAULT_MODEL: &'static str = "glm-5";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

/// Client for Z.AI API
///
/// Provides access to Z.AI's GLM family of models including GLM-5, GLM-4.7, and others.
/// The default endpoint is the Coding Plan endpoint optimized for code generation tasks.
pub struct Zai {
    provider: OpenAICompatibleProvider<ZaiConfig>,
}

impl Zai {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tool_choice: Option<crate::chat::ToolChoice>,
        extra_body: Option<serde_json::Value>,
        normalize_response: Option<bool>,
    ) -> Self {
        Self {
            provider: OpenAICompatibleProvider::new(
                api_key,
                base_url,
                model,
                max_tokens,
                temperature,
                timeout_seconds,
                top_p,
                top_k,
                tool_choice,
                None,
                None,
                extra_body,
                None,
                normalize_response,
                None,
                None,
            ),
        }
    }

    pub fn api_key(&self) -> &str {
        &self.provider.api_key
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }
}

#[async_trait]
impl ChatProvider for Zai {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.provider.chat(messages, json_schema).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.provider
            .chat_with_tools(messages, tools, json_schema)
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        self.provider.chat_stream(messages, json_schema).await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        self.provider
            .chat_stream_struct(messages, tools, json_schema)
            .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        self.provider
            .chat_stream_with_tools(messages, tools, json_schema)
            .await
    }
}

#[async_trait]
impl CompletionProvider for Zai {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        if self.api_key().is_empty() {
            return Err(LLMError::AuthError("Missing Z.AI API key".into()));
        }
        Err(LLMError::ProviderError(
            "Z.AI completion not implemented yet".into(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for Zai {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Zai {}

impl LLMProvider for Zai {}

impl LLMBuilder<Zai> {
    pub fn build(self) -> Result<Arc<Zai>, LLMError> {
        let api_key = self
            .api_key
            .ok_or_else(|| LLMError::InvalidRequest("No API key provided for Z.AI".to_string()))?;

        let zai = Zai::new(
            api_key,
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.tool_choice,
            self.extra_body,
            self.normalize_response,
        );

        Ok(Arc::new(zai))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::completion::CompletionRequest;
    use futures::StreamExt;

    #[test]
    fn test_new_defaults() {
        let client = Zai::new(
            "key", None, None, None, None, None, None, None, None, None, None,
        );
        assert_eq!(client.api_key(), "key");
        assert_eq!(client.model(), "glm-5");
    }

    #[test]
    fn test_new_with_base_url() {
        let client = Zai::new(
            "key",
            Some("https://api.z.ai/api/paas/v4/".to_string()),
            Some("glm-4.7".to_string()),
            Some(1024),
            Some(0.8),
            Some(60),
            Some(0.95),
            Some(10),
            None,
            None,
            None,
        );
        assert_eq!(client.provider.model, "glm-4.7");
        assert_eq!(
            client.provider.base_url.as_str(),
            "https://api.z.ai/api/paas/v4/"
        );
        assert_eq!(client.provider.max_tokens, Some(1024));
        assert_eq!(client.provider.temperature, Some(0.8));
        assert_eq!(client.provider.top_p, Some(0.95));
        assert_eq!(client.provider.top_k, Some(10));
    }

    #[test]
    fn test_new_with_extra_body() {
        let extra = serde_json::json!({"custom_field": "value"});
        let client = Zai::new(
            "key",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(extra.clone()),
            None,
        );
        assert_eq!(
            client.provider.extra_body.get("custom_field"),
            Some(&serde_json::json!("value"))
        );
    }

    #[tokio::test]
    async fn test_complete_missing_key() {
        let client = Zai::new(
            "", None, None, None, None, None, None, None, None, None, None,
        );
        let err = client
            .complete(
                &CompletionRequest {
                    prompt: "hi".to_string(),
                    max_tokens: None,
                    temperature: None,
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Missing Z.AI API key"));
    }

    #[tokio::test]
    async fn test_embed_not_supported() {
        let client = Zai::new(
            "key", None, None, None, None, None, None, None, None, None, None,
        );
        let err = client.embed(vec!["hello".to_string()]).await.unwrap_err();
        assert!(err.to_string().contains("Embedding not supported"));
    }

    #[test]
    fn test_builder_requires_api_key() {
        let result = LLMBuilder::<Zai>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided"));
    }

    #[test]
    fn test_builder_with_all_options() {
        let zai = LLMBuilder::<Zai>::new()
            .api_key("test_key")
            .base_url("https://api.z.ai/api/paas/v4/")
            .model("glm-4.7")
            .max_tokens(2048)
            .temperature(0.7)
            .timeout_seconds(120)
            .top_p(0.9)
            .top_k(10)
            .extra_body(serde_json::json!({"custom": "field"}))
            .normalize_response(false)
            .build()
            .unwrap();

        assert_eq!(zai.api_key(), "test_key");
        assert_eq!(zai.model(), "glm-4.7");
        assert_eq!(zai.provider.top_k, Some(10));
        assert_eq!(zai.provider.normalize_response, false);
    }

    #[tokio::test]
    #[ignore]
    async fn smoke_test_zai() {
        let api_key =
            std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY environment variable must be set");

        let zai = LLMBuilder::<Zai>::new().api_key(api_key).build().unwrap();

        let messages = vec![
            crate::chat::ChatMessage::user()
                .content("Say hello.")
                .build(),
        ];

        let response = zai.chat(&messages, None).await.unwrap();

        println!("Response debug: {:?}", response);
        let text = response.text();
        println!("Response text: {:?}", text);

        let text = text.expect("Expected response text");
        assert!(!text.is_empty(), "Response text should not be empty");
    }

    #[tokio::test]
    #[ignore]
    async fn smoke_test_zai_thinking() {
        let api_key =
            std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY environment variable must be set");

        let zai = LLMBuilder::<Zai>::new().api_key(api_key).build().unwrap();

        let messages = vec![
            crate::chat::ChatMessage::user()
                .content("What is 15 + 27? Show your reasoning.")
                .build(),
        ];

        let response = zai.chat(&messages, None).await.unwrap();

        println!("Response debug: {:?}", response);

        let text = response.text().expect("Expected response text");
        println!("Response text: {}", text);
        assert!(!text.is_empty(), "Response text should not be empty");

        let thinking = response.thinking();
        println!("Thinking content: {:?}", thinking);
    }

    #[tokio::test]
    #[ignore]
    async fn smoke_test_zai_streaming() {
        let api_key =
            std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY environment variable must be set");

        let zai = LLMBuilder::<Zai>::new().api_key(api_key).build().unwrap();

        let messages = vec![
            crate::chat::ChatMessage::user()
                .content("Count from 1 to 5, one number per line.")
                .build(),
        ];

        let mut stream = zai.chat_stream(&messages, None).await.unwrap();

        let mut collected = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("Stream chunk should not error");
            print!("{}", chunk);
            collected.push_str(&chunk);
        }
        println!();

        assert!(
            !collected.is_empty(),
            "Streamed content should not be empty"
        );
    }
}
