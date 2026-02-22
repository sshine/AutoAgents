//! z.ai (Zhipu AI) API client implementation for chat functionality.
//!
//! This module provides integration with z.ai's GLM models through their API.
//! z.ai uses an OpenAI-compatible API, so we leverage the OpenAICompatibleProvider.

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

/// z.ai configuration for the OpenAI-compatible provider
struct ZAIConfig;

impl OpenAIProviderConfig for ZAIConfig {
    const PROVIDER_NAME: &'static str = "ZAI";
    const DEFAULT_BASE_URL: &'static str = "https://api.z.ai/api/coding/paas/v4/";
    const DEFAULT_MODEL: &'static str = "glm-5";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

/// Client for z.ai API
pub struct ZAI {
    provider: OpenAICompatibleProvider<ZAIConfig>,
}

impl ZAI {
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
    ) -> Self {
        Self {
            provider: OpenAICompatibleProvider::new(
                api_key,
                None, // base_url - use default
                model,
                max_tokens,
                temperature,
                timeout_seconds,
                None, // top_p
                None, // top_k
                None, // tool_choice
                None, // reasoning_effort
                None, // voice
                None, // extra_body
                None, // parallel_tool_calls
                None, // normalize_response
                None, // embedding_encoding_format
                None, // embedding_dimensions
            ),
        }
    }

    /// Creates a new z.ai client with extended configuration options.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        tool_choice: Option<crate::chat::ToolChoice>,
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
                None, // top_k
                tool_choice,
                None, // reasoning_effort
                None, // voice
                None, // extra_body
                None, // parallel_tool_calls
                None, // normalize_response
                None, // embedding_encoding_format
                None, // embedding_dimensions
            ),
        }
    }

    /// Returns the API key
    pub fn api_key(&self) -> &str {
        &self.provider.api_key
    }

    /// Returns the model name
    pub fn model(&self) -> &str {
        &self.provider.model
    }
}

#[async_trait]
impl ChatProvider for ZAI {
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
impl CompletionProvider for ZAI {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        if self.api_key().is_empty() {
            return Err(LLMError::AuthError("Missing z.ai API key".into()));
        }
        Err(LLMError::ProviderError(
            "z.ai completion not implemented yet".into(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for ZAI {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for ZAI {}

impl LLMProvider for ZAI {}

impl LLMBuilder<ZAI> {
    pub fn build(self) -> Result<Arc<ZAI>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for z.ai".to_string())
        })?;

        let zai = ZAI::new(
            api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
        );

        Ok(Arc::new(zai))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::completion::CompletionRequest;

    #[test]
    fn test_new_defaults() {
        let client = ZAI::new("key", None, None, None, None);
        assert_eq!(client.api_key(), "key");
        assert_eq!(client.model(), "glm-5");
    }

    #[test]
    fn test_new_with_options_overrides() {
        let client = ZAI::new_with_options(
            "key",
            Some("https://api.z.ai/api/paas/v4/".to_string()),
            Some("glm-4-plus".to_string()),
            Some(111),
            Some(0.3),
            Some(9),
            Some(0.8),
            None,
        );
        assert_eq!(client.provider.model, "glm-4-plus");
        assert_eq!(
            client.provider.base_url.as_str(),
            "https://api.z.ai/api/paas/v4/"
        );
        assert_eq!(client.provider.max_tokens, Some(111));
        assert_eq!(client.provider.temperature, Some(0.3));
        assert_eq!(client.provider.top_p, Some(0.8));
    }

    #[tokio::test]
    async fn test_complete_missing_key() {
        let client = ZAI::new("", None, None, None, None);
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
        assert!(err.to_string().contains("Missing z.ai API key"));
    }

    #[tokio::test]
    async fn test_embed_not_supported() {
        let client = ZAI::new("key", None, None, None, None);
        let err = client.embed(vec!["hello".to_string()]).await.unwrap_err();
        assert!(err.to_string().contains("Embedding not supported"));
    }

    #[test]
    fn test_builder_requires_api_key() {
        let result = LLMBuilder::<ZAI>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided"));
    }

    #[tokio::test]
    #[ignore] // Requires ZAI_API_KEY environment variable
    async fn smoke_test_zai() {
        let api_key = std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY must be set");

        // Make a raw request first to see what the API actually returns
        let client = reqwest::Client::new();
        let raw_body = serde_json::json!({
            "model": "glm-5",
            "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": false
        });
        let raw_resp = client
            .post("https://api.z.ai/api/coding/paas/v4/chat/completions")
            .bearer_auth(&api_key)
            .json(&raw_body)
            .send()
            .await
            .expect("Raw request failed");
        println!("HTTP status: {}", raw_resp.status());
        let raw_text = raw_resp.text().await.expect("Failed to read raw response");
        println!("Raw API response: {raw_text}");

        let llm = LLMBuilder::<ZAI>::new()
            .api_key(api_key)
            .max_tokens(64)
            .temperature(0.0)
            .build()
            .expect("Failed to build ZAI provider");

        let message = ChatMessage::user()
            .content("What is 2 + 2? Answer with just the number.")
            .build();

        let response = llm.chat(&[message], None).await.expect("Chat request failed");
        let text = response.text().expect("No text in response");
        println!("z.ai response: {text}");
        assert!(text.contains('4'), "Expected response to contain '4', got: {text}");
    }
}
