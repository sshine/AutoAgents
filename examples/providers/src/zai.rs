use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::zai::Zai;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentHooks, agent};
use std::sync::Arc;

#[agent(
    name = "coding_agent",
    description = "You are a helpful coding assistant powered by Z.AI's GLM model.",
    tools = [],
)]
#[derive(Default, Clone, AgentHooks)]
pub struct CodingAgent {}

pub async fn run() -> Result<(), Error> {
    let api_key = std::env::var("ZAI_API_KEY").unwrap_or_else(|_| "".into());

    let llm: Arc<Zai> = LLMBuilder::<Zai>::new()
        .api_key(api_key)
        .model("glm-5")
        .max_tokens(512)
        .temperature(0.7)
        .build()
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = BasicAgent::new(CodingAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new(
            "Write a Rust function that checks if a number is prime",
        ))
        .await?;
    println!("Result: {:?}", result);
    Ok(())
}
