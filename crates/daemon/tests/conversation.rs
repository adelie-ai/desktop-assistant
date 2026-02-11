//! Integration tests for conversation lifecycle using mock LLM.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{ConversationId, Message, Role};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::service::ConversationHandler;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// --- In-memory store for integration tests ---

struct TestStore {
    data: Mutex<HashMap<String, desktop_assistant_core::domain::Conversation>>,
}

impl TestStore {
    fn new() -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
        }
    }
}

impl ConversationStore for TestStore {
    async fn create(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
        Ok(())
    }

    async fn get(
        &self,
        id: &ConversationId,
    ) -> Result<desktop_assistant_core::domain::Conversation, CoreError> {
        self.data
            .lock()
            .unwrap()
            .get(&id.0)
            .cloned()
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
    }

    async fn list(&self) -> Result<Vec<desktop_assistant_core::domain::Conversation>, CoreError> {
        Ok(self.data.lock().unwrap().values().cloned().collect())
    }

    async fn update(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        if data.contains_key(&conv.id.0) {
            data.insert(conv.id.0.clone(), conv);
            Ok(())
        } else {
            Err(CoreError::ConversationNotFound(conv.id.0.clone()))
        }
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.data
            .lock()
            .unwrap()
            .remove(&id.0)
            .map(|_| ())
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
    }
}

// --- Mock LLM for integration tests ---

struct TestLlm {
    chunks: Vec<String>,
}

impl TestLlm {
    fn new(chunks: Vec<&str>) -> Self {
        Self {
            chunks: chunks.into_iter().map(String::from).collect(),
        }
    }
}

impl LlmClient for TestLlm {
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        mut on_chunk: ChunkCallback,
    ) -> Result<String, CoreError> {
        let mut full = String::new();
        for chunk in &self.chunks {
            full.push_str(chunk);
            if !on_chunk(chunk.clone()) {
                return Ok(full);
            }
        }
        Ok(full)
    }
}

fn make_service(chunks: Vec<&str>) -> ConversationHandler<TestStore, TestLlm> {
    let counter = Arc::new(AtomicU64::new(0));
    ConversationHandler::new(
        TestStore::new(),
        TestLlm::new(chunks),
        Box::new(move || {
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            format!("conv-{n}")
        }),
    )
}

#[tokio::test]
async fn full_conversation_lifecycle() {
    let service = make_service(vec!["Hello", ", ", "world", "!"]);

    // 1. Create a conversation
    let conv = service.create_conversation("My Chat".into()).await.unwrap();
    assert_eq!(conv.id.as_str(), "conv-1");
    assert_eq!(conv.title, "My Chat");
    assert!(conv.messages.is_empty());

    // 2. List conversations — should have one
    let summaries = service.list_conversations().await.unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].id.as_str(), "conv-1");
    assert_eq!(summaries[0].message_count, 0);

    // 3. Send a prompt and collect streaming chunks
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let chunks_clone = Arc::clone(&chunks);
    let response = service
        .send_prompt(
            &conv.id,
            "Hi there!".into(),
            Box::new(move |chunk| {
                chunks_clone.lock().unwrap().push(chunk);
                true
            }),
        )
        .await
        .unwrap();
    assert_eq!(response, "Hello, world!");
    assert_eq!(*chunks.lock().unwrap(), vec!["Hello", ", ", "world", "!"]);

    // 4. Verify the conversation now has both messages
    let updated = service.get_conversation(&conv.id).await.unwrap();
    assert_eq!(updated.messages.len(), 2);
    assert_eq!(updated.messages[0].role, Role::User);
    assert_eq!(updated.messages[0].content, "Hi there!");
    assert_eq!(updated.messages[1].role, Role::Assistant);
    assert_eq!(updated.messages[1].content, "Hello, world!");

    // 5. List should show message_count = 2
    let summaries = service.list_conversations().await.unwrap();
    assert_eq!(summaries[0].message_count, 2);

    // 6. Delete the conversation
    service.delete_conversation(&conv.id).await.unwrap();

    // 7. Verify it's gone
    let result = service.get_conversation(&conv.id).await;
    assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));

    // 8. List should be empty
    let summaries = service.list_conversations().await.unwrap();
    assert!(summaries.is_empty());
}

#[tokio::test]
async fn multiple_conversations() {
    let service = make_service(vec!["response"]);

    let c1 = service.create_conversation("Chat 1".into()).await.unwrap();
    let c2 = service.create_conversation("Chat 2".into()).await.unwrap();

    assert_ne!(c1.id, c2.id);

    let summaries = service.list_conversations().await.unwrap();
    assert_eq!(summaries.len(), 2);

    // Send prompt only to c1
    service
        .send_prompt(&c1.id, "hello".into(), Box::new(|_| true))
        .await
        .unwrap();

    // c1 should have 2 messages, c2 should have 0
    let conv1 = service.get_conversation(&c1.id).await.unwrap();
    let conv2 = service.get_conversation(&c2.id).await.unwrap();
    assert_eq!(conv1.messages.len(), 2);
    assert_eq!(conv2.messages.len(), 0);
}

#[tokio::test]
async fn streaming_callback_abort() {
    let service = make_service(vec!["a", "b", "c", "d"]);
    let conv = service
        .create_conversation("Abort Test".into())
        .await
        .unwrap();

    let chunk_count = Arc::new(Mutex::new(0usize));
    let chunk_count_clone = Arc::clone(&chunk_count);
    let response = service
        .send_prompt(
            &conv.id,
            "test".into(),
            Box::new(move |_| {
                let mut count = chunk_count_clone.lock().unwrap();
                *count += 1;
                *count < 2 // abort after 2nd chunk
            }),
        )
        .await
        .unwrap();

    assert_eq!(response, "ab");
    assert_eq!(*chunk_count.lock().unwrap(), 2);
}
