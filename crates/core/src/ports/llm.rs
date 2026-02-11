use crate::CoreError;
use crate::domain::Message;

/// Callback invoked for each chunk of a streaming LLM response.
/// Return `true` to continue, `false` to abort the stream.
pub type ChunkCallback = Box<dyn FnMut(String) -> bool + Send>;

/// Outbound port for LLM completion requests.
pub trait LlmClient: Send + Sync {
    /// Stream a completion from the LLM given a message history.
    /// Calls `on_chunk` for each token/chunk received.
    /// Returns the fully assembled response text.
    fn stream_completion(
        &self,
        messages: Vec<Message>,
        on_chunk: ChunkCallback,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    struct MockLlm {
        chunks: Vec<String>,
    }

    impl LlmClient for MockLlm {
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

    #[tokio::test]
    async fn mock_llm_streams_chunks() {
        use std::sync::{Arc, Mutex};

        let llm = MockLlm {
            chunks: vec!["Hello".into(), " world".into()],
        };
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        let result = llm
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                Box::new(move |chunk| {
                    received_clone.lock().unwrap().push(chunk);
                    true
                }),
            )
            .await
            .unwrap();
        assert_eq!(result, "Hello world");
        assert_eq!(*received.lock().unwrap(), vec!["Hello", " world"]);
    }

    #[tokio::test]
    async fn mock_llm_abort_stops_stream() {
        use std::sync::{Arc, Mutex};

        let llm = MockLlm {
            chunks: vec!["a".into(), "b".into(), "c".into()],
        };
        let count = Arc::new(Mutex::new(0));
        let count_clone = Arc::clone(&count);
        let result = llm
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                Box::new(move |_chunk| {
                    let mut c = count_clone.lock().unwrap();
                    *c += 1;
                    *c < 2 // abort after second chunk
                }),
            )
            .await
            .unwrap();
        assert_eq!(result, "ab");
        assert_eq!(*count.lock().unwrap(), 2);
    }
}
