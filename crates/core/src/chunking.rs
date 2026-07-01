//! Text chunking for embedding long content.
//!
//! Splits content into overlapping chunks that fit within an embedding model's
//! context window.  Each chunk is embedded independently; all chunk vectors are
//! stored together so that search can compare against every chunk.

/// Target maximum characters per chunk (~800 chars ≈ ~200 tokens for English text,
/// ~600-700 tokens for dense content like ASCII tables/diagrams). Sized to stay
/// under the 512-token context of `mxbai-embed-large` even when tokenization is
/// near 1:1 with chars.
pub const CHUNK_MAX_CHARS: usize = 800;

/// Overlap between adjacent chunks to preserve context across boundaries.
pub const CHUNK_OVERLAP: usize = 200;

/// Split `content` into overlapping chunks for embedding.
///
/// - Content shorter than `max_chars` is returned as a single chunk.
/// - Splits on paragraph boundaries (`\n\n`), falling back to sentence (`. `),
///   then word boundaries.
/// - Each chunk targets `max_chars` with `overlap` characters of overlap.
pub fn chunk_text(content: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return vec![String::new()];
    }
    if trimmed.len() <= max_chars {
        return vec![trimmed.to_string()];
    }

    // Split into paragraphs first.
    let paragraphs: Vec<&str> = trimmed.split("\n\n").collect();

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in &paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        // If adding this paragraph would exceed max_chars, flush current chunk.
        if !current.is_empty() && current.len() + 2 + para.len() > max_chars {
            chunks.push(current.clone());
            // Start next chunk with overlap from the end of the current chunk.
            current = overlap_tail(&current, overlap);
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);

        // If a single paragraph exceeds max_chars, split it further.
        while current.len() > max_chars {
            let split_pos = find_split_point(&current, max_chars);
            let head = current[..split_pos].trim().to_string();
            let tail = current[split_pos..].trim().to_string();
            chunks.push(head.clone());
            current = overlap_tail(&head, overlap);
            if !tail.is_empty() {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&tail);
            }
        }
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    // Filter out any empty chunks that may have been created.
    chunks.retain(|c| !c.is_empty());
    if chunks.is_empty() {
        chunks.push(trimmed.to_string());
    }

    chunks
}

/// Find a good split point at or before `max_pos` in `text`.
/// Prefers sentence boundaries (`. `), then word boundaries (` `).
fn find_split_point(text: &str, max_pos: usize) -> usize {
    // Snap to a char boundary BEFORE slicing: `max_pos` (the target chunk size
    // in bytes) can land inside a multi-byte char, and `&text[..search_end]`
    // below would panic on such an index. Multibyte content with no `\n\n`/`. `/
    // ` ` boundary before the cap reaches exactly this path.
    let search_end = text.floor_char_boundary(max_pos.min(text.len()));
    let region = &text[..search_end];

    // Try sentence boundary: look for ". " from the end.
    if let Some(pos) = region.rfind(". ") {
        let split = pos + 1; // include the period
        if split > max_pos / 2 {
            return split;
        }
    }

    // Try word boundary.
    if let Some(pos) = region.rfind(' ')
        && pos > max_pos / 2
    {
        return pos;
    }

    // Hard split at max_pos (or nearest char boundary).
    text.floor_char_boundary(search_end)
}

/// Return the last `overlap` characters of `text` (on a word boundary).
fn overlap_tail(text: &str, overlap: usize) -> String {
    if text.len() <= overlap {
        return text.to_string();
    }
    let start = text.len() - overlap;
    // Advance to a char boundary.
    let start = text.ceil_char_boundary(start);
    let tail = &text[start..];
    // Try to start on a word boundary.
    if let Some(space_pos) = tail.find(' ') {
        tail[space_pos + 1..].to_string()
    } else {
        tail.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_returns_single_chunk() {
        let chunks = chunk_text("Hello world", CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn empty_text_returns_single_empty_chunk() {
        let chunks = chunk_text("", CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn long_text_gets_chunked() {
        let text = "word ".repeat(500); // ~2500 chars
        let chunks = chunk_text(&text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        for chunk in &chunks {
            // Each chunk should be at most max_chars + some tolerance for overlap tail
            assert!(
                chunk.len() <= CHUNK_MAX_CHARS + CHUNK_OVERLAP + 50,
                "chunk too long: {} chars",
                chunk.len()
            );
        }
    }

    #[test]
    fn paragraphs_split_at_boundaries() {
        let para1 = "a".repeat(800);
        let para2 = "b".repeat(800);
        let text = format!("{para1}\n\n{para2}");
        let chunks = chunk_text(&text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        assert!(
            chunks.len() >= 2,
            "expected >= 2 chunks, got {}",
            chunks.len()
        );
        assert!(chunks[0].contains(&"a".repeat(100)));
        assert!(chunks.last().unwrap().contains(&"b".repeat(100)));
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let chunks = chunk_text("   \n\n   ", CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn exactly_max_chars_returns_single_chunk() {
        let text = "x".repeat(CHUNK_MAX_CHARS);
        let chunks = chunk_text(&text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        assert_eq!(chunks.len(), 1);
    }

    /// A long run of multibyte characters with NO spaces or paragraph breaks
    /// forces the hard-split path (`find_split_point` falls through to
    /// `floor_char_boundary`) and the overlap tail (`ceil_char_boundary`). A
    /// naive byte slice at `max_chars` would land mid-character and panic; the
    /// char-boundary helpers must keep every chunk valid UTF-8.
    #[test]
    fn chunking_multibyte_safe() {
        // 250 distinct CJK code points (3 bytes each = 750 bytes), no spaces.
        let text: String = (0x4E00u32..0x4E00 + 250)
            .filter_map(char::from_u32)
            .collect();
        assert!(text.len() > 300, "input must exceed max_chars to split");

        let chunks = chunk_text(&text, 300, 60);
        assert!(
            chunks.len() > 1,
            "long text must split into multiple chunks"
        );
        for chunk in &chunks {
            assert!(!chunk.is_empty());
            // Round-trips as valid UTF-8 — a mid-char cut would not.
            assert!(std::str::from_utf8(chunk.as_bytes()).is_ok());
        }
    }

    /// Adjacent chunks must actually SHARE overlap text so context isn't lost
    /// across a boundary. Uses distinct (non-repeating) multibyte characters so
    /// a shared slice unambiguously proves overlap rather than coincidental
    /// repetition.
    #[test]
    fn chunking_overlap_shares_text() {
        let text: String = (0x4E00u32..0x4E00 + 250)
            .filter_map(char::from_u32)
            .collect();
        let chunks = chunk_text(&text, 300, 60);
        assert!(chunks.len() >= 2);

        // The last 10 chars of chunk 0 fall inside chunk 1's leading overlap
        // region (a 60-byte / 20-char tail), so they must reappear in chunk 1.
        let tail: String = {
            let n = chunks[0].chars().count();
            chunks[0].chars().skip(n - 10).collect()
        };
        assert!(
            chunks[1].contains(&tail),
            "chunk 1 must carry chunk 0's tail as overlap; tail={tail:?}, chunk1 start={:?}",
            chunks[1].chars().take(30).collect::<String>()
        );
    }
}
