//! Length-prefixed frame codec shared by every local transport.
//!
//! The wire format is a 4-byte little-endian `u32` length followed by that
//! many body bytes. A 0-length frame carries an empty body (used as a clean
//! close marker by some peers). The header is read with `read_exact`, the
//! body is then allocated and read with `read_exact`.
//!
//! The [`MAX_FRAME_LEN`] cap applies to both directions. On read it keeps a
//! buggy or hostile peer from claiming a multi-GB length and forcing an
//! allocation blow-up. On write it stops this side putting a frame on the wire
//! that the peer's reader is bound to reject - which used to cost the whole
//! connection, because a peer that rejects a frame breaks its read loop and
//! fails every outstanding request, not only the one that was too large.
//!
//! This crate is intentionally tiny and dependency-light (only `tokio` for
//! the async read/write traits) so both the server transports
//! (`uds-interface`, `dbus-bridge`) and the client transports
//! (`client-common`) can share one definition. Before this crate, the codec
//! was copy-pasted verbatim into all three, so the frame cap and framing
//! rules could silently drift between client and server.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Maximum accepted frame body size (4 MB).
///
/// Reads that announce a larger length are rejected with
/// [`std::io::ErrorKind::InvalidData`] before any allocation happens.
pub const MAX_FRAME_LEN: u32 = 4 * 1024 * 1024;

/// Read one length-prefixed frame.
///
/// The header is a 4-byte little-endian `u32` length. We `read_exact` the
/// header, validate it against [`MAX_FRAME_LEN`], allocate the body, then
/// `read_exact` the body. A 0-length frame yields an empty `Vec`.
pub async fn read_frame<R>(reader: &mut R) -> std::io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds cap {MAX_FRAME_LEN}"),
        ));
    }
    let mut body = vec![0u8; len as usize];
    if len > 0 {
        reader.read_exact(&mut body).await?;
    }
    Ok(body)
}

/// `true` when a body is too large for one frame, so a caller can answer the
/// waiting request with a failure instead of putting an unreadable frame on
/// the wire.
///
/// [`write_frame`] applies the same check itself and refuses. This predicate
/// exists so a transport can tell the two failures apart *before* it writes:
/// an oversize body means one request cannot be answered, while a write error
/// means the connection is gone. Confusing the two is what made a single large
/// response tear down every outstanding request on the connection.
pub fn exceeds_frame_cap(body: &[u8]) -> bool {
    body.len() > MAX_FRAME_LEN as usize
}

/// Write one length-prefixed frame and flush.
///
/// Refuses a body larger than [`read_frame`] would accept, with
/// [`std::io::ErrorKind::InvalidData`] and **before writing any bytes**. The
/// reader has always rejected an oversize frame; the writer used to send one
/// happily, so the daemon could put a frame on the wire that no peer was able
/// to read. A partial write would be worse still: it would desynchronize the
/// stream and break every later request on that connection.
pub async fn write_frame<W>(writer: &mut W, body: &[u8]) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    if exceeds_frame_cap(body) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame body {} exceeds cap {MAX_FRAME_LEN}", body.len()),
        ));
    }
    let len = body.len() as u32;
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(body).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trips_a_frame() {
        let mut buf = Vec::new();
        write_frame(&mut buf, b"hello world").await.unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let body = read_frame(&mut cursor).await.unwrap();
        assert_eq!(body, b"hello world");
    }

    #[tokio::test]
    async fn round_trips_an_empty_frame() {
        let mut buf = Vec::new();
        write_frame(&mut buf, b"").await.unwrap();
        // 4-byte header, zero body.
        assert_eq!(buf, vec![0, 0, 0, 0]);

        let mut cursor = std::io::Cursor::new(buf);
        let body = read_frame(&mut cursor).await.unwrap();
        assert!(body.is_empty());
    }

    #[tokio::test]
    async fn round_trips_back_to_back_frames() {
        let mut buf = Vec::new();
        write_frame(&mut buf, b"one").await.unwrap();
        write_frame(&mut buf, b"two").await.unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        assert_eq!(read_frame(&mut cursor).await.unwrap(), b"one");
        assert_eq!(read_frame(&mut cursor).await.unwrap(), b"two");
    }

    #[tokio::test]
    async fn rejects_oversized_length_without_allocating() {
        // Header claims MAX_FRAME_LEN + 1, no body follows.
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_LEN + 1).to_le_bytes());

        let mut cursor = std::io::Cursor::new(buf);
        let err = read_frame(&mut cursor).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn accepts_exactly_max_len_header() {
        // A header of exactly MAX_FRAME_LEN must not be rejected by the cap
        // check; it fails on the truncated body instead (UnexpectedEof).
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAX_FRAME_LEN.to_le_bytes());

        let mut cursor = std::io::Cursor::new(buf);
        let err = read_frame(&mut cursor).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    /// Acceptance (#1303): `write_frame` refuses a body larger than
    /// `read_frame` would accept. The two used to disagree - the reader
    /// rejected an oversize frame and the writer wrote one happily - so the
    /// daemon could put a frame on the wire that no peer was able to read.
    #[tokio::test]
    async fn write_frame_refuses_a_body_larger_than_read_frame_would_accept() {
        let body = vec![b'x'; MAX_FRAME_LEN as usize + 1];
        let mut buf = Vec::new();
        let err = write_frame(&mut buf, &body)
            .await
            .expect_err("an oversize body must be refused, not written");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// #1303: the refusal writes NOTHING. A partial frame would desynchronize
    /// the stream and break every later request on the same connection, which
    /// is the failure the refusal exists to prevent.
    #[tokio::test]
    async fn a_refused_frame_puts_no_bytes_on_the_wire() {
        let body = vec![b'x'; MAX_FRAME_LEN as usize + 1];
        let mut buf = Vec::new();
        let _ = write_frame(&mut buf, &body).await;
        assert!(
            buf.is_empty(),
            "a refused frame must leave the stream untouched, got {} bytes",
            buf.len()
        );
    }

    /// #1303: the boundary is the same on both sides. A body of exactly
    /// `MAX_FRAME_LEN` is accepted by the reader, so the writer accepts it too.
    #[tokio::test]
    async fn write_frame_accepts_a_body_of_exactly_max_len() {
        let body = vec![b'x'; MAX_FRAME_LEN as usize];
        let mut buf = Vec::new();
        write_frame(&mut buf, &body)
            .await
            .expect("a body at the cap is within it, not over it");

        let mut cursor = std::io::Cursor::new(buf);
        assert_eq!(read_frame(&mut cursor).await.unwrap().len(), body.len());
    }

    #[tokio::test]
    async fn truncated_header_is_eof() {
        let mut cursor = std::io::Cursor::new(vec![0u8, 0u8]);
        let err = read_frame(&mut cursor).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
