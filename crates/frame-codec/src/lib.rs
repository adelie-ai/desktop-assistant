//! Length-prefixed frame codec shared by every local transport.
//!
//! The wire format is a 4-byte little-endian `u32` length followed by that
//! many body bytes. A 0-length frame carries an empty body (used as a clean
//! close marker by some peers). The header is read with `read_exact`, the
//! body is then allocated and read with `read_exact`.
//!
//! The [`MAX_FRAME_LEN`] cap keeps a buggy or hostile peer from claiming a
//! multi-GB length and forcing an allocation blow-up.
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

/// Write one length-prefixed frame and flush.
pub async fn write_frame<W>(writer: &mut W, body: &[u8]) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
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

    #[tokio::test]
    async fn truncated_header_is_eof() {
        let mut cursor = std::io::Cursor::new(vec![0u8, 0u8]);
        let err = read_frame(&mut cursor).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
