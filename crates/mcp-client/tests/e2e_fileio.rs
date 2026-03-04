//! End-to-end test with a real MCP server (fileio-mcp).
//! Requires `fileio-mcp` to be installed. Skips if not available.

use desktop_assistant_mcp_client::McpClient;

fn fileio_available() -> bool {
    std::process::Command::new("fileio-mcp")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn connect_and_list_tools() {
    if !fileio_available() {
        eprintln!("SKIP: fileio-mcp not found");
        return;
    }

    let mut client = McpClient::connect(
        "fileio-mcp",
        &["serve".into(), "--mode".into(), "stdio".into()],
    )
    .await
    .expect("failed to connect to fileio-mcp");

    let tools = client.list_tools().await.expect("failed to list tools");
    assert!(!tools.is_empty(), "fileio-mcp should provide tools");

    // Check for a known tool
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(
        tool_names.contains(&"fileio_read_lines"),
        "expected fileio_read_lines tool, got: {tool_names:?}"
    );

    println!("fileio-mcp provides {} tools:", tools.len());
    for tool in &tools {
        println!(
            "  - {} ({})",
            tool.name,
            tool.description.chars().take(60).collect::<String>()
        );
    }

    client.shutdown().await;
}

#[tokio::test]
async fn call_read_lines_tool() {
    if !fileio_available() {
        eprintln!("SKIP: fileio-mcp not found");
        return;
    }

    // Create a temp file to read
    let tmp_dir = std::env::temp_dir().join("desktop-assistant-e2e-test");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let test_file = tmp_dir.join("test.txt");
    std::fs::write(&test_file, "hello from e2e test\nsecond line\n").unwrap();

    let mut client = McpClient::connect(
        "fileio-mcp",
        &["serve".into(), "--mode".into(), "stdio".into()],
    )
    .await
    .expect("failed to connect to fileio-mcp");

    let result = client
        .call_tool(
            "fileio_read_lines",
            serde_json::json!({
                "path": test_file.to_str().unwrap()
            }),
        )
        .await
        .expect("failed to call fileio_read_lines");

    println!("Tool result: {result}");
    assert!(
        result.contains("hello from e2e test"),
        "expected file contents in result, got: {result}"
    );

    // Cleanup
    client.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[tokio::test]
async fn executor_with_real_mcp_server() {
    use desktop_assistant_core::ports::tools::ToolExecutor;
    use desktop_assistant_mcp_client::executor::{McpServerConfig, McpToolExecutor};

    if !fileio_available() {
        eprintln!("SKIP: fileio-mcp not found");
        return;
    }

    let configs = vec![McpServerConfig {
        name: "fileio".into(),
        command: "fileio-mcp".into(),
        args: vec!["serve".into(), "--mode".into(), "stdio".into()],
        namespace: None,
        enabled: true,
    }];

    let executor = McpToolExecutor::new(configs);
    executor.start().await.expect("failed to start executor");

    // Verify tools are available
    let tools = executor.core_tools().await;
    assert!(
        !tools.is_empty(),
        "executor should have tools from fileio-mcp"
    );

    // Verify additional metadata endpoints are callable
    let resources = executor.available_resources().await;
    let prompts = executor.available_prompts().await;
    for resource in &resources {
        assert!(resource.is_object(), "resource entry should be an object");
    }
    for prompt in &prompts {
        assert!(prompt.is_object(), "prompt entry should be an object");
    }

    // Create test file
    let tmp_dir = std::env::temp_dir().join("desktop-assistant-e2e-executor");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let test_file = tmp_dir.join("executor_test.txt");
    std::fs::write(&test_file, "executor e2e test content\n").unwrap();

    // Execute a tool through the executor
    let result = executor
        .execute_tool(
            "fileio_read_lines",
            serde_json::json!({
                "path": test_file.to_str().unwrap()
            }),
        )
        .await
        .expect("execute_tool should succeed");

    println!("Executor result: {result}");
    assert!(result.contains("executor e2e test content"));

    // Unknown tool should fail
    let err = executor
        .execute_tool("nonexistent_tool", serde_json::json!({}))
        .await;
    assert!(err.is_err());

    // Cleanup
    executor.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
