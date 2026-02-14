import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.plasma5support as Plasma5Support

PlasmoidItem {
    id: root

    preferredRepresentation: Plasmoid.compactRepresentation
    switchWidth: 320
    switchHeight: 420

    property string helperPath: Qt.resolvedUrl("../code/dbus_client.py").toString().replace("file://", "")
    property string conversationId: ""
    property bool busy: false
    property bool debugEnabled: false
    property string transcriptText: "[status] Ready"
    property string promptText: ""

    function shellEscape(value) {
        return "'" + value.replace(/'/g, "'\\''") + "'"
    }

    function runCommand(command, onSuccess, onError) {
        appendDebugStatus("run: " + command)
        pending[command] = {
            success: onSuccess,
            error: onError,
        }
        executable.connectSource(command)
    }

    function ensureConversation(onReady) {
        if (conversationId.length > 0) {
            if (onReady) {
                onReady()
            }
            return
        }
        let command = "python3 " + shellEscape(helperPath) + " ensure --title 'Panel Chat'"
        runCommand(
            command,
            function(stdout) {
                let payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Failed to create conversation: " + payload.error)
                    return
                }
                conversationId = payload.conversation_id
                if (onReady) {
                    onReady()
                }
            },
            function(stderr) {
                appendStatus("Conversation error: " + stderr)
            }
        )
    }

    function newConversation() {
        if (busy) {
            return
        }

        let title = "Panel Chat " + Date.now()
        let command = "python3 " + shellEscape(helperPath) + " create --title " + shellEscape(title)
        runCommand(
            command,
            function(stdout) {
                let payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Failed to create conversation: " + payload.error)
                    return
                }

                conversationId = payload.conversation_id
                currentMessageCount = 0
                transcriptText = "[status] New conversation ready\n[status] Connected to conversation " + conversationId
                promptText = ""
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function appendStatus(text) {
        transcriptText += "\n[status] " + text
    }

    function appendDebugStatus(text) {
        if (debugEnabled) {
            appendStatus(text)
        }
    }

    function appendMessage(role, text) {
        if (role === "tool" && !debugEnabled) {
            return
        }
        transcriptText += "\n" + role + ": " + text
    }

    function messageCountFromTranscript() {
        if (conversationId.length === 0) {
            return 0
        }
        let command = "python3 " + shellEscape(helperPath) + " get " + shellEscape(conversationId)
        runCommand(
            command,
            function(stdout) {
                let payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus(payload.error)
                    return
                }
                currentMessageCount = payload.messages.length
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    property int currentMessageCount: 0

    function sendPrompt(textValue) {
        let sourceText = (textValue === undefined || textValue === null) ? promptText : textValue
        let prompt = sourceText.trim()
        if (prompt.length === 0 || busy) {
            return
        }

        ensureConversation(function() {
            busy = true
            appendMessage("you", prompt)
            promptText = ""

            let getCommand = "python3 " + shellEscape(helperPath) + " get " + shellEscape(conversationId)
            runCommand(
                getCommand,
                function(stdout) {
                    let payload = JSON.parse(stdout)
                    if (payload.error) {
                        busy = false
                        appendStatus(payload.error)
                        return
                    }
                    currentMessageCount = payload.messages.length

                    let sendCommand = "python3 " + shellEscape(helperPath) + " send " + shellEscape(conversationId) + " " + shellEscape(prompt)
                    runCommand(
                        sendCommand,
                        function(sendOut) {
                            let sendPayload = JSON.parse(sendOut)
                            if (sendPayload.error) {
                                busy = false
                                appendStatus(sendPayload.error)
                                return
                            }
                            appendStatus("Waiting for assistant response…")
                            let awaitCommand = "python3 " + shellEscape(helperPath) + " await " + shellEscape(conversationId) + " --initial-count " + currentMessageCount
                            runCommand(
                                awaitCommand,
                                function(awaitOut) {
                                    busy = false
                                    let awaitPayload = JSON.parse(awaitOut)
                                    if (awaitPayload.error) {
                                        appendStatus(awaitPayload.error)
                                        return
                                    }
                                    if (awaitPayload.assistant_reply && awaitPayload.assistant_reply.length > 0) {
                                        appendMessage("assistant", awaitPayload.assistant_reply)
                                    } else {
                                        appendStatus("No assistant message received before timeout")
                                    }
                                },
                                function(awaitErr) {
                                    busy = false
                                    appendStatus(awaitErr)
                                }
                            )
                        },
                        function(sendErr) {
                            busy = false
                            appendStatus(sendErr)
                        }
                    )
                },
                function(getErr) {
                    busy = false
                    appendStatus(getErr)
                }
            )
        })
    }

    readonly property var pending: ({})

    Plasma5Support.DataSource {
        id: executable
        engine: "executable"
        connectedSources: []

        onNewData: function(sourceName, data) {
            const callbacks = pending[sourceName]
            if (!callbacks) {
                disconnectSource(sourceName)
                return
            }

            const exitCode = data["exit code"]
            const stdout = (data.stdout || "").trim()
            const stderr = (data.stderr || "").trim()

            if (exitCode === 0) {
                appendDebugStatus("ok: " + sourceName)
                callbacks.success(stdout)
            } else {
                appendDebugStatus("error: " + sourceName)
                callbacks.error(stderr.length > 0 ? stderr : stdout)
            }

            delete pending[sourceName]
            disconnectSource(sourceName)
        }
    }

    compactRepresentation: PlasmaComponents.ToolButton {
        text: "Assistant"
        icon.name: "preferences-system-network"
        onClicked: root.expanded = !root.expanded
    }

    fullRepresentation: Item {
        Layout.minimumWidth: 320
        Layout.minimumHeight: 420

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 8
            spacing: 8

            QQC2.Label {
                text: "Desktop Assistant"
                font.bold: true
                Layout.fillWidth: true
            }

            RowLayout {
                Layout.fillWidth: true

                Item {
                    Layout.fillWidth: true
                }

                QQC2.CheckBox {
                    text: "Debug"
                    checked: root.debugEnabled
                    onToggled: root.debugEnabled = checked
                }
            }

            QQC2.ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true

                QQC2.TextArea {
                    id: transcript
                    readOnly: true
                    wrapMode: Text.Wrap
                    text: root.transcriptText
                    onTextChanged: cursorPosition = length
                }
            }

            RowLayout {
                Layout.fillWidth: true

                QQC2.TextField {
                    Layout.fillWidth: true
                    placeholderText: "Ask desktop assistant…"
                    text: root.promptText
                    enabled: !busy
                    onTextChanged: root.promptText = text
                    onAccepted: sendPrompt(text)
                }

                QQC2.Button {
                    text: busy ? "…" : "Send"
                    enabled: !busy
                    onClicked: sendPrompt(root.promptText)
                }

                QQC2.Button {
                    text: "New"
                    enabled: !busy
                    onClicked: newConversation()
                }
            }
        }
    }

    Component.onCompleted: {
        ensureConversation(function() {
            appendDebugStatus("Connected to conversation " + conversationId)
        })
    }
}
