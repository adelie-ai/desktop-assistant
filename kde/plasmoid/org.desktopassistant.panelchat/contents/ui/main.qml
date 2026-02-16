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
    property var transcriptEntries: []
    property string promptText: ""
    readonly property bool hasRealMessages: {
        for (let i = 0; i < transcriptEntries.length; i++) {
            if (transcriptEntries[i].kind === "message") return true
        }
        return false
    }

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
                transcriptEntries = [
                    {
                        kind: "status",
                        role: "status",
                        text: "New conversation ready",
                    },
                    {
                        kind: "status",
                        role: "status",
                        text: "Connected to conversation " + conversationId,
                    }
                ]
                promptText = ""
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function appendStatus(text) {
        transcriptEntries = transcriptEntries.concat([
            {
                kind: "status",
                role: "status",
                text: text,
            }
        ])
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
        transcriptEntries = transcriptEntries.concat([
            {
                kind: "message",
                role: role,
                text: text,
            }
        ])
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
            appendMessage("user", prompt)
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

    function clearTranscriptView() {
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "View cleared",
            }
        ]
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
        text: "Adele"
        icon.source: Qt.resolvedUrl("../images/adele.png")
        icon.width: PlasmaCore.Units.iconSizes.smallMedium
        icon.height: PlasmaCore.Units.iconSizes.smallMedium
        onClicked: root.expanded = !root.expanded
    }

    fullRepresentation: Item {
        Layout.minimumWidth: 320
        Layout.minimumHeight: 420

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 8
            spacing: 8

            RowLayout {
                Layout.fillWidth: true
                spacing: 6

                Image {
                    source: root.busy
                        ? Qt.resolvedUrl("../images/adele_thinking.png")
                        : Qt.resolvedUrl("../images/adele.png")
                    sourceSize.width: 24
                    sourceSize.height: 24
                    fillMode: Image.PreserveAspectFit
                    Layout.preferredWidth: 24
                    Layout.preferredHeight: 24
                }

                QQC2.Label {
                    text: "Adele"
                    font.bold: true
                    Layout.fillWidth: true
                }
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

                ListView {
                    id: transcript
                    model: root.transcriptEntries
                    spacing: 6
                    clip: true

                    header: Column {
                        width: transcript.width
                        visible: !root.hasRealMessages
                        spacing: 8
                        topPadding: 40

                        Image {
                            source: Qt.resolvedUrl("../images/adele.png")
                            sourceSize.width: 96
                            sourceSize.height: 96
                            fillMode: Image.PreserveAspectFit
                            anchors.horizontalCenter: parent.horizontalCenter
                        }

                        QQC2.Label {
                            text: "Hi! I'm Adele! Ask me anything..."
                            font.pointSize: 12
                            color: PlasmaCore.Theme.disabledTextColor
                            anchors.horizontalCenter: parent.horizontalCenter
                        }
                    }

                    onCountChanged: {
                        if (count > 0) {
                            positionViewAtEnd()
                        }
                    }

                    delegate: Item {
                        required property var modelData
                        readonly property bool isStatus: modelData.kind === "status"
                        readonly property bool isAssistant: modelData.role === "assistant"

                        visible: !isStatus || root.debugEnabled
                        width: ListView.view.width
                        implicitHeight: visible ? bubble.height + 2 : 0

                        Rectangle {
                            id: bubble
                            anchors.top: parent.top
                            anchors.left: isAssistant ? undefined : parent.left
                            anchors.right: isAssistant ? parent.right : undefined
                            width: isStatus
                                ? parent.width
                                : Math.min(parent.width * 0.88, Math.max(120, messageText.implicitWidth + 12))
                            height: Math.max(messageText.contentHeight, messageText.implicitHeight) + 12
                            radius: isStatus ? 0 : 8
                            color: isStatus
                                ? "transparent"
                                : (isAssistant ? PlasmaCore.Theme.backgroundColor : PlasmaCore.Theme.highlightColor)
                            border.width: isStatus ? 0 : 1
                            border.color: PlasmaCore.Theme.disabledTextColor

                            TextEdit {
                                id: messageText
                                anchors.fill: parent
                                anchors.margins: 6
                                readOnly: true
                                selectByMouse: true
                                selectByKeyboard: true
                                wrapMode: TextEdit.Wrap
                                textFormat: (modelData.kind === "message" && isAssistant) ? Text.MarkdownText : Text.PlainText
                                text: isStatus
                                    ? "[status] " + String(modelData.text || "")
                                    : (isAssistant
                                        ? String(modelData.text || "")
                                        : "You:\n" + String(modelData.text || ""))
                                color: isStatus
                                    ? PlasmaCore.Theme.disabledTextColor
                                    : (isAssistant
                                        ? PlasmaCore.Theme.textColor
                                        : PlasmaCore.Theme.highlightedTextColor)
                                font.italic: isStatus
                                font.bold: false
                                activeFocusOnPress: true
                                selectedTextColor: isAssistant ? PlasmaCore.Theme.highlightedTextColor : PlasmaCore.Theme.textColor
                                selectionColor: isAssistant ? PlasmaCore.Theme.highlightColor : PlasmaCore.Theme.backgroundColor
                                onLinkActivated: function(link) {
                                    Qt.openUrlExternally(link)
                                }
                            }
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true

                QQC2.TextField {
                    Layout.fillWidth: true
                    placeholderText: "Ask Adele…"
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
            }

            RowLayout {
                Layout.fillWidth: true

                QQC2.Button {
                    text: "New"
                    enabled: !busy
                    onClicked: newConversation()
                }

                QQC2.Button {
                    text: "Clear"
                    enabled: !busy
                    onClicked: clearTranscriptView()
                }
            }
        }
    }

    Component.onCompleted: {
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "Ready",
            }
        ]
        ensureConversation(function() {
            appendDebugStatus("Connected to conversation " + conversationId)
        })
    }
}
