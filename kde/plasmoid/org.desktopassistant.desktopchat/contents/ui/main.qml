import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.plasma5support as Plasma5Support

PlasmoidItem {
    id: root
    implicitWidth: 460
    implicitHeight: 560

    property string helperPath: Qt.resolvedUrl("../code/dbus_client.py").toString().replace("file://", "")
    property string conversationId: ""
    property bool busy: false
    property bool debugEnabled: false
    property int currentMessageCount: 0
    property var transcriptEntries: []
    property string promptText: ""
    property var conversationChoices: []
    readonly property bool hasRealMessages: {
        for (let i = 0; i < transcriptEntries.length; i++) {
            if (transcriptEntries[i].kind === "message") return true
        }
        return false
    }
    readonly property var pending: ({})

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

    function ensureConversation(onReady) {
        if (conversationId.length > 0) {
            if (onReady) {
                onReady()
            }
            return
        }

        const command = "python3 " + shellEscape(helperPath) + " ensure --title 'Desktop Chat'"
        appendDebugStatus("Initializing conversation…")
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Conversation init failed: " + payload.error)
                    return
                }
                conversationId = payload.conversation_id
                appendStatus("Using conversation " + conversationId)
                reloadConversationList()
                refreshConversation()
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

        const title = "Desktop Chat " + Date.now()
        const command = "python3 " + shellEscape(helperPath) + " create --title " + shellEscape(title)
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Failed to create new conversation: " + payload.error)
                    return
                }

                conversationId = payload.conversation_id
                promptText = ""
                currentMessageCount = 0
                transcriptEntries = [
                    {
                        kind: "status",
                        role: "status",
                        text: "New conversation ready",
                    }
                ]
                appendStatus("Using conversation " + conversationId)
                reloadConversationList()
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function conversationIndexById(id) {
        for (let i = 0; i < conversationChoices.length; i++) {
            if (conversationChoices[i].id === id) {
                return i
            }
        }
        return -1
    }

    function reloadConversationList() {
        const command = "python3 " + shellEscape(helperPath) + " list"
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus(payload.error)
                    return
                }

                const conversations = payload.conversations || []
                conversationChoices = conversations.map(function(conversation) {
                    return {
                        id: conversation.id,
                        title: conversation.title + " (" + conversation.message_count + ")",
                    }
                })

                const idx = conversationIndexById(conversationId)
                if (idx >= 0) {
                    conversationPicker.currentIndex = idx
                }
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function switchConversation(index) {
        if (index < 0 || index >= conversationChoices.length) {
            return
        }

        const selectedId = conversationChoices[index].id
        if (selectedId === conversationId) {
            return
        }

        conversationId = selectedId
        promptText = ""
        appendStatus("Switched to conversation " + conversationId)
        refreshConversation()
    }

    function refreshConversation() {
        if (conversationId.length === 0 || busy) {
            return
        }

        const command = "python3 " + shellEscape(helperPath) + " get " + shellEscape(conversationId)
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus(payload.error)
                    return
                }

                transcriptEntries = []
                for (let i = 0; i < payload.messages.length; i++) {
                    const message = payload.messages[i]
                    appendMessage(message.role, message.content)
                }
                currentMessageCount = payload.messages.length
                if (currentMessageCount === 0) {
                    appendStatus("No messages yet")
                }
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
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

    function sendPrompt(textValue) {
        const sourceText = (textValue === undefined || textValue === null) ? promptText : textValue
        const prompt = sourceText.trim()
        if (prompt.length === 0 || busy) {
            return
        }

        ensureConversation(function() {
            busy = true
            appendMessage("user", prompt)
            promptText = ""

            const sendCommand = "python3 " + shellEscape(helperPath) + " send " + shellEscape(conversationId) + " " + shellEscape(prompt)
            runCommand(
                sendCommand,
                function(sendOut) {
                    const payload = JSON.parse(sendOut)
                    if (payload.error) {
                        busy = false
                        appendStatus(payload.error)
                        return
                    }
                    appendStatus("Waiting for assistant response…")
                    const awaitCommand = "python3 " + shellEscape(helperPath) + " await " + shellEscape(conversationId) + " --initial-count " + currentMessageCount
                    runCommand(
                        awaitCommand,
                        function(awaitOut) {
                            busy = false
                            const awaitPayload = JSON.parse(awaitOut)
                            if (awaitPayload.error) {
                                appendStatus(awaitPayload.error)
                                return
                            }
                            if (awaitPayload.assistant_reply && awaitPayload.assistant_reply.length > 0) {
                                appendMessage("assistant", awaitPayload.assistant_reply)
                                currentMessageCount = currentMessageCount + 2
                            } else {
                                appendStatus("No assistant response before timeout")
                                currentMessageCount = currentMessageCount + 1
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
        })
    }

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

    Timer {
        id: startupTimer
        interval: 250
        repeat: false
        running: false
        onTriggered: ensureConversation()
    }

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

            QQC2.CheckBox {
                text: "Debug"
                checked: root.debugEnabled
                onToggled: root.debugEnabled = checked
            }

            PlasmaComponents.ComboBox {
                id: conversationPicker
                Layout.fillWidth: true
                model: root.conversationChoices
                textRole: "title"
                onActivated: function(index) {
                    switchConversation(index)
                }
            }

            QQC2.Button {
                text: "Load"
                enabled: !busy
                onClicked: reloadConversationList()
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
                        sourceSize.width: 64
                        sourceSize.height: 64
                        fillMode: Image.PreserveAspectFit
                        anchors.horizontalCenter: parent.horizontalCenter
                    }

                    QQC2.Label {
                        text: "Ask me anything..."
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

                    width: ListView.view.width
                    implicitHeight: bubble.height + 2

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
                placeholderText: "Ask something"
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
                text: "Refresh"
                enabled: !busy
                onClicked: refreshConversation()
            }

            QQC2.Button {
                text: "Clear"
                enabled: !busy
                onClicked: clearTranscriptView()
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
        appendDebugStatus("Widget loaded")
        startupTimer.start()
    }
}
