import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import QtCore
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.plasma5support as Plasma5Support

Item {
    id: root
    property bool panelMode: false
    implicitWidth: 460
    implicitHeight: 560

    property string helperPath: Qt.resolvedUrl("../code/dbus_client.py").toString().replace("file://", "")
    property string productionService: "org.desktopAssistant"
    property string developmentService: "org.desktopAssistant.Dev"
    property string activeService: productionService
    property bool serviceInitialized: false
    property bool productionServiceRunning: false
    property bool devServiceRunning: false
    readonly property bool hideWidget: !devServiceRunning
    property var serviceChoices: []
    property string conversationId: ""
    property bool busy: false
    property bool debugEnabled: false
    property int currentMessageCount: 0
    property var transcriptEntries: []
    property string promptText: ""
    property var conversationChoices: []
    property real uiScale: 1.0
    readonly property string conversationTitle: panelMode ? "Panel Chat" : "Desktop Chat"
    readonly property real awaitTimeoutSeconds: panelMode ? 45.0 : 60.0
    readonly property real minUiScale: 0.9
    readonly property real maxUiScale: 1.35
    readonly property real zoomStep: 0.05
    readonly property string adeleAvatarSource: Qt.resolvedUrl("../images/adele.png")
    property string configuredUserAvatarPath: String(Plasmoid.configuration.userAvatarPath || "").trim()
    readonly property real baseFontPointSize: Math.max(1, Number(Kirigami.Theme.defaultFont.pointSize || Qt.application.font.pointSize || 10))
    readonly property int scaledTopIconSize: Math.max(16, Math.round(24 * uiScale))
    readonly property int scaledHeaderIconSize: Math.max(64, Math.round(96 * uiScale))
    readonly property string homeDirectory: StandardPaths.writableLocation(StandardPaths.HomeLocation)
    readonly property string accountName: {
        const trimmedHome = String(homeDirectory || "").replace(/\/+$/, "")
        const chunks = trimmedHome.split("/").filter(function(chunk) {
            return chunk.length > 0
        })
        return chunks.length > 0 ? chunks[chunks.length - 1] : ""
    }
    readonly property bool hasRealMessages: {
        for (let i = 0; i < transcriptEntries.length; i++) {
            if (transcriptEntries[i].kind === "message") return true
        }
        return false
    }
    readonly property var pending: ({})

    function toImageSource(pathValue) {
        const value = String(pathValue || "").trim()
        if (value.length === 0) {
            return ""
        }
        if (value.indexOf("file://") === 0 || value.indexOf("image://") === 0 || value.indexOf("qrc:/") === 0 || value.indexOf(":/") === 0) {
            return value
        }
        if (value[0] === "/") {
            return "file://" + value
        }
        return value
    }

    function userAvatarCandidates() {
        const candidates = []
        const configured = toImageSource(configuredUserAvatarPath)
        if (configured.length > 0) {
            candidates.push(configured)
        }
        if (accountName.length > 0) {
            candidates.push(toImageSource("/var/lib/AccountsService/icons/" + accountName))
        }
        candidates.push(toImageSource(homeDirectory + "/.face.icon"))
        candidates.push(toImageSource(homeDirectory + "/.face"))
        return candidates
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

    function helperCommand(commandText) {
        return "python3 "
            + shellEscape(helperPath)
            + " --service "
            + shellEscape(activeService)
            + " "
            + commandText
    }

    function zoomInUi() {
        uiScale = Math.min(maxUiScale, uiScale + zoomStep)
    }

    function zoomOutUi() {
        uiScale = Math.max(minUiScale, uiScale - zoomStep)
    }

    function resetZoomUi() {
        uiScale = 1.0
    }

    function maxSessionAgeDays() {
        const configured = Number(Plasmoid.configuration.maxSessionAgeDays)
        if (!Number.isFinite(configured) || configured < 0) {
            return 7
        }
        return Math.floor(configured)
    }

    function loadPersistedService() {
        const persisted = String(Plasmoid.configuration.selectedService || "").trim()
        if (persisted === productionService || persisted === developmentService) {
            activeService = persisted
        }
        serviceInitialized = true
    }

    function persistActiveService() {
        if (Plasmoid.configuration.selectedService !== activeService) {
            Plasmoid.configuration.selectedService = activeService
        }
    }

    function serviceIndexByValue(serviceName) {
        for (let i = 0; i < serviceChoices.length; i++) {
            if (serviceChoices[i].value === serviceName) {
                return i
            }
        }
        return -1
    }

    function syncServicePicker() {
        const idx = serviceIndexByValue(activeService)
        if (idx >= 0) {
            servicePicker.currentIndex = idx
        }
    }

    function switchService(index) {
        if (index < 0 || index >= serviceChoices.length || busy) {
            return
        }

        const selectedService = serviceChoices[index].value
        if (selectedService === developmentService && !devServiceRunning) {
            appendStatus("Development service is not running")
            syncServicePicker()
            return
        }
        if (selectedService === activeService) {
            return
        }

        activeService = selectedService
        persistActiveService()
        conversationId = ""
        promptText = ""
        currentMessageCount = 0
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "Switched to " + activeService,
            }
        ]
        reloadConversationList()
        refreshConversation()
    }

    function refreshServiceStatus(onReady) {
        const command = "python3 " + shellEscape(helperPath) + " status"
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                productionServiceRunning = !!payload.production_running
                devServiceRunning = !!payload.dev_running
                if (payload.default_service && payload.default_service.length > 0) {
                    productionService = payload.default_service
                }
                if (payload.dev_service && payload.dev_service.length > 0) {
                    developmentService = payload.dev_service
                }

                serviceChoices = [
                    {
                        value: productionService,
                        label: "Production",
                    }
                ]

                if (devServiceRunning) {
                    serviceChoices = serviceChoices.concat([
                        {
                            value: developmentService,
                            label: "Development",
                        }
                    ])
                }

                if (!serviceInitialized) {
                    loadPersistedService()
                }

                if (activeService !== productionService && activeService !== developmentService) {
                    activeService = String(payload.selected_service || productionService)
                }

                if (!devServiceRunning && activeService === developmentService) {
                    appendStatus("Development service stopped")
                }

                syncServicePicker()

                if (onReady) {
                    onReady()
                }
            },
            function(stderr) {
                appendStatus("Service status error: " + stderr)
                if (onReady) {
                    onReady()
                }
            }
        )
    }

    function openSettingsDialog() {
        runCommand(
            "kcmshell6 kcm_desktopassistant",
            function(_stdout) {},
            function(_stderr) {
                appendStatus("Settings dialog failed to open")
            }
        )
    }

    function appendMessage(role, text) {
        if (role === "tool" && !debugEnabled) {
            return
        }
        const normalizedText = String(text === undefined || text === null ? "" : text)
        if (normalizedText.trim().length === 0) {
            return
        }
        transcriptEntries = transcriptEntries.concat([
            {
                kind: "message",
                role: role,
                text: normalizedText,
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

        const command = helperCommand("ensure --title " + shellEscape(conversationTitle))
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

        const title = conversationTitle + " " + Date.now()
        const command = helperCommand("create --title " + shellEscape(title))
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

    function reloadConversationList(onLoaded) {
        const command = helperCommand("list --max-age-days " + maxSessionAgeDays())
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
                    const timestamp = String(conversation.updated_at || "").trim()
                    const titleText = timestamp.length > 0
                        ? (conversation.title + " · " + timestamp)
                        : conversation.title
                    return {
                        id: conversation.id,
                        title: titleText + " (" + conversation.message_count + ")",
                    }
                })

                const idx = conversationIndexById(conversationId)
                if (idx >= 0 && !panelMode) {
                    conversationPicker.currentIndex = idx
                }

                if (onLoaded) {
                    onLoaded(conversations)
                }
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function deleteConversation(targetId) {
        if (busy || targetId.length === 0) {
            return
        }

        const deletingCurrent = targetId === conversationId
        const command = helperCommand("delete " + shellEscape(targetId))
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Failed to delete conversation: " + payload.error)
                    return
                }

                appendStatus("Deleted conversation " + targetId)
                reloadConversationList(function(conversations) {
                    if (!deletingCurrent) {
                        return
                    }

                    if (conversations.length > 0) {
                        conversationId = conversations[0].id
                        conversationPicker.currentIndex = 0
                        refreshConversation()
                        return
                    }

                    conversationId = ""
                    promptText = ""
                    currentMessageCount = 0
                    transcriptEntries = [
                        {
                            kind: "status",
                            role: "status",
                            text: "No conversations yet",
                        }
                    ]
                })
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

        const command = helperCommand("get " + shellEscape(conversationId))
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

            const sendCommand = helperCommand("send " + shellEscape(conversationId) + " " + shellEscape(prompt))
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
                    const awaitCommand = helperCommand(
                        "await "
                        + shellEscape(conversationId)
                        + " --initial-count "
                        + currentMessageCount
                        + " --timeout "
                        + awaitTimeoutSeconds
                    )
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
        onTriggered: {
            if (!root.hideWidget) {
                ensureConversation()
            }
        }
    }

    Timer {
        id: servicePollTimer
        interval: 5000
        repeat: true
        running: true
        onTriggered: refreshServiceStatus()
    }

    Shortcut {
        sequence: "Ctrl++"
        context: Qt.WindowShortcut
        onActivated: root.zoomInUi()
    }

    Shortcut {
        sequence: "Ctrl+="
        context: Qt.WindowShortcut
        onActivated: root.zoomInUi()
    }

    Shortcut {
        sequence: "Ctrl+-"
        context: Qt.WindowShortcut
        onActivated: root.zoomOutUi()
    }

    Shortcut {
        sequence: "Ctrl+_"
        context: Qt.WindowShortcut
        onActivated: root.zoomOutUi()
    }

    Shortcut {
        sequence: "Ctrl+0"
        context: Qt.WindowShortcut
        onActivated: root.resetZoomUi()
    }

    Rectangle {
        anchors.fill: parent
        visible: !root.hideWidget
        color: PlasmaCore.Theme.backgroundColor
        border.width: 1
        border.color: PlasmaCore.Theme.disabledTextColor
        radius: 8
    }

    ColumnLayout {
        visible: !root.hideWidget
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
                sourceSize.width: root.scaledTopIconSize
                sourceSize.height: root.scaledTopIconSize
                fillMode: Image.PreserveAspectFit
                Layout.preferredWidth: root.scaledTopIconSize
                Layout.preferredHeight: root.scaledTopIconSize
            }

            QQC2.Label {
                text: root.activeService === root.developmentService ? "Adele (Dev)" : "Adele"
                font.bold: true
                color: PlasmaCore.Theme.textColor
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

            QQC2.ComboBox {
                id: servicePicker
                visible: root.devServiceRunning
                Layout.preferredWidth: 170
                Layout.fillWidth: false
                model: root.serviceChoices
                textRole: "label"
                palette.text: PlasmaCore.Theme.textColor
                palette.buttonText: PlasmaCore.Theme.textColor
                onActivated: function(index) {
                    switchService(index)
                }
            }

            PlasmaComponents.ComboBox {
                id: conversationPicker
                visible: true
                Layout.fillWidth: true
                model: root.conversationChoices
                textRole: "title"
                delegate: QQC2.ItemDelegate {
                    required property var modelData
                    width: conversationPicker.width

                    contentItem: RowLayout {
                        spacing: 6

                        QQC2.Label {
                            Layout.fillWidth: true
                            text: modelData.title
                            color: PlasmaCore.Theme.textColor
                            elide: Text.ElideRight
                        }

                        QQC2.ToolButton {
                            icon.name: "edit-delete"
                            display: QQC2.AbstractButton.IconOnly
                            enabled: !root.busy
                            onClicked: {
                                root.deleteConversation(modelData.id)
                                conversationPicker.popup.close()
                            }
                        }
                    }

                    onClicked: {
                        const idx = root.conversationIndexById(modelData.id)
                        conversationPicker.currentIndex = idx
                        conversationPicker.popup.close()
                        switchConversation(idx)
                    }
                }
                onActivated: function(index) {
                    switchConversation(index)
                }
            }

            QQC2.Button {
                text: "Load"
                enabled: !busy
                onClicked: panelMode ? refreshServiceStatus() : reloadConversationList()
            }

            QQC2.Button {
                visible: panelMode
                text: "Settings"
                enabled: !busy
                onClicked: openSettingsDialog()
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
                        sourceSize.width: root.scaledHeaderIconSize
                        sourceSize.height: root.scaledHeaderIconSize
                        width: root.scaledHeaderIconSize
                        height: root.scaledHeaderIconSize
                        fillMode: Image.PreserveAspectFit
                        anchors.horizontalCenter: parent.horizontalCenter
                    }

                    QQC2.Label {
                        text: "Hi! I'm Adele! Ask me anything..."
                        font.pointSize: root.baseFontPointSize * root.uiScale
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
                    readonly property real avatarSize: 24 * root.uiScale
                    readonly property var avatarSources: isAssistant ? [root.adeleAvatarSource] : root.userAvatarCandidates()

                    visible: !isStatus || root.debugEnabled
                    width: ListView.view.width
                    implicitHeight: visible ? rowContainer.implicitHeight + 2 : 0

                    RowLayout {
                        id: rowContainer
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        spacing: 6
                        layoutDirection: isStatus ? Qt.LeftToRight : (isAssistant ? Qt.RightToLeft : Qt.LeftToRight)

                        Item {
                            Layout.preferredWidth: isStatus ? 0 : avatarSize
                            Layout.preferredHeight: isStatus ? 0 : avatarSize
                            Layout.alignment: Qt.AlignTop
                            visible: !isStatus

                            Rectangle {
                                anchors.fill: parent
                                radius: width / 2
                                color: PlasmaCore.Theme.backgroundColor
                                border.width: 1
                                border.color: PlasmaCore.Theme.disabledTextColor
                                clip: true

                                Image {
                                    id: avatarImage
                                    property int candidateIndex: 0
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.top: parent.top
                                    width: isAssistant ? parent.width * 1.9 : parent.width
                                    height: isAssistant ? parent.height * 1.9 : parent.height
                                    fillMode: Image.PreserveAspectCrop
                                    horizontalAlignment: Image.AlignHCenter
                                    verticalAlignment: isAssistant ? Image.AlignTop : Image.AlignVCenter
                                    source: avatarSources.length > 0 ? avatarSources[Math.min(candidateIndex, avatarSources.length - 1)] : ""
                                    visible: status === Image.Ready

                                    onStatusChanged: {
                                        if (status === Image.Error && candidateIndex < avatarSources.length - 1) {
                                            candidateIndex += 1
                                        }
                                    }
                                }

                                Kirigami.Icon {
                                    anchors.fill: parent
                                    source: isAssistant ? "preferences-desktop-user" : "user-identity"
                                    visible: !avatarImage.visible
                                }
                            }
                        }

                        Rectangle {
                            id: bubble
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignTop
                            implicitWidth: isStatus
                                ? rowContainer.width
                                : Math.min(rowContainer.width * 0.88, Math.max(120, messageText.implicitWidth + 12))
                            implicitHeight: messageText.implicitHeight + 12
                            height: implicitHeight
                            radius: isStatus ? 0 : 8
                            color: isStatus
                                ? "transparent"
                                : (isAssistant ? PlasmaCore.Theme.backgroundColor : PlasmaCore.Theme.highlightColor)
                            border.width: isStatus ? 0 : 1
                            border.color: PlasmaCore.Theme.disabledTextColor

                            TextEdit {
                                id: messageText
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 6
                                height: contentHeight
                                readOnly: true
                                selectByMouse: true
                                selectByKeyboard: true
                                wrapMode: TextEdit.Wrap
                                textFormat: (modelData.kind === "message" && isAssistant) ? Text.MarkdownText : Text.PlainText
                                text: isStatus
                                    ? "[status] " + String(modelData.text || "")
                                    : String(modelData.text || "")
                                color: isStatus
                                    ? PlasmaCore.Theme.disabledTextColor
                                    : (isAssistant
                                        ? PlasmaCore.Theme.textColor
                                        : PlasmaCore.Theme.highlightedTextColor)
                                font.pointSize: root.baseFontPointSize * root.uiScale
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
                text: "Refresh"
                visible: !panelMode
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
        refreshServiceStatus(function() {
            if (root.hideWidget) {
                appendStatus("Development service not running; widget hidden")
                return
            }
            if (panelMode) {
                ensureConversation()
            } else {
                startupTimer.start()
            }
        })
    }
}
