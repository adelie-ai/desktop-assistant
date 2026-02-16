import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import QtCore
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.plasma5support as Plasma5Support

PlasmoidItem {
    id: root

    preferredRepresentation: Plasmoid.compactRepresentation
    switchWidth: 320
    switchHeight: 420
    Plasmoid.status: root.hideWidget ? PlasmaCore.Types.HiddenStatus : PlasmaCore.Types.ActiveStatus

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
    property var transcriptEntries: []
    property string promptText: ""
    readonly property string adeleAvatarSource: Qt.resolvedUrl("../images/adele.png")
    property string configuredUserAvatarPath: String(Plasmoid.configuration.userAvatarPath || "").trim()
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
        candidates.push(toImageSource(homeDirectory + "/.face.icon"))
        candidates.push(toImageSource(homeDirectory + "/.face"))
        if (accountName.length > 0) {
            candidates.push(toImageSource("/var/lib/AccountsService/icons/" + accountName))
        }
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
        currentMessageCount = 0
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "Switched to " + activeService,
            }
        ]
        ensureConversation(function() {
            appendDebugStatus("Connected to conversation " + conversationId)
        })
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
        let command = "kcmshell6 kcm_desktopassistant"
        runCommand(
            command,
            function(_stdout) {},
            function(stderr) {
                appendStatus("Settings error: " + stderr)
            }
        )
    }

    function ensureConversation(onReady) {
        if (conversationId.length > 0) {
            if (onReady) {
                onReady()
            }
            return
        }
        let command = helperCommand("ensure --title 'Panel Chat'")
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
        let command = helperCommand("create --title " + shellEscape(title))
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
        let command = helperCommand("get " + shellEscape(conversationId))
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

            let getCommand = helperCommand("get " + shellEscape(conversationId))
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

                    let sendCommand = helperCommand("send " + shellEscape(conversationId) + " " + shellEscape(prompt))
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
                            let awaitCommand = helperCommand("await " + shellEscape(conversationId) + " --initial-count " + currentMessageCount)
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

    Plasmoid.contextualActions: [
        PlasmaCore.Action {
            text: "Settings"
            icon.name: "settings-configure"
            onTriggered: root.openSettingsDialog()
        }
    ]

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
        id: servicePollTimer
        interval: 5000
        repeat: true
        running: true
        onTriggered: refreshServiceStatus()
    }

    compactRepresentation: PlasmaComponents.ToolButton {
        text: root.activeService === root.developmentService ? "Adele (Dev)" : "Adele"
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
                    text: root.activeService === root.developmentService ? "Adele (Dev)" : "Adele"
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

                QQC2.ComboBox {
                    id: servicePicker
                    Layout.fillWidth: true
                    model: root.serviceChoices
                    textRole: "label"
                    onActivated: function(index) {
                        switchService(index)
                    }
                }

                QQC2.Button {
                    text: "Load"
                    enabled: !busy
                    onClicked: refreshServiceStatus()
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
                        readonly property real avatarSize: 24
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
                                        anchors.fill: parent
                                        fillMode: Image.PreserveAspectCrop
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
                                        : String(modelData.text || "")
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
        refreshServiceStatus(function() {
            if (root.hideWidget) {
                appendStatus("Development service not running; widget hidden")
                return
            }
            ensureConversation(function() {
                appendDebugStatus("Connected to conversation " + conversationId)
            })
        })
    }
}
