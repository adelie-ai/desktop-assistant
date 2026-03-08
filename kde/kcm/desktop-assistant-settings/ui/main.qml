import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kcmutils as KCM
import org.kde.kirigami as Kirigami

KCM.SimpleKCM {
    implicitWidth: 560
    implicitHeight: 460

    function indexForName(values, needle) {
        if (!values || values.length === 0) {
            return -1
        }
        for (let i = 0; i < values.length; i++) {
            if (values[i] === needle) {
                return i
            }
        }
        return -1
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        QQC2.TabBar {
            id: tabs
            Layout.fillWidth: true

            QQC2.TabButton { text: "Chat LLM" }
            QQC2.TabButton { text: "Search" }
            QQC2.TabButton { text: "Backend Tasks" }
            QQC2.TabButton { text: "Data Sync" }
            QQC2.TabButton { text: "Connections" }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex

            QQC2.ScrollView {
                clip: true

                ColumnLayout {
                    width: parent.width
                    spacing: 12

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: kcm.hasApiKey
                            ? "API key is configured in the secret backend."
                            : "No API key stored yet."
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Connector" }
                        QQC2.ComboBox {
                            id: connectorBox
                            Layout.fillWidth: true
                            model: ["ollama", "openai", "anthropic", "aws-bedrock"]
                            currentIndex: {
                                if (kcm.connector === "ollama") return 0
                                if (kcm.connector === "openai") return 1
                                if (kcm.connector === "anthropic") return 2
                                if (kcm.connector === "bedrock" || kcm.connector === "aws-bedrock") return 3
                                return 1
                            }
                            onActivated: kcm.connector = currentText
                        }
                    }

                    QQC2.Button {
                        text: "Set Defaults"
                        onClicked: kcm.applyChatDefaults()
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            id: llmModelField
                            Layout.fillWidth: true
                            placeholderText: "gpt-5.4 / llama3.1 / ..."
                            text: kcm.model
                            onTextEdited: kcm.model = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Base URL" }
                        QQC2.TextField {
                            id: llmBaseUrlField
                            Layout.fillWidth: true
                            placeholderText: "https://api.openai.com/v1"
                            text: kcm.baseUrl
                            onTextEdited: kcm.baseUrl = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "API Key" }
                        QQC2.TextField {
                            id: apiKeyField
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                            placeholderText: "Write-only; leave blank to keep existing"
                            text: kcm.apiKeyInput
                            onTextEdited: kcm.apiKeyInput = text
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }

            QQC2.ScrollView {
                clip: true

                ColumnLayout {
                    width: parent.width
                    spacing: 12

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Search helps Adele find relevant past messages and preferences. Choose which service powers search below."
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Provider" }
                        QQC2.ComboBox {
                            id: embConnectorBox
                            Layout.fillWidth: true
                            model: ["auto (same as Chat LLM)", "ollama", "openai", "aws-bedrock"]
                            currentIndex: {
                                if (kcm.embConnector === "ollama") return 1
                                if (kcm.embConnector === "openai") return 2
                                if (kcm.embConnector === "bedrock" || kcm.embConnector === "aws-bedrock") return 3
                                return 0
                            }
                            onActivated: {
                                if (currentIndex === 0) kcm.embConnector = ""
                                else kcm.embConnector = currentText
                            }
                        }
                    }

                    QQC2.Button {
                        text: "Set Defaults"
                        onClicked: kcm.applySearchDefaults()
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            id: embModelField
                            Layout.fillWidth: true
                            placeholderText: {
                                let c = kcm.embConnector || kcm.connector
                                if (c === "bedrock" || c === "aws-bedrock") return "amazon.titan-embed-text-v2:0"
                                return c === "ollama" ? "nomic-embed-text" : "text-embedding-3-small"
                            }
                            text: kcm.embModel
                            onTextEdited: kcm.embModel = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Base URL" }
                        QQC2.TextField {
                            id: embBaseUrlField
                            Layout.fillWidth: true
                            placeholderText: {
                                let c = kcm.embConnector || kcm.connector
                                if (c === "bedrock" || c === "aws-bedrock") return "us-east-1"
                                return c === "ollama" ? "http://localhost:11434" : "https://api.openai.com/v1"
                            }
                            text: kcm.embBaseUrl
                            onTextEdited: kcm.embBaseUrl = text
                        }
                    }

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        visible: !kcm.embAvailable
                        color: Kirigami.Theme.neutralTextColor
                        text: "The current choice cannot power Search right now. Pick another Search provider, or switch the Chat LLM connector."
                    }

                    Item { Layout.fillHeight: true }
                }
            }

            QQC2.ScrollView {
                clip: true

                ColumnLayout {
                    width: parent.width
                    spacing: 12

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Backend tasks use a cheaper LLM for title generation, context summary compaction, and dreaming (periodic fact extraction). When no separate LLM is configured, the primary Chat LLM is used."
                    }

                    QQC2.CheckBox {
                        id: btSeparateLlmCheck
                        text: "Use a separate LLM for backend tasks"
                        checked: kcm.btLlmConnector !== ""
                        onToggled: {
                            if (!checked) {
                                kcm.btLlmConnector = ""
                                kcm.btLlmModel = ""
                                kcm.btLlmBaseUrl = ""
                            } else {
                                kcm.btLlmConnector = "ollama"
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: btSeparateLlmCheck.checked
                        QQC2.Label { text: "Connector" }
                        QQC2.ComboBox {
                            id: btConnectorBox
                            Layout.fillWidth: true
                            model: ["ollama", "openai", "anthropic", "aws-bedrock"]
                            currentIndex: {
                                if (kcm.btLlmConnector === "ollama") return 0
                                if (kcm.btLlmConnector === "openai") return 1
                                if (kcm.btLlmConnector === "anthropic") return 2
                                if (kcm.btLlmConnector === "bedrock" || kcm.btLlmConnector === "aws-bedrock") return 3
                                return 0
                            }
                            onActivated: kcm.btLlmConnector = currentText
                        }
                    }

                    QQC2.Button {
                        text: "Set Defaults"
                        enabled: btSeparateLlmCheck.checked
                        onClicked: kcm.applyBackendDefaults()
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: btSeparateLlmCheck.checked
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            id: btModelField
                            Layout.fillWidth: true
                            placeholderText: "llama3.2:3b / gpt-4o-mini / ..."
                            text: kcm.btLlmModel
                            onTextEdited: kcm.btLlmModel = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: btSeparateLlmCheck.checked
                        QQC2.Label { text: "Base URL" }
                        QQC2.TextField {
                            id: btBaseUrlField
                            Layout.fillWidth: true
                            placeholderText: "http://localhost:11434"
                            text: kcm.btLlmBaseUrl
                            onTextEdited: kcm.btLlmBaseUrl = text
                        }
                    }

                    Kirigami.Separator { Layout.fillWidth: true }

                    QQC2.Label {
                        font.bold: true
                        text: "Dreaming"
                    }

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Dreaming periodically reviews conversations and extracts long-term facts into the knowledge base."
                    }

                    QQC2.CheckBox {
                        id: btDreamingEnabledCheck
                        text: "Enable dreaming"
                        checked: kcm.btDreamingEnabled
                        onToggled: kcm.btDreamingEnabled = checked
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: btDreamingEnabledCheck.checked
                        QQC2.Label { text: "Interval (seconds)" }
                        QQC2.SpinBox {
                            id: btDreamingIntervalBox
                            from: 60
                            to: 86400
                            stepSize: 300
                            value: kcm.btDreamingIntervalSecs
                            onValueModified: kcm.btDreamingIntervalSecs = value
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }

            QQC2.ScrollView {
                clip: true

                ColumnLayout {
                    width: parent.width
                    spacing: 12

                    Kirigami.Separator { Layout.fillWidth: true }

                    QQC2.Label {
                        font.bold: true
                        text: "Database"
                    }

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Optional PostgreSQL database for structured storage. Leave the URL empty to use the built-in SQLite default."
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "URL" }
                        QQC2.TextField {
                            id: dbUrlField
                            Layout.fillWidth: true
                            placeholderText: "postgres://user:pass@localhost/dbname"
                            text: kcm.dbUrl
                            onTextEdited: kcm.dbUrl = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Max Connections" }
                        QQC2.SpinBox {
                            id: dbMaxConnectionsBox
                            from: 1
                            to: 100
                            value: kcm.dbMaxConnections
                            onValueModified: kcm.dbMaxConnections = value
                        }
                    }

                    Kirigami.Separator { Layout.fillWidth: true }

                    QQC2.Label {
                        font.bold: true
                        text: "Git Versioning"
                    }

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Version built-in memory and preferences in a local git repository. Optionally push each update to a remote for backup."
                    }

                    QQC2.CheckBox {
                        id: gitEnabledCheck
                        text: "Enable git versioning for data directory"
                        checked: kcm.gitEnabled
                        onToggled: kcm.gitEnabled = checked
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: gitEnabledCheck.checked
                        QQC2.Label { text: "Remote URL" }
                        QQC2.TextField {
                            id: gitRemoteUrlField
                            Layout.fillWidth: true
                            placeholderText: "git@github.com:you/assistant-memory.git (optional)"
                            text: kcm.gitRemoteUrl
                            onTextEdited: kcm.gitRemoteUrl = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: gitEnabledCheck.checked
                        QQC2.Label { text: "Remote name" }
                        QQC2.TextField {
                            id: gitRemoteNameField
                            Layout.fillWidth: true
                            placeholderText: "origin"
                            text: kcm.gitRemoteName
                            onTextEdited: kcm.gitRemoteName = text
                        }
                    }

                    QQC2.CheckBox {
                        id: gitPushOnUpdateCheck
                        enabled: gitEnabledCheck.checked && gitRemoteUrlField.text.trim() !== ""
                        text: "Push to remote on every update"
                        checked: kcm.gitPushOnUpdate
                        onToggled: kcm.gitPushOnUpdate = checked
                    }

                    Item { Layout.fillHeight: true }
                }
            }

            QQC2.ScrollView {
                clip: true

                ColumnLayout {
                    width: parent.width
                    spacing: 12

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Define Adelie connections once, set a global default, and let each widget pick a connection by name. 'local' is the default fallback, but any configured connection can be selected as default."
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label {
                            text: "Default"
                            Layout.preferredWidth: 120
                        }
                        QQC2.ComboBox {
                            id: defaultConnectionBox
                            Layout.fillWidth: true
                            model: kcm.connectionNames
                            currentIndex: Math.max(0, indexForName(kcm.connectionNames, kcm.defaultConnectionName))
                            onActivated: {
                                if (currentIndex >= 0) {
                                    kcm.defaultConnectionName = currentText
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label {
                            text: "Edit"
                            Layout.preferredWidth: 120
                        }
                        QQC2.ComboBox {
                            id: selectedConnectionBox
                            Layout.fillWidth: true
                            model: kcm.connectionNames
                            currentIndex: Math.max(0, indexForName(kcm.connectionNames, kcm.selectedConnectionName))
                            onActivated: {
                                if (currentIndex >= 0) {
                                    kcm.selectedConnectionName = currentText
                                }
                            }
                        }
                        QQC2.Button {
                            text: "Remove"
                            enabled: kcm.selectedConnectionRemovable
                            onClicked: kcm.removeSelectedConnection()
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label {
                            text: "Add remote"
                            Layout.preferredWidth: 120
                        }
                        QQC2.TextField {
                            id: newConnectionNameField
                            Layout.fillWidth: true
                            placeholderText: "my-cluster"
                            onAccepted: addConnectionButton.clicked()
                        }
                        QQC2.Button {
                            id: addConnectionButton
                            text: "Add"
                            onClicked: {
                                const value = newConnectionNameField.text.trim()
                                if (value.length === 0) {
                                    return
                                }
                                kcm.addRemoteConnection(value)
                                newConnectionNameField.text = ""
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label {
                            text: "Transport"
                            Layout.preferredWidth: 120
                        }
                        QQC2.TextField {
                            Layout.fillWidth: true
                            readOnly: true
                            text: kcm.selectedConnectionTransport
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: kcm.selectedConnectionTransport === "dbus"
                        QQC2.Label {
                            text: "D-Bus service"
                            Layout.preferredWidth: 120
                        }
                        QQC2.TextField {
                            id: selectedConnectionDbusServiceField
                            Layout.fillWidth: true
                            placeholderText: "org.desktopAssistant"
                            text: kcm.selectedConnectionDbusService
                            onTextEdited: kcm.selectedConnectionDbusService = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: kcm.selectedConnectionTransport === "ws"
                        QQC2.Label {
                            text: "WebSocket URL"
                            Layout.preferredWidth: 120
                        }
                        QQC2.TextField {
                            id: selectedConnectionWsUrlField
                            Layout.fillWidth: true
                            placeholderText: "wss://cluster.example.com/ws"
                            text: kcm.selectedConnectionWsUrl
                            onTextEdited: kcm.selectedConnectionWsUrl = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        enabled: kcm.selectedConnectionTransport === "ws"
                        QQC2.Label {
                            text: "JWT subject"
                            Layout.preferredWidth: 120
                        }
                        QQC2.TextField {
                            id: selectedConnectionWsSubjectField
                            Layout.fillWidth: true
                            placeholderText: "desktop-widget"
                            text: kcm.selectedConnectionWsSubject
                            onTextEdited: kcm.selectedConnectionWsSubject = text
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true

            QQC2.Button {
                text: "Reload"
                onClicked: kcm.load()
            }

            QQC2.Button {
                text: "Apply"
                onClicked: kcm.save()
            }

            QQC2.Button {
                text: "Restart Daemon"
                onClicked: kcm.restartDaemon()
            }
        }

        QQC2.Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            text: kcm.statusText
        }

        Connections {
            target: kcm

            function onModelChanged() {
                if (llmModelField.text !== kcm.model) {
                    llmModelField.text = kcm.model
                }
            }

            function onBaseUrlChanged() {
                if (llmBaseUrlField.text !== kcm.baseUrl) {
                    llmBaseUrlField.text = kcm.baseUrl
                }
            }

            function onApiKeyInputChanged() {
                if (apiKeyField.text !== kcm.apiKeyInput) {
                    apiKeyField.text = kcm.apiKeyInput
                }
            }

            function onEmbModelChanged() {
                if (embModelField.text !== kcm.embModel) {
                    embModelField.text = kcm.embModel
                }
            }

            function onEmbBaseUrlChanged() {
                if (embBaseUrlField.text !== kcm.embBaseUrl) {
                    embBaseUrlField.text = kcm.embBaseUrl
                }
            }

            function onDbUrlChanged() {
                if (dbUrlField.text !== kcm.dbUrl) {
                    dbUrlField.text = kcm.dbUrl
                }
            }

            function onDbMaxConnectionsChanged() {
                if (dbMaxConnectionsBox.value !== kcm.dbMaxConnections) {
                    dbMaxConnectionsBox.value = kcm.dbMaxConnections
                }
            }

            function onGitRemoteUrlChanged() {
                if (gitRemoteUrlField.text !== kcm.gitRemoteUrl) {
                    gitRemoteUrlField.text = kcm.gitRemoteUrl
                }
            }

            function onGitRemoteNameChanged() {
                if (gitRemoteNameField.text !== kcm.gitRemoteName) {
                    gitRemoteNameField.text = kcm.gitRemoteName
                }
            }

            function onSelectedConnectionDbusServiceChanged() {
                if (selectedConnectionDbusServiceField.text !== kcm.selectedConnectionDbusService) {
                    selectedConnectionDbusServiceField.text = kcm.selectedConnectionDbusService
                }
            }

            function onSelectedConnectionWsUrlChanged() {
                if (selectedConnectionWsUrlField.text !== kcm.selectedConnectionWsUrl) {
                    selectedConnectionWsUrlField.text = kcm.selectedConnectionWsUrl
                }
            }

            function onSelectedConnectionWsSubjectChanged() {
                if (selectedConnectionWsSubjectField.text !== kcm.selectedConnectionWsSubject) {
                    selectedConnectionWsSubjectField.text = kcm.selectedConnectionWsSubject
                }
            }

            function onBtDreamingIntervalSecsChanged() {
                if (btDreamingIntervalBox.value !== kcm.btDreamingIntervalSecs) {
                    btDreamingIntervalBox.value = kcm.btDreamingIntervalSecs
                }
            }

            function onBtLlmModelChanged() {
                if (btModelField.text !== kcm.btLlmModel) {
                    btModelField.text = kcm.btLlmModel
                }
            }

            function onBtLlmBaseUrlChanged() {
                if (btBaseUrlField.text !== kcm.btLlmBaseUrl) {
                    btBaseUrlField.text = kcm.btLlmBaseUrl
                }
            }
        }
    }
}
