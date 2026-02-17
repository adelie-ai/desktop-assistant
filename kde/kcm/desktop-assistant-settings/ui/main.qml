import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kcmutils as KCM

KCM.SimpleKCM {
    implicitWidth: 520
    implicitHeight: 420

    function normalizedConnector(value) {
        const connector = (value || "").toLowerCase()
        return connector.length > 0 ? connector : "openai"
    }

    function applyChatDefaults() {
        const connector = normalizedConnector(kcm.connector)
        if (connector === "ollama") {
            kcm.model = "qwen3:0.6b"
            kcm.baseUrl = "http://localhost:11434"
            return
        }

        if (connector === "anthropic") {
            kcm.model = "claude-sonnet-4-5-20250929"
            kcm.baseUrl = "https://api.anthropic.com"
            return
        }

        kcm.model = "gpt-5.2"
        kcm.baseUrl = "https://api.openai.com/v1"
    }

    function effectiveSearchConnector() {
        const connector = normalizedConnector(kcm.embConnector || kcm.connector)
        return connector === "anthropic" ? "openai" : connector
    }

    function applySearchDefaults() {
        const connector = effectiveSearchConnector()
        if (connector === "ollama") {
            kcm.embModel = "nomic-embed-text"
            kcm.embBaseUrl = "http://localhost:11434"
            if (kcm.embConnector === "anthropic") {
                kcm.embConnector = "ollama"
            }
            return
        }

        kcm.embModel = "text-embedding-3-small"
        kcm.embBaseUrl = "https://api.openai.com/v1"
        if (kcm.embConnector === "anthropic") {
            kcm.embConnector = "openai"
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        QQC2.TabBar {
            id: tabs
            Layout.fillWidth: true

            QQC2.TabButton { text: "Chat LLM" }
            QQC2.TabButton { text: "Search" }
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
                            model: ["ollama", "openai", "anthropic"]
                            currentIndex: Math.max(0, model.indexOf(kcm.connector))
                            onActivated: kcm.connector = currentText
                        }
                    }

                    QQC2.Button {
                        text: "Set Defaults"
                        onClicked: applyChatDefaults()
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            id: llmModelField
                            Layout.fillWidth: true
                            placeholderText: "gpt-4o / llama3.1 / ..."
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
                            model: ["auto (same as Chat LLM)", "ollama", "openai"]
                            currentIndex: {
                                if (kcm.embConnector === "ollama") return 1
                                if (kcm.embConnector === "openai") return 2
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
                        onClicked: applySearchDefaults()
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            id: embModelField
                            Layout.fillWidth: true
                            placeholderText: {
                                let c = kcm.embConnector || kcm.connector
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
                        color: "orange"
                        text: "The current choice cannot power Search right now. Pick another Search provider, or switch the Chat LLM connector."
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
        }
    }
}
