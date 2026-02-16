import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kcmutils as KCM

KCM.SimpleKCM {
    implicitWidth: 520
    implicitHeight: 420

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

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            Layout.fillWidth: true
                            placeholderText: "gpt-4o / llama3.1 / ..."
                            text: kcm.model
                            onTextChanged: kcm.model = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Base URL" }
                        QQC2.TextField {
                            Layout.fillWidth: true
                            placeholderText: "https://api.openai.com/v1"
                            text: kcm.baseUrl
                            onTextChanged: kcm.baseUrl = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "API Key" }
                        QQC2.TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                            placeholderText: "Write-only; leave blank to keep existing"
                            text: kcm.apiKeyInput
                            onTextChanged: kcm.apiKeyInput = text
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
                        text: "Search uses semantic matching to find relevant memories and preferences. Configure the provider below."
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Provider" }
                        QQC2.ComboBox {
                            id: embConnectorBox
                            Layout.fillWidth: true
                            model: ["auto (use Chat LLM)", "ollama", "openai"]
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

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Model" }
                        QQC2.TextField {
                            Layout.fillWidth: true
                            placeholderText: {
                                let c = kcm.embConnector || kcm.connector
                                return c === "ollama" ? "nomic-embed-text" : "text-embedding-3-small"
                            }
                            text: kcm.embModel
                            onTextChanged: kcm.embModel = text
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        QQC2.Label { text: "Base URL" }
                        QQC2.TextField {
                            Layout.fillWidth: true
                            placeholderText: {
                                let c = kcm.embConnector || kcm.connector
                                return c === "ollama" ? "http://localhost:11434" : "https://api.openai.com/v1"
                            }
                            text: kcm.embBaseUrl
                            onTextChanged: kcm.embBaseUrl = text
                        }
                    }

                    QQC2.Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        visible: !kcm.embAvailable
                        color: "orange"
                        text: "Current connector does not support search indexing. Choose a different search provider or switch the Chat LLM connector."
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
    }
}
