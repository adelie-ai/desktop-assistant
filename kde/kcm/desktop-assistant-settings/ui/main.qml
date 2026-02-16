import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kcmutils as KCM

KCM.SimpleKCM {
    implicitWidth: 520
    implicitHeight: 420

    ColumnLayout {
        anchors.fill: parent
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

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: Qt.rgba(0.5, 0.5, 0.5, 0.3)
        }

        QQC2.Label {
            text: "Embeddings"
            font.bold: true
            Layout.fillWidth: true
        }

        RowLayout {
            Layout.fillWidth: true
            QQC2.Label { text: "Connector" }
            QQC2.ComboBox {
                id: embConnectorBox
                Layout.fillWidth: true
                model: ["auto (use LLM)", "ollama", "openai"]
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
            text: "Current connector does not support embeddings. Choose a different embeddings connector or switch the LLM connector."
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

        Item { Layout.fillHeight: true }
    }
}
