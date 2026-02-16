import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.plasma5support as Plasma5Support

PlasmoidItem {
    id: root

    preferredRepresentation: Plasmoid.compactRepresentation
    switchWidth: 440
    switchHeight: 520

    property string helperPath: Qt.resolvedUrl("../code/config_helper.py").toString().replace("file://", "")
    property bool busy: false

    property string connector: "openai"
    property string model: ""
    property string baseUrl: ""
    property string apiKey: ""
    property bool hasApiKey: false
    property string dbusService: "org.desktopAssistant"
    property bool devMode: false
    property string statusText: "Ready"

    readonly property var pending: ({})

    function shellEscape(value) {
        return "'" + value.replace(/'/g, "'\\''") + "'"
    }

    function runCommand(command, onSuccess, onError) {
        pending[command] = {
            success: onSuccess,
            error: onError,
        }
        executable.connectSource(command)
    }

    function loadSettings() {
        busy = true
        let command = "python3 " + shellEscape(helperPath) + " load"
        runCommand(
            command,
            function(stdout) {
                busy = false
                let payload = JSON.parse(stdout)
                if (payload.error) {
                    statusText = payload.error
                    return
                }
                connector = payload.connector || "openai"
                model = payload.model || ""
                baseUrl = payload.base_url || ""
                hasApiKey = !!payload.has_api_key
                dbusService = payload.dbus_service || "org.desktopAssistant"
                devMode = dbusService === "org.desktopAssistant.Dev"
                apiKey = ""
                statusText = "Loaded settings (D-Bus: " + dbusService + ")"
            },
            function(stderr) {
                busy = false
                statusText = stderr
            }
        )
    }

    function saveSettings(restartService) {
        busy = true
        let command = "python3 " + shellEscape(helperPath)
            + " save"
            + " --connector " + shellEscape(connector)
            + " --model " + shellEscape(model)
            + " --base-url " + shellEscape(baseUrl)
            + " --dbus-service " + shellEscape(devMode ? "org.desktopAssistant.Dev" : "org.desktopAssistant")
            + " --api-key " + shellEscape(apiKey)

        runCommand(
            command,
            function(stdout) {
                let payload = JSON.parse(stdout)
                if (payload.error) {
                    busy = false
                    statusText = payload.error
                    return
                }
                dbusService = payload.dbus_service || (devMode ? "org.desktopAssistant.Dev" : "org.desktopAssistant")
                statusText = "Saved settings (D-Bus: " + dbusService + ")"
                if (!restartService) {
                    busy = false
                    return
                }
                restartDaemon()
            },
            function(stderr) {
                busy = false
                statusText = stderr
            }
        )
    }

    function restartDaemon() {
        let command = "python3 " + shellEscape(helperPath) + " restart"
        runCommand(
            command,
            function(stdout) {
                busy = false
                let payload = JSON.parse(stdout)
                statusText = payload.error ? payload.error : "Saved + restarted desktop-assistant-daemon"
            },
            function(stderr) {
                busy = false
                statusText = stderr
            }
        )
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
                callbacks.success(stdout)
            } else {
                callbacks.error(stderr.length > 0 ? stderr : stdout)
            }

            delete pending[sourceName]
            disconnectSource(sourceName)
        }
    }

    compactRepresentation: PlasmaComponents.ToolButton {
        text: "Adele Settings"
        icon.name: "settings-configure"
        onClicked: root.expanded = !root.expanded
    }

    fullRepresentation: Item {
        Layout.minimumWidth: 420
        Layout.minimumHeight: 500

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 8

            RowLayout {
                Layout.fillWidth: true
                spacing: 6

                Image {
                    source: Qt.resolvedUrl("../images/adele_with_text.png")
                    sourceSize.width: 28
                    sourceSize.height: 28
                    fillMode: Image.PreserveAspectFit
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 28
                }

                QQC2.Label {
                    text: "Adele Settings"
                    font.bold: true
                    Layout.fillWidth: true
                }
            }

            QQC2.Label {
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                text: hasApiKey
                    ? "API key is configured in the secret backend."
                    : "No API key stored yet. Enter one and click Save."
            }

            RowLayout {
                Layout.fillWidth: true

                QQC2.Label { text: "Connector" }
                QQC2.ComboBox {
                    id: connectorBox
                    Layout.fillWidth: true
                    model: ["openai", "ollama", "custom"]
                    currentIndex: Math.max(0, model.indexOf(root.connector))
                    onActivated: {
                        root.connector = currentText
                        if (currentText === "ollama" && root.baseUrl.length === 0) {
                            root.baseUrl = "http://localhost:11434/v1"
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                QQC2.Label { text: "Model" }
                QQC2.TextField {
                    Layout.fillWidth: true
                    placeholderText: "gpt-4o / llama3.1 / ..."
                    text: root.model
                    onTextChanged: root.model = text
                }
            }

            RowLayout {
                Layout.fillWidth: true
                QQC2.Label { text: "Base URL" }
                QQC2.TextField {
                    Layout.fillWidth: true
                    placeholderText: "https://api.openai.com/v1"
                    text: root.baseUrl
                    onTextChanged: root.baseUrl = text
                }
            }

            RowLayout {
                Layout.fillWidth: true
                QQC2.Label { text: "API Key" }
                QQC2.TextField {
                    Layout.fillWidth: true
                    echoMode: TextInput.Password
                    placeholderText: "Write-only; leave blank to keep existing"
                    text: root.apiKey
                    onTextChanged: root.apiKey = text
                }
            }

            RowLayout {
                Layout.fillWidth: true
                QQC2.Label { text: "Mode" }
                QQC2.ComboBox {
                    id: modeBox
                    Layout.fillWidth: true
                    model: ["Production", "Development"]
                    currentIndex: root.devMode ? 1 : 0
                    onActivated: {
                        root.devMode = (currentIndex === 1)
                    }
                }
            }

            QQC2.Label {
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                text: root.devMode
                    ? "Development mode targets org.desktopAssistant.Dev. Run `just dev-backend` to launch it."
                    : "Production mode targets org.desktopAssistant (systemd user service)."
            }

            RowLayout {
                Layout.fillWidth: true

                QQC2.Button {
                    text: "Load"
                    enabled: !busy
                    onClicked: loadSettings()
                }

                QQC2.Button {
                    text: "Save"
                    enabled: !busy
                    onClicked: saveSettings(false)
                }

                QQC2.Button {
                    text: "Save + Restart"
                    enabled: !busy
                    onClicked: saveSettings(true)
                }
            }

            QQC2.Label {
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                text: root.statusText
            }

            Item { Layout.fillHeight: true }
        }
    }

    Component.onCompleted: loadSettings()
}
