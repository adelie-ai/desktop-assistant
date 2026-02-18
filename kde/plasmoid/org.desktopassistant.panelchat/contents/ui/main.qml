import QtQuick
import QtQuick.Layouts
import QtCore
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents

PlasmoidItem {
    id: root
    readonly property string xdgDataHome: String(StandardPaths.writableLocation(StandardPaths.GenericDataLocation) || "")
    readonly property string normalizedDataHome: xdgDataHome.indexOf("file://") === 0 ? xdgDataHome.substring(7) : xdgDataHome
    readonly property string sharedModuleChatViewPath: "file://" + normalizedDataHome + "/desktop-assistant/chat-module/ui/ChatView.qml"

    preferredRepresentation: compactRepresentation
    switchWidth: 320
    switchHeight: 420
    Plasmoid.status: PlasmaCore.Types.ActiveStatus

    compactRepresentation: PlasmaComponents.ToolButton {
        text: "Adele AI"
        icon.source: Qt.resolvedUrl("../images/adele.png")
        icon.width: Kirigami.Units.iconSizes.smallMedium
        icon.height: Kirigami.Units.iconSizes.smallMedium
        onClicked: root.expanded = !root.expanded
    }

    fullRepresentation: Item {
        Layout.minimumWidth: 320
        Layout.minimumHeight: 420

        Loader {
            id: chatViewLoader
            anchors.fill: parent
            property int sourceIndex: 0
            readonly property var sourceCandidates: [
                sharedModuleChatViewPath,
                Qt.resolvedUrl("../../../org.desktopassistant.desktopchat/contents/ui/ChatView.qml")
            ]
            source: sourceCandidates[sourceIndex]
            onLoaded: {
                if (item) {
                    item.panelMode = true
                }
            }
            onStatusChanged: {
                if (status === Loader.Error && sourceIndex < sourceCandidates.length - 1) {
                    sourceIndex += 1
                    source = sourceCandidates[sourceIndex]
                }
            }
        }
    }
}
