import QtQuick
import QtCore
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore

PlasmoidItem {
    id: root
    implicitWidth: 460
    implicitHeight: 560
    readonly property string xdgDataHome: String(StandardPaths.writableLocation(StandardPaths.GenericDataLocation) || "")
    readonly property string normalizedDataHome: xdgDataHome.indexOf("file://") === 0 ? xdgDataHome.substring(7) : xdgDataHome
    readonly property string sharedModuleChatViewPath: "file://" + normalizedDataHome + "/desktop-assistant/chat-module/ui/ChatView.qml"

    Plasmoid.status: (chatViewLoader.item && chatViewLoader.item.hideWidget)
        ? PlasmaCore.Types.HiddenStatus
        : PlasmaCore.Types.ActiveStatus

    Loader {
        id: chatViewLoader
        anchors.fill: parent
        property int sourceIndex: 0
        readonly property var sourceCandidates: [
            sharedModuleChatViewPath,
            Qt.resolvedUrl("./ChatView.qml")
        ]
        source: sourceCandidates[sourceIndex]
        onLoaded: {
            if (item) {
                item.panelMode = false
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
