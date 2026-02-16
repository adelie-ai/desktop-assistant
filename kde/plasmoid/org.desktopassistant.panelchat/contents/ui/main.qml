import QtQuick
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents

PlasmoidItem {
    id: root

    preferredRepresentation: Plasmoid.compactRepresentation
    switchWidth: 320
    switchHeight: 420
    Plasmoid.status: PlasmaCore.Types.ActiveStatus

    compactRepresentation: PlasmaComponents.ToolButton {
        text: "Adele AI"
        icon.source: Qt.resolvedUrl("../images/adele.png")
        icon.width: PlasmaCore.Units.iconSizes.smallMedium
        icon.height: PlasmaCore.Units.iconSizes.smallMedium
        onClicked: root.expanded = !root.expanded
    }

    fullRepresentation: Item {
        Layout.minimumWidth: 320
        Layout.minimumHeight: 420

        Loader {
            id: chatViewLoader
            anchors.fill: parent
            source: Qt.resolvedUrl("../../../org.desktopassistant.desktopchat/contents/ui/ChatView.qml")
            onLoaded: {
                if (item) {
                    item.panelMode = true
                }
            }
        }
    }
}
