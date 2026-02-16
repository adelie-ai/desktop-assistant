import QtQuick
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore

PlasmoidItem {
    id: root
    implicitWidth: 460
    implicitHeight: 560
    Plasmoid.status: chatView.hideWidget ? PlasmaCore.Types.HiddenStatus : PlasmaCore.Types.ActiveStatus

    ChatView {
        id: chatView
        anchors.fill: parent
        panelMode: false
    }
}
