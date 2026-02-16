import QtQuick
import QtCore
import QtQuick.Layouts
import QtQuick.Controls as QQC2
import org.kde.kirigami as Kirigami

Kirigami.FormLayout {
    property alias cfg_userAvatarPath: userAvatarPathField.text
    property alias cfg_maxSessionAgeDays: maxSessionAgeSpinBox.value
    readonly property string homeDirectory: StandardPaths.writableLocation(StandardPaths.HomeLocation)
    readonly property string accountName: {
        const trimmedHome = String(homeDirectory || "").replace(/\/+$/, "")
        const chunks = trimmedHome.split("/").filter(function(chunk) {
            return chunk.length > 0
        })
        return chunks.length > 0 ? chunks[chunks.length - 1] : ""
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

    function avatarCandidates() {
        const candidates = []
        const configured = toImageSource(userAvatarPathField.text)
        if (configured.length > 0) {
            candidates.push(configured)
        }
        if (accountName.length > 0) {
            candidates.push(toImageSource("/var/lib/AccountsService/icons/" + accountName))
        }
        candidates.push(toImageSource(homeDirectory + "/.face.icon"))
        candidates.push(toImageSource(homeDirectory + "/.face"))
        return candidates
    }

    readonly property var previewCandidates: avatarCandidates()

    QQC2.TextField {
        id: userAvatarPathField
        Kirigami.FormData.label: i18n("User avatar path")
        placeholderText: i18n("/home/you/.face.icon or file:///…")
        ToolTip.visible: hovered
        ToolTip.text: i18n("Leave empty to use your system user avatar")
    }

    QQC2.Label {
        Kirigami.FormData.isSection: false
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        color: Kirigami.Theme.disabledTextColor
        text: i18n("Leave empty to use the avatar from System Settings user profile.")
    }

    QQC2.SpinBox {
        id: maxSessionAgeSpinBox
        Kirigami.FormData.label: i18n("Max session age (days)")
        from: 0
        to: 365
        stepSize: 1
        editable: true
        value: 7
        ToolTip.visible: hovered
        ToolTip.text: i18n("Hide previous sessions older than this many days (0 disables filtering)")
    }

    Item {
        Kirigami.FormData.label: i18n("Preview")
        implicitWidth: 32
        implicitHeight: 32

        Rectangle {
            anchors.fill: parent
            radius: width / 2
            color: Kirigami.Theme.backgroundColor
            border.width: 1
            border.color: Kirigami.Theme.disabledTextColor
            clip: true

            Image {
                id: previewImage
                property int candidateIndex: 0
                anchors.fill: parent
                fillMode: Image.PreserveAspectCrop
                source: previewCandidates.length > 0 ? previewCandidates[Math.min(candidateIndex, previewCandidates.length - 1)] : ""
                visible: status === Image.Ready

                onSourceChanged: {
                    candidateIndex = 0
                }

                onStatusChanged: {
                    if (status === Image.Error && candidateIndex < previewCandidates.length - 1) {
                        candidateIndex += 1
                    }
                }
            }

            Kirigami.Icon {
                anchors.fill: parent
                source: "user-identity"
                visible: !previewImage.visible
            }
        }
    }
}
