#include "desktopassistantkcm.h"

#include <QDBusInterface>
#include <QDBusMessage>
#include <QDBusReply>
#include <QProcess>

#include <KPluginFactory>

namespace {
constexpr auto SERVICE = "org.desktopAssistant";
constexpr auto PATH = "/org/desktopAssistant/Settings";
constexpr auto IFACE = "org.desktopAssistant.Settings";
}

K_PLUGIN_CLASS_WITH_JSON(DesktopAssistantKcm, "kcm_desktopassistant.json")

DesktopAssistantKcm::DesktopAssistantKcm(QObject *parent, const KPluginMetaData &metaData, const QVariantList &args)
    : KQuickConfigModule(parent, metaData)
{
    Q_UNUSED(args);
    setButtons(Apply | Default);
    load();
}

QString DesktopAssistantKcm::connector() const
{
    return m_connector;
}

void DesktopAssistantKcm::setConnector(const QString &value)
{
    if (m_connector == value) {
        return;
    }

    m_connector = value;
    Q_EMIT connectorChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::model() const
{
    return m_model;
}

void DesktopAssistantKcm::setModel(const QString &value)
{
    if (m_model == value) {
        return;
    }

    m_model = value;
    Q_EMIT modelChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::baseUrl() const
{
    return m_baseUrl;
}

void DesktopAssistantKcm::setBaseUrl(const QString &value)
{
    if (m_baseUrl == value) {
        return;
    }

    m_baseUrl = value;
    Q_EMIT baseUrlChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::apiKeyInput() const
{
    return m_apiKeyInput;
}

void DesktopAssistantKcm::setApiKeyInput(const QString &value)
{
    if (m_apiKeyInput == value) {
        return;
    }

    m_apiKeyInput = value;
    Q_EMIT apiKeyInputChanged();
    setNeedsSave(true);
}

bool DesktopAssistantKcm::hasApiKey() const
{
    return m_hasApiKey;
}

QString DesktopAssistantKcm::statusText() const
{
    return m_statusText;
}

void DesktopAssistantKcm::load()
{
    QDBusInterface iface(SERVICE, PATH, IFACE, QDBusConnection::sessionBus());
    QDBusMessage reply = iface.call("GetLlmSettings");

    if (setStatusFromDbusError(reply)) {
        return;
    }

    const auto args = reply.arguments();
    if (args.size() < 4) {
        m_statusText = QStringLiteral("Unexpected GetLlmSettings reply");
        Q_EMIT statusTextChanged();
        return;
    }

    m_connector = args[0].toString();
    m_model = args[1].toString();
    m_baseUrl = args[2].toString();
    m_hasApiKey = args[3].toBool();
    m_apiKeyInput.clear();

    Q_EMIT connectorChanged();
    Q_EMIT modelChanged();
    Q_EMIT baseUrlChanged();
    Q_EMIT hasApiKeyChanged();
    Q_EMIT apiKeyInputChanged();

    m_statusText = QStringLiteral("Loaded settings from desktop-assistant daemon");
    Q_EMIT statusTextChanged();
    setNeedsSave(false);
}

void DesktopAssistantKcm::save()
{
    QDBusInterface iface(SERVICE, PATH, IFACE, QDBusConnection::sessionBus());

    QDBusMessage settingsReply = iface.call("SetLlmSettings", m_connector, m_model, m_baseUrl);
    if (setStatusFromDbusError(settingsReply)) {
        return;
    }

    if (!m_apiKeyInput.trimmed().isEmpty()) {
        QDBusMessage secretReply = iface.call("SetApiKey", m_apiKeyInput);
        if (setStatusFromDbusError(secretReply)) {
            return;
        }
        m_apiKeyInput.clear();
        Q_EMIT apiKeyInputChanged();
        if (!m_hasApiKey) {
            m_hasApiKey = true;
            Q_EMIT hasApiKeyChanged();
        }
    }

    m_statusText = QStringLiteral("Saved settings");
    Q_EMIT statusTextChanged();
    setNeedsSave(false);
}

void DesktopAssistantKcm::defaults()
{
    setConnector(QStringLiteral("openai"));
    setModel(QString());
    setBaseUrl(QString());
    setApiKeyInput(QString());
    m_statusText = QStringLiteral("Restored form defaults; click Apply to save");
    Q_EMIT statusTextChanged();
}

void DesktopAssistantKcm::restartDaemon()
{
    QProcess process;
    process.start(QStringLiteral("systemctl"), {QStringLiteral("--user"), QStringLiteral("restart"), QStringLiteral("desktop-assistant-daemon")});
    process.waitForFinished(10000);

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        m_statusText = QStringLiteral("Failed to restart daemon: ") + QString::fromUtf8(process.readAllStandardError()).trimmed();
        if (m_statusText.trimmed().endsWith(QLatin1Char(':'))) {
            m_statusText = QStringLiteral("Failed to restart daemon");
        }
    } else {
        m_statusText = QStringLiteral("Restarted desktop-assistant-daemon");
    }
    Q_EMIT statusTextChanged();
}

bool DesktopAssistantKcm::setStatusFromDbusError(const QDBusMessage &message)
{
    if (message.type() != QDBusMessage::ErrorMessage) {
        return false;
    }

    m_statusText = message.errorMessage();
    if (m_statusText.isEmpty()) {
        m_statusText = QStringLiteral("D-Bus call failed");
    }
    Q_EMIT statusTextChanged();
    return true;
}

#include "desktopassistantkcm.moc"
