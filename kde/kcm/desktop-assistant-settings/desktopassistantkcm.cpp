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

QString normalizeConnector(const QString &connector)
{
    const auto normalized = connector.trimmed().toLower();
    return normalized.isEmpty() ? QStringLiteral("openai") : normalized;
}

struct ConnectorDefaults {
    QString llmModel;
    QString llmBaseUrl;
    QString embeddingsModel;
    QString embeddingsBaseUrl;
    bool embeddingsAvailable = true;
};

bool fetchConnectorDefaults(
    QDBusInterface &iface,
    const QString &connector,
    ConnectorDefaults *out,
    QString *errorText
)
{
    QDBusMessage reply = iface.call("GetConnectorDefaults", connector);
    if (reply.type() == QDBusMessage::ErrorMessage) {
        if (errorText != nullptr) {
            *errorText = reply.errorMessage().isEmpty() ? QStringLiteral("D-Bus call failed") : reply.errorMessage();
        }
        return false;
    }

    const auto args = reply.arguments();
    if (args.size() < 5) {
        if (errorText != nullptr) {
            *errorText = QStringLiteral("Unexpected GetConnectorDefaults reply");
        }
        return false;
    }

    if (out != nullptr) {
        out->llmModel = args[0].toString();
        out->llmBaseUrl = args[1].toString();
        out->embeddingsModel = args[2].toString();
        out->embeddingsBaseUrl = args[3].toString();
        out->embeddingsAvailable = args[4].toBool();
    }

    return true;
}
}

K_PLUGIN_CLASS_WITH_JSON(DesktopAssistantKcm, "kcm_desktopassistant.json")

DesktopAssistantKcm::DesktopAssistantKcm(QObject *parent, const KPluginMetaData &metaData, const QVariantList &args)
    : KQuickConfigModule(parent, metaData)
{
    Q_UNUSED(args);
    setButtons(Apply);
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

QString DesktopAssistantKcm::embConnector() const
{
    return m_embConnector;
}

void DesktopAssistantKcm::setEmbConnector(const QString &value)
{
    if (m_embConnector == value) {
        return;
    }

    m_embConnector = value;
    Q_EMIT embConnectorChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::embModel() const
{
    return m_embModel;
}

void DesktopAssistantKcm::setEmbModel(const QString &value)
{
    if (m_embModel == value) {
        return;
    }

    m_embModel = value;
    Q_EMIT embModelChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::embBaseUrl() const
{
    return m_embBaseUrl;
}

void DesktopAssistantKcm::setEmbBaseUrl(const QString &value)
{
    if (m_embBaseUrl == value) {
        return;
    }

    m_embBaseUrl = value;
    Q_EMIT embBaseUrlChanged();
    setNeedsSave(true);
}

bool DesktopAssistantKcm::embHasApiKey() const
{
    return m_embHasApiKey;
}

bool DesktopAssistantKcm::embAvailable() const
{
    return m_embAvailable;
}

bool DesktopAssistantKcm::embIsDefault() const
{
    return m_embIsDefault;
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

bool DesktopAssistantKcm::gitEnabled() const
{
    return m_gitEnabled;
}

void DesktopAssistantKcm::setGitEnabled(bool value)
{
    if (m_gitEnabled == value) {
        return;
    }
    m_gitEnabled = value;
    Q_EMIT gitEnabledChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::gitRemoteUrl() const
{
    return m_gitRemoteUrl;
}

void DesktopAssistantKcm::setGitRemoteUrl(const QString &value)
{
    if (m_gitRemoteUrl == value) {
        return;
    }
    m_gitRemoteUrl = value;
    Q_EMIT gitRemoteUrlChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::gitRemoteName() const
{
    return m_gitRemoteName;
}

void DesktopAssistantKcm::setGitRemoteName(const QString &value)
{
    if (m_gitRemoteName == value) {
        return;
    }
    m_gitRemoteName = value;
    Q_EMIT gitRemoteNameChanged();
    setNeedsSave(true);
}

bool DesktopAssistantKcm::gitPushOnUpdate() const
{
    return m_gitPushOnUpdate;
}

void DesktopAssistantKcm::setGitPushOnUpdate(bool value)
{
    if (m_gitPushOnUpdate == value) {
        return;
    }
    m_gitPushOnUpdate = value;
    Q_EMIT gitPushOnUpdateChanged();
    setNeedsSave(true);
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

    QDBusMessage embReply = iface.call("GetEmbeddingsSettings");
    if (setStatusFromDbusError(embReply)) {
        return;
    }

    const auto embArgs = embReply.arguments();
    if (embArgs.size() < 6) {
        m_statusText = QStringLiteral("Unexpected GetEmbeddingsSettings reply");
        Q_EMIT statusTextChanged();
        return;
    }

    m_embConnector = embArgs[5].toBool() ? QString() : embArgs[0].toString();
    m_embModel = embArgs[1].toString();
    m_embBaseUrl = embArgs[2].toString();
    m_embHasApiKey = embArgs[3].toBool();
    m_embAvailable = embArgs[4].toBool();
    m_embIsDefault = embArgs[5].toBool();

    m_apiKeyInput.clear();

    QDBusMessage gitReply = iface.call("GetPersistenceSettings");
    if (setStatusFromDbusError(gitReply)) {
        return;
    }

    const auto gitArgs = gitReply.arguments();
    if (gitArgs.size() < 4) {
        m_statusText = QStringLiteral("Unexpected GetPersistenceSettings reply");
        Q_EMIT statusTextChanged();
        return;
    }

    m_gitEnabled = gitArgs[0].toBool();
    m_gitRemoteUrl = gitArgs[1].toString();
    m_gitRemoteName = gitArgs[2].toString();
    m_gitPushOnUpdate = gitArgs[3].toBool();

    Q_EMIT connectorChanged();
    Q_EMIT modelChanged();
    Q_EMIT baseUrlChanged();
    Q_EMIT embConnectorChanged();
    Q_EMIT embModelChanged();
    Q_EMIT embBaseUrlChanged();
    Q_EMIT embHasApiKeyChanged();
    Q_EMIT embAvailableChanged();
    Q_EMIT embIsDefaultChanged();
    Q_EMIT hasApiKeyChanged();
    Q_EMIT apiKeyInputChanged();
    Q_EMIT gitEnabledChanged();
    Q_EMIT gitRemoteUrlChanged();
    Q_EMIT gitRemoteNameChanged();
    Q_EMIT gitPushOnUpdateChanged();

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

    QDBusMessage embeddingsReply = iface.call("SetEmbeddingsSettings", m_embConnector, m_embModel, m_embBaseUrl);
    if (setStatusFromDbusError(embeddingsReply)) {
        return;
    }

    QDBusMessage gitSaveReply = iface.call(
        "SetPersistenceSettings",
        m_gitEnabled,
        m_gitRemoteUrl,
        m_gitRemoteName,
        m_gitPushOnUpdate
    );
    if (setStatusFromDbusError(gitSaveReply)) {
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
    applyChatDefaults();
    applySearchDefaults();
    setApiKeyInput(QString());
    m_statusText = QStringLiteral("Applied connector defaults; click Apply to save");
    Q_EMIT statusTextChanged();
}

void DesktopAssistantKcm::applyChatDefaults()
{
    QDBusInterface iface(SERVICE, PATH, IFACE, QDBusConnection::sessionBus());
    const auto llmConnector = normalizeConnector(m_connector);

    ConnectorDefaults defaults;
    QString errorText;
    if (!fetchConnectorDefaults(iface, llmConnector, &defaults, &errorText)) {
        m_statusText = errorText;
        Q_EMIT statusTextChanged();
        return;
    }

    setModel(defaults.llmModel);
    setBaseUrl(defaults.llmBaseUrl);
}

void DesktopAssistantKcm::applySearchDefaults()
{
    QDBusInterface iface(SERVICE, PATH, IFACE, QDBusConnection::sessionBus());
    auto embeddingConnector = normalizeConnector(m_embConnector.isEmpty() ? m_connector : m_embConnector);

    ConnectorDefaults defaults;
    QString errorText;
    if (!fetchConnectorDefaults(iface, embeddingConnector, &defaults, &errorText)) {
        m_statusText = errorText;
        Q_EMIT statusTextChanged();
        return;
    }

    if (!defaults.embeddingsAvailable) {
        embeddingConnector = QStringLiteral("openai");
        if (m_embConnector == QLatin1String("anthropic")) {
            setEmbConnector(embeddingConnector);
        }

        if (!fetchConnectorDefaults(iface, embeddingConnector, &defaults, &errorText)) {
            m_statusText = errorText;
            Q_EMIT statusTextChanged();
            return;
        }
    }

    setEmbModel(defaults.embeddingsModel);
    setEmbBaseUrl(defaults.embeddingsBaseUrl);
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
