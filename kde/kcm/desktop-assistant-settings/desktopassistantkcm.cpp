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

QString defaultModelForConnector(const QString &connector)
{
    if (connector == QLatin1String("ollama")) {
        return QStringLiteral("qwen3:0.6b");
    }
    if (connector == QLatin1String("anthropic")) {
        return QStringLiteral("claude-sonnet-4-5-20250929");
    }
    return QStringLiteral("gpt-5.2");
}

QString defaultEmbeddingModelForConnector(const QString &connector)
{
    if (connector == QLatin1String("ollama")) {
        return QStringLiteral("nomic-embed-text");
    }
    return QStringLiteral("text-embedding-3-small");
}

QString defaultBaseUrlForConnector(const QString &connector)
{
    if (connector == QLatin1String("ollama")) {
        return QStringLiteral("http://localhost:11434");
    }
    if (connector == QLatin1String("anthropic")) {
        return QStringLiteral("https://api.anthropic.com");
    }
    return QStringLiteral("https://api.openai.com/v1");
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
    const auto llmConnector = normalizeConnector(m_connector);
    setModel(defaultModelForConnector(llmConnector));
    setBaseUrl(defaultBaseUrlForConnector(llmConnector));

    auto embeddingConnector = normalizeConnector(m_embConnector.isEmpty() ? llmConnector : m_embConnector);
    if (embeddingConnector == QLatin1String("anthropic")) {
        embeddingConnector = QStringLiteral("openai");
        setEmbConnector(embeddingConnector);
    }

    setEmbModel(defaultEmbeddingModelForConnector(embeddingConnector));
    setEmbBaseUrl(defaultBaseUrlForConnector(embeddingConnector));
    setApiKeyInput(QString());
    m_statusText = QStringLiteral("Applied connector defaults; click Apply to save");
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
