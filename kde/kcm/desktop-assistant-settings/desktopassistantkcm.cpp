#include "desktopassistantkcm.h"

#include <QDBusInterface>
#include <QDBusMessage>
#include <QDBusReply>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>

#include <KPluginFactory>

namespace {
constexpr auto SERVICE = "org.desktopAssistant";
constexpr auto PATH = "/org/desktopAssistant/Settings";
constexpr auto IFACE = "org.desktopAssistant.Settings";
constexpr auto DEFAULT_CONNECTION_NAME = "local";
constexpr auto DEFAULT_WS_URL = "ws://127.0.0.1:11339/ws";
constexpr auto DEFAULT_WS_SUBJECT = "desktop-widget";

QString normalizeConnector(const QString &connector)
{
    const auto normalized = connector.trimmed().toLower();
    return normalized.isEmpty() ? QStringLiteral("openai") : normalized;
}

QString widgetSettingsPath()
{
    const auto configHome = QStandardPaths::writableLocation(QStandardPaths::ConfigLocation);
    return QDir(configHome).filePath(QStringLiteral("desktop-assistant/widget_settings.json"));
}

QString normalizeConnectionName(const QString &name)
{
    return name.trimmed();
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

QStringList DesktopAssistantKcm::connectionNames() const
{
    QStringList names;
    names.reserve(m_connections.size());
    for (const auto &connection : m_connections) {
        names.push_back(connection.name);
    }
    return names;
}

QString DesktopAssistantKcm::defaultConnectionName() const
{
    return m_defaultConnectionName;
}

void DesktopAssistantKcm::setDefaultConnectionName(const QString &value)
{
    const auto normalized = normalizeConnectionName(value);
    if (normalized.isEmpty() || connectionIndexByName(normalized) < 0) {
        return;
    }
    if (m_defaultConnectionName == normalized) {
        return;
    }

    m_defaultConnectionName = normalized;
    Q_EMIT defaultConnectionNameChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::selectedConnectionName() const
{
    return m_selectedConnectionName;
}

void DesktopAssistantKcm::setSelectedConnectionName(const QString &value)
{
    const auto normalized = normalizeConnectionName(value);
    const auto index = connectionIndexByName(normalized);
    if (index < 0) {
        return;
    }
    setSelectedConnectionByIndex(index);
}

QString DesktopAssistantKcm::selectedConnectionTransport() const
{
    const auto index = selectedConnectionIndex();
    if (index < 0) {
        return QStringLiteral("dbus");
    }
    return m_connections[index].transport;
}

QString DesktopAssistantKcm::selectedConnectionDbusService() const
{
    const auto index = selectedConnectionIndex();
    if (index < 0) {
        return QString::fromUtf8(SERVICE);
    }
    return m_connections[index].dbusService;
}

void DesktopAssistantKcm::setSelectedConnectionDbusService(const QString &value)
{
    const auto index = selectedConnectionIndex();
    if (index < 0 || m_connections[index].transport != QLatin1String("dbus")) {
        return;
    }

    const auto normalized = value.trimmed().isEmpty() ? QString::fromUtf8(SERVICE) : value.trimmed();
    if (m_connections[index].dbusService == normalized) {
        return;
    }

    m_connections[index].dbusService = normalized;
    Q_EMIT selectedConnectionDbusServiceChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::selectedConnectionWsUrl() const
{
    const auto index = selectedConnectionIndex();
    if (index < 0) {
        return QString::fromUtf8(DEFAULT_WS_URL);
    }
    return m_connections[index].wsUrl;
}

void DesktopAssistantKcm::setSelectedConnectionWsUrl(const QString &value)
{
    const auto index = selectedConnectionIndex();
    if (index < 0 || m_connections[index].transport != QLatin1String("ws")) {
        return;
    }

    const auto normalized = value.trimmed().isEmpty() ? QString::fromUtf8(DEFAULT_WS_URL) : value.trimmed();
    if (m_connections[index].wsUrl == normalized) {
        return;
    }

    m_connections[index].wsUrl = normalized;
    Q_EMIT selectedConnectionWsUrlChanged();
    setNeedsSave(true);
}

QString DesktopAssistantKcm::selectedConnectionWsSubject() const
{
    const auto index = selectedConnectionIndex();
    if (index < 0) {
        return QString::fromUtf8(DEFAULT_WS_SUBJECT);
    }
    return m_connections[index].wsSubject;
}

void DesktopAssistantKcm::setSelectedConnectionWsSubject(const QString &value)
{
    const auto index = selectedConnectionIndex();
    if (index < 0 || m_connections[index].transport != QLatin1String("ws")) {
        return;
    }

    const auto normalized = value.trimmed().isEmpty() ? QString::fromUtf8(DEFAULT_WS_SUBJECT) : value.trimmed();
    if (m_connections[index].wsSubject == normalized) {
        return;
    }

    m_connections[index].wsSubject = normalized;
    Q_EMIT selectedConnectionWsSubjectChanged();
    setNeedsSave(true);
}

bool DesktopAssistantKcm::selectedConnectionRemovable() const
{
    return selectedConnectionName() != QLatin1String(DEFAULT_CONNECTION_NAME);
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

    loadWidgetConnectionSettings();

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
    Q_EMIT connectionNamesChanged();
    Q_EMIT defaultConnectionNameChanged();
    emitConnectionSelectionChanged();

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

    if (!saveWidgetConnectionSettings()) {
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

void DesktopAssistantKcm::addRemoteConnection(const QString &name)
{
    const auto normalized = normalizeConnectionName(name);
    if (normalized.isEmpty()) {
        m_statusText = QStringLiteral("Connection name is required");
        Q_EMIT statusTextChanged();
        return;
    }

    static const QRegularExpression validName(QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"));
    if (!validName.match(normalized).hasMatch()) {
        m_statusText = QStringLiteral("Connection name may include letters, numbers, dot, underscore, and dash");
        Q_EMIT statusTextChanged();
        return;
    }

    if (normalized == QLatin1String(DEFAULT_CONNECTION_NAME)) {
        m_statusText = QStringLiteral("'local' is reserved for the local D-Bus connection");
        Q_EMIT statusTextChanged();
        return;
    }

    const auto existing = connectionIndexByName(normalized);
    if (existing >= 0) {
        setSelectedConnectionByIndex(existing);
        m_statusText = QStringLiteral("Connection already exists");
        Q_EMIT statusTextChanged();
        return;
    }

    ConnectionProfile connection;
    connection.name = normalized;
    connection.transport = QStringLiteral("ws");
    connection.wsUrl = QString::fromUtf8(DEFAULT_WS_URL);
    connection.wsSubject = QString::fromUtf8(DEFAULT_WS_SUBJECT);
    m_connections.push_back(connection);

    Q_EMIT connectionNamesChanged();
    setSelectedConnectionByIndex(m_connections.size() - 1);
    m_statusText = QStringLiteral("Added connection '%1'").arg(normalized);
    Q_EMIT statusTextChanged();
    setNeedsSave(true);
}

void DesktopAssistantKcm::removeSelectedConnection()
{
    const auto index = selectedConnectionIndex();
    if (index < 0) {
        return;
    }

    const auto name = m_connections[index].name;
    if (name == QLatin1String(DEFAULT_CONNECTION_NAME)) {
        m_statusText = QStringLiteral("Local connection cannot be removed");
        Q_EMIT statusTextChanged();
        return;
    }

    m_connections.removeAt(index);
    if (m_defaultConnectionName == name) {
        m_defaultConnectionName = QStringLiteral(DEFAULT_CONNECTION_NAME);
        Q_EMIT defaultConnectionNameChanged();
    }

    Q_EMIT connectionNamesChanged();
    ensureLocalConnection();
    setSelectedConnectionName(m_defaultConnectionName);
    m_statusText = QStringLiteral("Removed connection '%1'").arg(name);
    Q_EMIT statusTextChanged();
    setNeedsSave(true);
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

int DesktopAssistantKcm::connectionIndexByName(const QString &name) const
{
    const auto normalized = normalizeConnectionName(name);
    for (qsizetype i = 0; i < m_connections.size(); ++i) {
        if (m_connections[i].name == normalized) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int DesktopAssistantKcm::selectedConnectionIndex() const
{
    return connectionIndexByName(m_selectedConnectionName);
}

void DesktopAssistantKcm::loadWidgetConnectionSettings()
{
    m_connections.clear();
    m_defaultConnectionName = QStringLiteral(DEFAULT_CONNECTION_NAME);
    m_selectedConnectionName = QStringLiteral(DEFAULT_CONNECTION_NAME);

    QString localDbusService = QString::fromUtf8(SERVICE);
    QString legacyTransport;
    QString legacyWsUrl;
    QString legacyWsSubject;
    QString configuredDefaultConnection;

    QFile file(widgetSettingsPath());
    if (file.exists() && file.open(QIODevice::ReadOnly)) {
        QJsonParseError parseError;
        const auto doc = QJsonDocument::fromJson(file.readAll(), &parseError);
        file.close();

        if (parseError.error == QJsonParseError::NoError && doc.isObject()) {
            const auto root = doc.object();

            localDbusService = root.value(QStringLiteral("dbus_service")).toString().trimmed();
            if (localDbusService.isEmpty()) {
                localDbusService = QString::fromUtf8(SERVICE);
            }

            legacyTransport = root.value(QStringLiteral("transport")).toString().trimmed().toLower();
            legacyWsUrl = root.value(QStringLiteral("ws_url")).toString().trimmed();
            legacyWsSubject = root.value(QStringLiteral("ws_subject")).toString().trimmed();
            configuredDefaultConnection = normalizeConnectionName(
                root.value(QStringLiteral("default_connection")).toString()
            );

            const auto rawConnections = root.value(QStringLiteral("connections"));
            if (rawConnections.isArray()) {
                const auto array = rawConnections.toArray();
                for (const auto &item : array) {
                    if (!item.isObject()) {
                        continue;
                    }
                    const auto obj = item.toObject();
                    ConnectionProfile connection;
                    connection.name = normalizeConnectionName(obj.value(QStringLiteral("name")).toString());
                    if (connection.name.isEmpty() || connectionIndexByName(connection.name) >= 0) {
                        continue;
                    }

                    connection.transport = obj.value(QStringLiteral("transport")).toString().trimmed().toLower();
                    if (connection.name == QLatin1String(DEFAULT_CONNECTION_NAME)) {
                        connection.transport = QStringLiteral("dbus");
                    } else if (connection.transport != QLatin1String("ws")) {
                        connection.transport = QStringLiteral("ws");
                    }

                    connection.dbusService = obj.value(QStringLiteral("dbus_service")).toString().trimmed();
                    if (connection.transport == QLatin1String("dbus") && connection.dbusService.isEmpty()) {
                        connection.dbusService = QString::fromUtf8(SERVICE);
                    }

                    connection.wsUrl = obj.value(QStringLiteral("ws_url")).toString().trimmed();
                    connection.wsSubject = obj.value(QStringLiteral("ws_subject")).toString().trimmed();
                    if (connection.transport == QLatin1String("ws")) {
                        if (connection.wsUrl.isEmpty()) {
                            connection.wsUrl = QString::fromUtf8(DEFAULT_WS_URL);
                        }
                        if (connection.wsSubject.isEmpty()) {
                            connection.wsSubject = QString::fromUtf8(DEFAULT_WS_SUBJECT);
                        }
                    }

                    m_connections.push_back(connection);
                }
            }
        }
    }

    if (m_connections.isEmpty()) {
        ConnectionProfile localConnection;
        localConnection.name = QStringLiteral(DEFAULT_CONNECTION_NAME);
        localConnection.transport = QStringLiteral("dbus");
        localConnection.dbusService = localDbusService;
        m_connections.push_back(localConnection);

        const auto useLegacyWs = legacyTransport == QLatin1String("ws") || !legacyWsUrl.isEmpty();
        if (useLegacyWs) {
            ConnectionProfile legacyConnection;
            legacyConnection.name = QStringLiteral("legacy-ws");
            legacyConnection.transport = QStringLiteral("ws");
            legacyConnection.wsUrl = legacyWsUrl.isEmpty() ? QString::fromUtf8(DEFAULT_WS_URL) : legacyWsUrl;
            legacyConnection.wsSubject = legacyWsSubject.isEmpty() ? QString::fromUtf8(DEFAULT_WS_SUBJECT) : legacyWsSubject;
            m_connections.push_back(legacyConnection);
            m_defaultConnectionName = QStringLiteral("legacy-ws");
        }

        if (legacyTransport == QLatin1String("dbus")) {
            m_defaultConnectionName = QStringLiteral(DEFAULT_CONNECTION_NAME);
        }
    }

    ensureLocalConnection();

    if (!configuredDefaultConnection.isEmpty() && connectionIndexByName(configuredDefaultConnection) >= 0) {
        m_defaultConnectionName = configuredDefaultConnection;
    }

    if (connectionIndexByName(m_defaultConnectionName) < 0) {
        m_defaultConnectionName = QStringLiteral(DEFAULT_CONNECTION_NAME);
    }
    m_selectedConnectionName = m_defaultConnectionName;
}

bool DesktopAssistantKcm::saveWidgetConnectionSettings()
{
    ensureLocalConnection();
    if (connectionIndexByName(m_defaultConnectionName) < 0) {
        m_defaultConnectionName = QStringLiteral(DEFAULT_CONNECTION_NAME);
    }

    QJsonObject root;
    QFile existing(widgetSettingsPath());
    if (existing.exists() && existing.open(QIODevice::ReadOnly)) {
        const auto existingDoc = QJsonDocument::fromJson(existing.readAll());
        if (existingDoc.isObject()) {
            root = existingDoc.object();
        }
        existing.close();
    }

    QJsonArray connections;
    for (const auto &connection : m_connections) {
        QJsonObject item;
        item.insert(QStringLiteral("name"), connection.name);
        item.insert(QStringLiteral("transport"), connection.transport);
        if (connection.transport == QLatin1String("dbus")) {
            item.insert(QStringLiteral("dbus_service"), connection.dbusService.isEmpty() ? QString::fromUtf8(SERVICE) : connection.dbusService);
        } else {
            item.insert(QStringLiteral("ws_url"), connection.wsUrl.isEmpty() ? QString::fromUtf8(DEFAULT_WS_URL) : connection.wsUrl);
            item.insert(QStringLiteral("ws_subject"), connection.wsSubject.isEmpty() ? QString::fromUtf8(DEFAULT_WS_SUBJECT) : connection.wsSubject);
        }
        connections.push_back(item);
    }

    root.insert(QStringLiteral("connections"), connections);
    root.insert(QStringLiteral("default_connection"), m_defaultConnectionName);

    const auto localIndex = connectionIndexByName(QStringLiteral(DEFAULT_CONNECTION_NAME));
    if (localIndex >= 0) {
        root.insert(
            QStringLiteral("dbus_service"),
            m_connections[localIndex].dbusService.isEmpty() ? QString::fromUtf8(SERVICE) : m_connections[localIndex].dbusService
        );
    }

    const auto defaultIndex = connectionIndexByName(m_defaultConnectionName);
    if (defaultIndex >= 0 && m_connections[defaultIndex].transport == QLatin1String("ws")) {
        root.insert(QStringLiteral("transport"), QStringLiteral("ws"));
        root.insert(QStringLiteral("ws_url"), m_connections[defaultIndex].wsUrl);
        root.insert(QStringLiteral("ws_subject"), m_connections[defaultIndex].wsSubject);
    } else {
        root.insert(QStringLiteral("transport"), QStringLiteral("dbus"));
    }

    QFile file(widgetSettingsPath());
    const auto fileInfo = QFileInfo(file);
    QDir dir;
    if (!dir.mkpath(fileInfo.absolutePath())) {
        m_statusText = QStringLiteral("Unable to create widget settings directory");
        Q_EMIT statusTextChanged();
        return false;
    }

    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        m_statusText = QStringLiteral("Unable to write widget settings file");
        Q_EMIT statusTextChanged();
        return false;
    }

    const QJsonDocument doc(root);
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

void DesktopAssistantKcm::ensureLocalConnection()
{
    const auto index = connectionIndexByName(QStringLiteral(DEFAULT_CONNECTION_NAME));
    if (index < 0) {
        ConnectionProfile localConnection;
        localConnection.name = QStringLiteral(DEFAULT_CONNECTION_NAME);
        localConnection.transport = QStringLiteral("dbus");
        localConnection.dbusService = QString::fromUtf8(SERVICE);
        m_connections.prepend(localConnection);
    } else {
        m_connections[index].name = QStringLiteral(DEFAULT_CONNECTION_NAME);
        m_connections[index].transport = QStringLiteral("dbus");
        if (m_connections[index].dbusService.trimmed().isEmpty()) {
            m_connections[index].dbusService = QString::fromUtf8(SERVICE);
        }
        if (index != 0) {
            const auto localConnection = m_connections[index];
            m_connections.removeAt(index);
            m_connections.prepend(localConnection);
        }
    }
}

void DesktopAssistantKcm::setSelectedConnectionByIndex(int index)
{
    if (index < 0 || index >= m_connections.size()) {
        return;
    }

    const auto nextName = m_connections[index].name;
    if (m_selectedConnectionName == nextName) {
        return;
    }

    m_selectedConnectionName = nextName;
    Q_EMIT selectedConnectionNameChanged();
    emitConnectionSelectionChanged();
}

void DesktopAssistantKcm::emitConnectionSelectionChanged()
{
    Q_EMIT selectedConnectionTransportChanged();
    Q_EMIT selectedConnectionDbusServiceChanged();
    Q_EMIT selectedConnectionWsUrlChanged();
    Q_EMIT selectedConnectionWsSubjectChanged();
    Q_EMIT selectedConnectionRemovableChanged();
}

#include "desktopassistantkcm.moc"
