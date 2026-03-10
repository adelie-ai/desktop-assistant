#pragma once

#include <KQuickConfigModule>
#include <QStringList>
#include <QVector>

class QDBusMessage;

class DesktopAssistantKcm : public KQuickConfigModule {
    Q_OBJECT
    Q_PROPERTY(QString connector READ connector WRITE setConnector NOTIFY connectorChanged)
    Q_PROPERTY(QString model READ model WRITE setModel NOTIFY modelChanged)
    Q_PROPERTY(QString baseUrl READ baseUrl WRITE setBaseUrl NOTIFY baseUrlChanged)
    Q_PROPERTY(QString embConnector READ embConnector WRITE setEmbConnector NOTIFY embConnectorChanged)
    Q_PROPERTY(QString embModel READ embModel WRITE setEmbModel NOTIFY embModelChanged)
    Q_PROPERTY(QString embBaseUrl READ embBaseUrl WRITE setEmbBaseUrl NOTIFY embBaseUrlChanged)
    Q_PROPERTY(bool embHasApiKey READ embHasApiKey NOTIFY embHasApiKeyChanged)
    Q_PROPERTY(bool embAvailable READ embAvailable NOTIFY embAvailableChanged)
    Q_PROPERTY(bool embIsDefault READ embIsDefault NOTIFY embIsDefaultChanged)
    Q_PROPERTY(QString apiKeyInput READ apiKeyInput WRITE setApiKeyInput NOTIFY apiKeyInputChanged)
    Q_PROPERTY(bool hasApiKey READ hasApiKey NOTIFY hasApiKeyChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusTextChanged)
    Q_PROPERTY(bool gitEnabled READ gitEnabled WRITE setGitEnabled NOTIFY gitEnabledChanged)
    Q_PROPERTY(QString gitRemoteUrl READ gitRemoteUrl WRITE setGitRemoteUrl NOTIFY gitRemoteUrlChanged)
    Q_PROPERTY(QString gitRemoteName READ gitRemoteName WRITE setGitRemoteName NOTIFY gitRemoteNameChanged)
    Q_PROPERTY(bool gitPushOnUpdate READ gitPushOnUpdate WRITE setGitPushOnUpdate NOTIFY gitPushOnUpdateChanged)
    Q_PROPERTY(QString dbUrl READ dbUrl WRITE setDbUrl NOTIFY dbUrlChanged)
    Q_PROPERTY(int dbMaxConnections READ dbMaxConnections WRITE setDbMaxConnections NOTIFY dbMaxConnectionsChanged)
    Q_PROPERTY(QStringList connectionNames READ connectionNames NOTIFY connectionNamesChanged)
    Q_PROPERTY(QString defaultConnectionName READ defaultConnectionName WRITE setDefaultConnectionName NOTIFY defaultConnectionNameChanged)
    Q_PROPERTY(QString selectedConnectionName READ selectedConnectionName WRITE setSelectedConnectionName NOTIFY selectedConnectionNameChanged)
    Q_PROPERTY(QString selectedConnectionTransport READ selectedConnectionTransport NOTIFY selectedConnectionTransportChanged)
    Q_PROPERTY(QString selectedConnectionDbusService READ selectedConnectionDbusService WRITE setSelectedConnectionDbusService NOTIFY selectedConnectionDbusServiceChanged)
    Q_PROPERTY(QString selectedConnectionWsUrl READ selectedConnectionWsUrl WRITE setSelectedConnectionWsUrl NOTIFY selectedConnectionWsUrlChanged)
    Q_PROPERTY(QString selectedConnectionWsSubject READ selectedConnectionWsSubject WRITE setSelectedConnectionWsSubject NOTIFY selectedConnectionWsSubjectChanged)
    Q_PROPERTY(bool selectedConnectionRemovable READ selectedConnectionRemovable NOTIFY selectedConnectionRemovableChanged)
    Q_PROPERTY(bool btDreamingEnabled READ btDreamingEnabled WRITE setBtDreamingEnabled NOTIFY btDreamingEnabledChanged)
    Q_PROPERTY(int btDreamingIntervalSecs READ btDreamingIntervalSecs WRITE setBtDreamingIntervalSecs NOTIFY btDreamingIntervalSecsChanged)
    Q_PROPERTY(bool btHasSeparateLlm READ btHasSeparateLlm NOTIFY btHasSeparateLlmChanged)
    Q_PROPERTY(QString btLlmConnector READ btLlmConnector WRITE setBtLlmConnector NOTIFY btLlmConnectorChanged)
    Q_PROPERTY(QString btLlmModel READ btLlmModel WRITE setBtLlmModel NOTIFY btLlmModelChanged)
    Q_PROPERTY(QString btLlmBaseUrl READ btLlmBaseUrl WRITE setBtLlmBaseUrl NOTIFY btLlmBaseUrlChanged)
    Q_PROPERTY(int hostedToolSearch READ hostedToolSearch WRITE setHostedToolSearch NOTIFY hostedToolSearchChanged)
    Q_PROPERTY(bool hostedToolSearchAvailable READ hostedToolSearchAvailable NOTIFY hostedToolSearchAvailableChanged)
    Q_PROPERTY(bool wsAuthPasswordEnabled READ wsAuthPasswordEnabled WRITE setWsAuthPasswordEnabled NOTIFY wsAuthMethodsChanged)
    Q_PROPERTY(bool wsAuthOidcEnabled READ wsAuthOidcEnabled WRITE setWsAuthOidcEnabled NOTIFY wsAuthMethodsChanged)
    Q_PROPERTY(QString oidcIssuer READ oidcIssuer WRITE setOidcIssuer NOTIFY oidcIssuerChanged)
    Q_PROPERTY(QString oidcAuthEndpoint READ oidcAuthEndpoint WRITE setOidcAuthEndpoint NOTIFY oidcAuthEndpointChanged)
    Q_PROPERTY(QString oidcTokenEndpoint READ oidcTokenEndpoint WRITE setOidcTokenEndpoint NOTIFY oidcTokenEndpointChanged)
    Q_PROPERTY(QString oidcClientId READ oidcClientId WRITE setOidcClientId NOTIFY oidcClientIdChanged)
    Q_PROPERTY(QString oidcScopes READ oidcScopes WRITE setOidcScopes NOTIFY oidcScopesChanged)

public:
    DesktopAssistantKcm(QObject *parent, const KPluginMetaData &metaData, const QVariantList &args);

    QString connector() const;
    void setConnector(const QString &value);

    QString model() const;
    void setModel(const QString &value);

    QString baseUrl() const;
    void setBaseUrl(const QString &value);

    QString embConnector() const;
    void setEmbConnector(const QString &value);

    QString embModel() const;
    void setEmbModel(const QString &value);

    QString embBaseUrl() const;
    void setEmbBaseUrl(const QString &value);

    bool embHasApiKey() const;
    bool embAvailable() const;
    bool embIsDefault() const;

    QString apiKeyInput() const;
    void setApiKeyInput(const QString &value);

    bool hasApiKey() const;
    QString statusText() const;

    bool gitEnabled() const;
    void setGitEnabled(bool value);

    QString gitRemoteUrl() const;
    void setGitRemoteUrl(const QString &value);

    QString gitRemoteName() const;
    void setGitRemoteName(const QString &value);

    bool gitPushOnUpdate() const;
    void setGitPushOnUpdate(bool value);

    QString dbUrl() const;
    void setDbUrl(const QString &value);

    int dbMaxConnections() const;
    void setDbMaxConnections(int value);

    QStringList connectionNames() const;

    QString defaultConnectionName() const;
    void setDefaultConnectionName(const QString &value);

    QString selectedConnectionName() const;
    void setSelectedConnectionName(const QString &value);

    QString selectedConnectionTransport() const;

    QString selectedConnectionDbusService() const;
    void setSelectedConnectionDbusService(const QString &value);

    QString selectedConnectionWsUrl() const;
    void setSelectedConnectionWsUrl(const QString &value);

    QString selectedConnectionWsSubject() const;
    void setSelectedConnectionWsSubject(const QString &value);

    bool selectedConnectionRemovable() const;

    bool btDreamingEnabled() const;
    void setBtDreamingEnabled(bool value);

    int btDreamingIntervalSecs() const;
    void setBtDreamingIntervalSecs(int value);

    bool btHasSeparateLlm() const;

    QString btLlmConnector() const;
    void setBtLlmConnector(const QString &value);

    QString btLlmModel() const;
    void setBtLlmModel(const QString &value);

    QString btLlmBaseUrl() const;
    void setBtLlmBaseUrl(const QString &value);

    int hostedToolSearch() const;
    void setHostedToolSearch(int value);
    bool hostedToolSearchAvailable() const;

    bool wsAuthPasswordEnabled() const;
    void setWsAuthPasswordEnabled(bool value);
    bool wsAuthOidcEnabled() const;
    void setWsAuthOidcEnabled(bool value);

    QString oidcIssuer() const;
    void setOidcIssuer(const QString &value);
    QString oidcAuthEndpoint() const;
    void setOidcAuthEndpoint(const QString &value);
    QString oidcTokenEndpoint() const;
    void setOidcTokenEndpoint(const QString &value);
    QString oidcClientId() const;
    void setOidcClientId(const QString &value);
    QString oidcScopes() const;
    void setOidcScopes(const QString &value);

    Q_INVOKABLE void load() override;
    Q_INVOKABLE void save() override;
    Q_INVOKABLE void defaults() override;
    Q_INVOKABLE void applyChatDefaults();
    Q_INVOKABLE void applySearchDefaults();
    Q_INVOKABLE void applyBackendDefaults();
    Q_INVOKABLE void restartDaemon();
    Q_INVOKABLE void addRemoteConnection(const QString &name);
    Q_INVOKABLE void removeSelectedConnection();

Q_SIGNALS:
    void connectorChanged();
    void modelChanged();
    void baseUrlChanged();
    void embConnectorChanged();
    void embModelChanged();
    void embBaseUrlChanged();
    void embHasApiKeyChanged();
    void embAvailableChanged();
    void embIsDefaultChanged();
    void apiKeyInputChanged();
    void hasApiKeyChanged();
    void statusTextChanged();
    void gitEnabledChanged();
    void gitRemoteUrlChanged();
    void gitRemoteNameChanged();
    void gitPushOnUpdateChanged();
    void dbUrlChanged();
    void dbMaxConnectionsChanged();
    void connectionNamesChanged();
    void defaultConnectionNameChanged();
    void selectedConnectionNameChanged();
    void selectedConnectionTransportChanged();
    void selectedConnectionDbusServiceChanged();
    void selectedConnectionWsUrlChanged();
    void selectedConnectionWsSubjectChanged();
    void selectedConnectionRemovableChanged();
    void btDreamingEnabledChanged();
    void btDreamingIntervalSecsChanged();
    void btHasSeparateLlmChanged();
    void btLlmConnectorChanged();
    void btLlmModelChanged();
    void btLlmBaseUrlChanged();
    void hostedToolSearchChanged();
    void hostedToolSearchAvailableChanged();
    void wsAuthMethodsChanged();
    void oidcIssuerChanged();
    void oidcAuthEndpointChanged();
    void oidcTokenEndpointChanged();
    void oidcClientIdChanged();
    void oidcScopesChanged();

private:
    struct ConnectionProfile {
        QString name;
        QString transport;
        QString dbusService;
        QString wsUrl;
        QString wsSubject;
    };

    bool setStatusFromDbusError(const QDBusMessage &message);
    int connectionIndexByName(const QString &name) const;
    int selectedConnectionIndex() const;
    void loadWidgetConnectionSettings();
    bool saveWidgetConnectionSettings();
    void setSelectedConnectionByIndex(int index);
    void emitConnectionSelectionChanged();

    QString m_connector;
    QString m_model;
    QString m_baseUrl;
    QString m_embConnector;
    QString m_embModel;
    QString m_embBaseUrl;
    bool m_embHasApiKey = false;
    bool m_embAvailable = true;
    bool m_embIsDefault = true;
    QString m_apiKeyInput;
    bool m_hasApiKey = false;
    QString m_statusText;
    bool m_gitEnabled = false;
    QString m_gitRemoteUrl;
    QString m_gitRemoteName = QStringLiteral("origin");
    bool m_gitPushOnUpdate = true;
    QString m_dbUrl;
    int m_dbMaxConnections = 5;
    QVector<ConnectionProfile> m_connections;
    QString m_defaultConnectionName = QStringLiteral("local");
    QString m_selectedConnectionName = QStringLiteral("local");
    bool m_btDreamingEnabled = false;
    int m_btDreamingIntervalSecs = 3600;
    bool m_btHasSeparateLlm = false;
    QString m_btLlmConnector;
    QString m_btLlmModel;
    QString m_btLlmBaseUrl;
    int m_hostedToolSearch = -1; // -1 = connector default, 0 = off, 1 = on
    bool m_hostedToolSearchAvailable = true;
    QStringList m_wsAuthMethods = {QStringLiteral("password")};
    QString m_oidcIssuer;
    QString m_oidcAuthEndpoint;
    QString m_oidcTokenEndpoint;
    QString m_oidcClientId;
    QString m_oidcScopes = QStringLiteral("openid profile email");
};
