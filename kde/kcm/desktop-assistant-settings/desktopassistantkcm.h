#pragma once

#include <KQuickConfigModule>

class QDBusMessage;

class DesktopAssistantKcm : public KQuickConfigModule {
    Q_OBJECT
    Q_PROPERTY(QString connector READ connector WRITE setConnector NOTIFY connectorChanged)
    Q_PROPERTY(QString model READ model WRITE setModel NOTIFY modelChanged)
    Q_PROPERTY(QString baseUrl READ baseUrl WRITE setBaseUrl NOTIFY baseUrlChanged)
    Q_PROPERTY(QString apiKeyInput READ apiKeyInput WRITE setApiKeyInput NOTIFY apiKeyInputChanged)
    Q_PROPERTY(bool hasApiKey READ hasApiKey NOTIFY hasApiKeyChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusTextChanged)

public:
    DesktopAssistantKcm(QObject *parent, const KPluginMetaData &metaData, const QVariantList &args);

    QString connector() const;
    void setConnector(const QString &value);

    QString model() const;
    void setModel(const QString &value);

    QString baseUrl() const;
    void setBaseUrl(const QString &value);

    QString apiKeyInput() const;
    void setApiKeyInput(const QString &value);

    bool hasApiKey() const;
    QString statusText() const;

    Q_INVOKABLE void load() override;
    Q_INVOKABLE void save() override;
    Q_INVOKABLE void defaults() override;
    Q_INVOKABLE void restartDaemon();

Q_SIGNALS:
    void connectorChanged();
    void modelChanged();
    void baseUrlChanged();
    void apiKeyInputChanged();
    void hasApiKeyChanged();
    void statusTextChanged();

private:
    bool setStatusFromDbusError(const QDBusMessage &message);

    QString m_connector;
    QString m_model;
    QString m_baseUrl;
    QString m_apiKeyInput;
    bool m_hasApiKey = false;
    QString m_statusText;
};
