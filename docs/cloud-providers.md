# Cloud Providers (Quick Reference)

This page is a fast onboarding guide for the cloud connectors currently used in this repo.

> ⚠️ **Important:** Provider privacy policies, console URLs, and credential setup flows can change over time. Always re-check the linked official docs before production use.

Scope:
- OpenAI (`openai`)
- Anthropic (`anthropic`)
- AWS Bedrock (`bedrock` / `aws-bedrock`)
- OpenRouter (`openrouter`)
- Azure OpenAI (`azure`)
- Google Vertex AI / Gemini (`google`)

## Before You Start

- API privacy behavior is usually different from consumer chat apps.
- Data handling details can change over time.
- Always verify your account/org settings and legal requirements.

---

## OpenAI

**Privacy brief**
- API usage has separate policy docs from ChatGPT consumer usage.
- Retention and training behavior depend on product + account settings.
- Check your org settings and current API policy docs before production use.

**Policy links**
- API data usage policy: https://openai.com/policies/api-data-usage-policies/
- Privacy policy: https://openai.com/policies/privacy-policy/

**Console / key setup**
- API keys: https://platform.openai.com/api-keys
- Platform home: https://platform.openai.com/

---

## Anthropic

**Privacy brief**
- API terms/policies are separate from end-user chat experiences.
- Data handling and retention are policy-driven and can vary by plan.
- Confirm latest policy + account controls before handling sensitive data.

**Policy links**
- Privacy policy: https://www.anthropic.com/privacy
- Trust & security overview: https://trust.anthropic.com/

**Console / key setup**
- API keys: https://console.anthropic.com/settings/keys
- Console home: https://console.anthropic.com/

---

## AWS Bedrock

**Privacy brief**
- Bedrock uses AWS identity/permissions (IAM), not a single vendor API key.
- Credential resolution follows the standard AWS SDK credential provider chain.
- Privacy/compliance posture is documented through AWS service + privacy docs.

**Policy links**
- AWS Bedrock FAQ (security/privacy section): https://aws.amazon.com/bedrock/faqs/
- AWS privacy notice: https://aws.amazon.com/privacy/

**Console / credential setup**
- Bedrock console: https://console.aws.amazon.com/bedrock/
- IAM console: https://console.aws.amazon.com/iam/
- AWS CLI config guide (`aws configure` / profiles): https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html

---

## OpenRouter

**Privacy brief**
- Aggregator that routes one API to many upstream vendors; data handling depends on
  the upstream provider your request routes to.
- Per-model privacy/training behavior is surfaced in the model catalog; some models
  can be excluded from providers that log/train.

**Policy links**
- Privacy policy: https://openrouter.ai/privacy
- Terms: https://openrouter.ai/terms

**Console / key setup**
- API keys: https://openrouter.ai/keys
- Model catalog (for `vendor/model` ids): https://openrouter.ai/models

---

## Azure OpenAI

**Privacy brief**
- Runs in your Azure tenant/subscription with Azure identity (Entra ID) and RBAC;
  data-handling posture follows your Azure resource's region and settings.
- You provision model **deployments** in the Azure portal; the deployment name is
  what you reference as the model.

**Policy links**
- Azure OpenAI data, privacy, and security: https://learn.microsoft.com/azure/ai-foundry/openai/how-to/create-resource
- Azure privacy: https://privacy.microsoft.com/

**Console / credential setup**
- Azure AI Foundry / OpenAI portal: https://ai.azure.com/
- Create a resource + deployment, then get the resource endpoint and key (or use
  Entra ID / managed identity).
- Uses the v1 GA API (`{resource}/openai/v1/...`); no `api-version` needed.

---

## Google Vertex AI / Gemini

**Privacy brief**
- Vertex AI runs in your Google Cloud project with IAM/service-account auth (the GCP
  analogue of Bedrock's AWS chain); posture follows your project + region settings.
- The simpler Gemini API (AI Studio) uses a single API key and has separate terms.

**Policy links**
- Vertex AI data governance: https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance
- Google Cloud privacy: https://cloud.google.com/terms/cloud-privacy-notice

**Console / credential setup**
- Google Cloud console (enable the Vertex AI API): https://console.cloud.google.com/vertex-ai
- Service accounts / ADC: `gcloud auth application-default login`, or set
  `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON path.
- Gemini API keys (AI Studio mode): https://aistudio.google.com/apikey

---

## How this maps to Adelie config

Simple env-var providers:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Bedrock: AWS SDK chain (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, profiles, SSO, role credentials)
- Optional Bedrock shortcut in this project: `AWS_BEDROCK_API_KEY=ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]`
- OpenRouter: `OPENROUTER_API_KEY` (optionally `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`)

Azure and Vertex carry multi-field config that env vars alone cannot express, so
configure them as named connections in `daemon.toml`. Minimal examples:

```toml
# Azure OpenAI (v1 GA API). `model` is your deployment name; `base_url` is the
# resource endpoint. Set AZURE_OPENAI_API_KEY (or use auth_mode = "entra").
[connections.azure-prod]
type = "azure"
base_url = "https://YOUR-RESOURCE.openai.azure.com"
model = "my-gpt5-deployment"   # the deployment you created in the portal
# api_surface = "v1"           # default; "classic" for the legacy deployment-in-URL API
# auth_mode = "api_key"        # default; "entra" for Entra ID / managed identity

# Google Vertex AI (Gemini). Auth via ADC or GOOGLE_APPLICATION_CREDENTIALS.
[connections.vertex-prod]
type = "google"
project = "my-gcp-project"
location = "us-central1"
model = "gemini-2.5-pro"
# auth_mode = "vertex"         # default; "api_key" uses the Gemini API + GOOGLE_API_KEY

# OpenRouter (also works purely from OPENROUTER_API_KEY).
[connections.openrouter]
type = "openrouter"
model = "anthropic/claude-sonnet-4-6"
```

> Note: `type = "azure"` derives its env-var names from the connector key, so the
> connection defaults its key lookup to `AZURE_OPENAI_API_KEY` (Azure's own
> convention). Vertex reads `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` (with
> `GOOGLE_PROJECT` / `GOOGLE_LOCATION` as fallbacks) when the fields are omitted.
