# Cloud Providers (Quick Reference)

This page is a fast onboarding guide for the cloud connectors currently used in this repo.

> ⚠️ **Important:** Provider privacy policies, console URLs, and credential setup flows can change over time. Always re-check the linked official docs before production use.

Scope:
- OpenAI (`openai`)
- Anthropic (`anthropic`)
- AWS Bedrock (`bedrock` / `aws-bedrock`)

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

## How this maps to Adelie config

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Bedrock: AWS SDK chain (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, profiles, SSO, role credentials)
- Optional Bedrock shortcut in this project: `AWS_BEDROCK_API_KEY=ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]`
