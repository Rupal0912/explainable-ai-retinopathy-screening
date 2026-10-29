# Security and Privacy Notes

## Scope

This is a student portfolio prototype. It has not been designed, reviewed, or validated as a production healthcare system. Do not use real patient data.

## Secrets

- Store `GROQ_API_KEY` and `MONGODB_URI` in a local `.env`.
- Commit `.env.example` with blank values only.
- Ensure `.env`, credential files, tokens, and local configuration are ignored by Git.
- If a secret is committed, revoke/rotate it immediately; deleting it in a later commit is not sufficient.
- Do not paste keys into prompts, screenshots, issues, or logs.

## Data minimization

- Use demo patient IDs only.
- Persist the SHA-256 hash of image bytes, not the image.
- Do not log image contents or sensitive user inputs.
- Do not commit APTOS images, derived image caches, private uploads, or screening records.
- Review demo recordings for credentials, private paths, and sensitive content.

## Database

- Use a dedicated development database and least-privilege credentials.
- Restrict network access where possible.
- Do not expose connection strings in exceptions or UI messages.
- Decide a retention/deletion policy even for demo records.
- Treat demo IDs as pseudonymous identifiers, not as a security boundary.

## External LLM and retrieved documents

- Send only the minimum required text to the LLM provider.
- Do not send uploaded images or patient identifiers to the LLM.
- Treat retrieved PDF text as untrusted content, not as instructions that can override the system prompt.
- The LLM must summarize only retrieved excerpts and must not invent missing clinical facts.
- Verify citations against source/page metadata before displaying them.

## Reporting a vulnerability

For a private portfolio repository, report issues to the repository owner privately. Do not publish secrets, patient data, or exploit details in a public issue. Rotate any exposed credentials immediately.

## Pre-push checklist

- [ ] `.env` and credentials are not staged.
- [ ] No real identifiers or patient images are present.
- [ ] No dataset files or private generated artifacts are staged.
- [ ] Git history has been checked for accidentally committed secrets.
- [ ] Error messages do not leak secrets.
- [ ] Dependencies and model identifiers are reviewed.
