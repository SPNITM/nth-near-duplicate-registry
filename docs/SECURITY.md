# Security and auditability statement

The Ethereum contract is an anchoring mechanism, not an on-chain perceptual
oracle. It establishes the time, registrant, content-identifier hash, NTH code,
model version, decision-policy version, and audit-log head asserted by a caller.
It cannot independently prove that an embedding score was computed correctly.

Off-chain decision records form a SHA-256 hash chain. A changed or deleted
historical record breaks verification against the most recently anchored head.
This makes tampering detectable after anchoring; it does not prevent tampering
before the first anchor. Production deployments should sign records, anchor
heads at a documented cadence, replicate the log, restrict writer credentials,
and maintain a human dispute/appeal procedure.

Threats considered include stolen signing keys, malicious registrants,
front-running, replay across networks, log deletion, model replacement,
threshold drift, adversarial image perturbations, denial of service, and loss of
off-chain data. The contract avoids administrative mutation of registrations,
stores no private key, and binds commitments to `msg.sender`. Network and
application-level controls remain deployment responsibilities.

