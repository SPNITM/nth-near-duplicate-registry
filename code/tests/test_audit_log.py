import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nth_registry.audit_log import (
    GENESIS_HASH,
    DecisionRecord,
    append_record,
    canonical_json,
    record_body,
    record_hash,
    verify_chain,
)


def example_record(request_id: str) -> DecisionRecord:
    return DecisionRecord(
        request_id=request_id,
        probe_sha256="1" * 64,
        candidate_commitments=("0xabc", "0xdef"),
        nth_distances=(3, 19),
        embedding_distances=(0.11, 0.72),
        matched_commitment="0xabc",
        nth_version="nth-v2-exclude-center",
        nth_radius=16,
        embedding_model_sha256="2" * 64,
        embedding_threshold=0.2,
        decision="match",
        policy_sha256="3" * 64,
        timestamp_utc="2026-08-31T00:00:00+00:00",
    )


class AuditLogTests(unittest.TestCase):
    def test_hash_formula_uses_previous_digest_bytes(self):
        body = record_body(example_record("req-1").payload())
        expected = hashlib.sha256(bytes.fromhex(GENESIS_HASH) + canonical_json(body)).hexdigest()
        self.assertEqual(record_hash(GENESIS_HASH, body), expected)

    def test_two_record_chain_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            first = append_record(path, example_record("req-1"))
            second = append_record(path, example_record("req-2"))
            self.assertNotEqual(first, second)
            self.assertEqual(verify_chain(path), (True, 2, second))

            rows = path.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(rows[0])
            tampered["decision"] = "reject"
            rows[0] = json.dumps(tampered, sort_keys=True)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.assertFalse(verify_chain(path)[0])


if __name__ == "__main__":
    unittest.main()
