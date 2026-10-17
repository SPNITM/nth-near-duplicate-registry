#!/usr/bin/env python
"""Compile and benchmark ImageRegistryV2 on a deterministic local EVM."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pandas as pd
import solcx
from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

PROJECT = Path(__file__).resolve().parents[2]
SOLC_VERSION = "0.8.24"


def word(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()


def main() -> None:
    if SOLC_VERSION not in {str(version) for version in solcx.get_installed_solc_versions()}:
        solcx.install_solc(SOLC_VERSION)
    solcx.set_solc_version(SOLC_VERSION)
    source_path = PROJECT / "contracts" / "ImageRegistryV2.sol"
    compiled = solcx.compile_source(
        source_path.read_text(encoding="utf-8"),
        output_values=["abi", "bin"],
        optimize=True,
        optimize_runs=200,
    )
    _, artifact = next(iter(compiled.items()))
    backend = PyEVMBackend()
    tester = EthereumTester(backend=backend)
    web3 = Web3(EthereumTesterProvider(tester))
    owner = web3.eth.accounts[0]
    outsider = web3.eth.accounts[1]
    factory = web3.eth.contract(abi=artifact["abi"], bytecode=artifact["bin"])
    deployment = web3.eth.wait_for_transaction_receipt(factory.constructor().transact({"from": owner}))
    contract = web3.eth.contract(address=deployment.contractAddress, abi=artifact["abi"])

    checkpoints = {0, 1, 10, 100, 1000}
    rows = []
    latest_commitment = None
    for index in range(1001):
        cid_hash = word(f"cid-{index}")
        nth_hash = word(f"nth-{index}")
        model_version = word("shufflenet-v2-x1-imagenet1k-v1")
        policy_hash = word("nth-v2-r16-embedding-calibrated-fpr-0.001")
        audit_head = word(f"audit-{index}")
        function = contract.functions.registerImage(
            cid_hash, nth_hash, model_version, policy_hash, audit_head
        )
        if index in checkpoints:
            estimated = function.estimate_gas({"from": owner})
            receipt = web3.eth.wait_for_transaction_receipt(function.transact({"from": owner}))
            rows.append({
                "operation": "registerImage",
                "registry_size_before": index,
                "estimated_gas": int(estimated),
                "actual_gas_used": int(receipt.gasUsed),
                "status": int(receipt.status),
            })
        else:
            web3.eth.wait_for_transaction_receipt(function.transact({"from": owner}))
        latest_commitment = contract.functions.computeCommitment(cid_hash, nth_hash, owner).call()

    anchor = contract.functions.anchorAuditHead(latest_commitment, word("audit-updated"))
    anchor_estimated = anchor.estimate_gas({"from": owner})
    anchor_receipt = web3.eth.wait_for_transaction_receipt(anchor.transact({"from": owner}))
    rows.append({
        "operation": "anchorAuditHead",
        "registry_size_before": 1001,
        "estimated_gas": int(anchor_estimated),
        "actual_gas_used": int(anchor_receipt.gasUsed),
        "status": int(anchor_receipt.status),
    })

    # Negative security checks are expected to revert.
    checks = []
    try:
        contract.functions.anchorAuditHead(latest_commitment, word("outsider")).call({"from": outsider})
        checks.append({"check": "non-registrant anchor rejected", "passed": False})
    except Exception:
        checks.append({"check": "non-registrant anchor rejected", "passed": True})
    try:
        contract.functions.registerImage(
            cid_hash, nth_hash, model_version, policy_hash, audit_head
        ).call({"from": owner})
        checks.append({"check": "duplicate commitment rejected", "passed": False})
    except Exception:
        checks.append({"check": "duplicate commitment rejected", "passed": True})
    try:
        contract.functions.registerImage(
            bytes(32), nth_hash, model_version, policy_hash, audit_head
        ).call({"from": owner})
        checks.append({"check": "zero CID hash rejected", "passed": False})
    except Exception:
        checks.append({"check": "zero CID hash rejected", "passed": True})

    results = pd.DataFrame(rows)
    results.to_csv(PROJECT / "results" / "contract_gas_scaling.csv", index=False)
    pd.DataFrame(checks).to_csv(PROJECT / "results" / "contract_security_checks.csv", index=False)
    metadata = {
        "solc_version": SOLC_VERSION,
        "optimizer": True,
        "optimizer_runs": 200,
        "evm": "PyEVMBackend via eth-tester",
        "chain": "deterministic local test chain; values are not mainnet fee quotes",
        "python": platform.python_version(),
        "deployment_gas_used": int(deployment.gasUsed),
        "contract_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    (PROJECT / "results" / "contract_benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print("\n", pd.DataFrame(checks).to_string(index=False))
    print("\n", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
