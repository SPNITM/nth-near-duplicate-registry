// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Minimal, tamper-evident image-registration anchor
/// @notice Similarity scoring remains off-chain. This contract proves what
/// commitment, model version, policy, and audit-log head were asserted by the
/// registrant at a given block; it does not prove that the perceptual decision
/// was correct.
contract ImageRegistryV2 {
    struct Registration {
        address registrant;
        bytes32 cidHash;
        bytes32 nthHash;
        bytes32 modelVersion;
        bytes32 policyHash;
        bytes32 auditHead;
        uint64 registeredAt;
    }

    mapping(bytes32 => Registration) private registrations;

    event ImageRegistered(
        bytes32 indexed commitment,
        address indexed registrant,
        bytes32 indexed cidHash,
        bytes32 nthHash,
        bytes32 modelVersion,
        bytes32 policyHash,
        bytes32 auditHead,
        uint64 registeredAt
    );

    event AuditHeadAnchored(
        bytes32 indexed commitment,
        bytes32 indexed previousHead,
        bytes32 indexed newHead,
        uint64 anchoredAt
    );

    error AlreadyRegistered(bytes32 commitment);
    error UnknownCommitment(bytes32 commitment);
    error NotRegistrant(address caller);
    error ZeroValue();

    function computeCommitment(
        bytes32 cidHash,
        bytes32 nthHash,
        address registrant
    ) public pure returns (bytes32) {
        return keccak256(abi.encode(cidHash, nthHash, registrant));
    }

    function registerImage(
        bytes32 cidHash,
        bytes32 nthHash,
        bytes32 modelVersion,
        bytes32 policyHash,
        bytes32 auditHead
    ) external returns (bytes32 commitment) {
        if (
            cidHash == bytes32(0) || nthHash == bytes32(0)
                || modelVersion == bytes32(0) || policyHash == bytes32(0)
        ) revert ZeroValue();
        commitment = computeCommitment(cidHash, nthHash, msg.sender);
        if (registrations[commitment].registrant != address(0)) {
            revert AlreadyRegistered(commitment);
        }
        uint64 timestamp = uint64(block.timestamp);
        registrations[commitment] = Registration({
            registrant: msg.sender,
            cidHash: cidHash,
            nthHash: nthHash,
            modelVersion: modelVersion,
            policyHash: policyHash,
            auditHead: auditHead,
            registeredAt: timestamp
        });
        emit ImageRegistered(
            commitment,
            msg.sender,
            cidHash,
            nthHash,
            modelVersion,
            policyHash,
            auditHead,
            timestamp
        );
    }

    function anchorAuditHead(bytes32 commitment, bytes32 newHead) external {
        Registration storage item = registrations[commitment];
        if (item.registrant == address(0)) revert UnknownCommitment(commitment);
        if (item.registrant != msg.sender) revert NotRegistrant(msg.sender);
        if (newHead == bytes32(0)) revert ZeroValue();
        bytes32 previous = item.auditHead;
        item.auditHead = newHead;
        emit AuditHeadAnchored(
            commitment, previous, newHead, uint64(block.timestamp)
        );
    }

    function getRegistration(bytes32 commitment)
        external
        view
        returns (Registration memory)
    {
        Registration memory item = registrations[commitment];
        if (item.registrant == address(0)) revert UnknownCommitment(commitment);
        return item;
    }
}

