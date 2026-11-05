// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface Vm {
    struct Log {
        bytes32[] topics;
        bytes data;
        address emitter;
    }

    function envBytes(string calldata name) external view returns (bytes memory value);
    function envString(string calldata name) external view returns (string memory value);
    function envUint(string calldata name) external view returns (uint256 value);
    function envOr(string calldata name, uint256 defaultValue) external view returns (uint256 value);
    function snapshotState() external returns (uint256 snapshotId);
    function revertToState(uint256 snapshotId) external returns (bool success);
    function record() external;
    function accesses(address target) external returns (bytes32[] memory readSlots, bytes32[] memory writeSlots);
    function load(address target, bytes32 slot) external view returns (bytes32 data);
    function deal(address who, uint256 newBalance) external;
    function recordLogs() external;
    function getRecordedLogs() external returns (Log[] memory entries);
}

library Console {
    address constant CONSOLE = 0x000000000000000000636F6e736F6c652e6c6f67;

    function log(string memory message) internal view {
        bytes memory payload = abi.encodeWithSignature("log(string)", message);
        address console = CONSOLE;
        assembly {
            pop(staticcall(gas(), console, add(payload, 0x20), mload(payload), 0, 0))
        }
    }
}

contract SemanticReplayTest {
    Vm constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
    bytes16 constant HEX = "0123456789abcdef";
    uint256 constant TARGET_CALL_GAS_LIMIT = 30_000_000;

    function testSemanticReplay() public {
        vm.deal(address(this), type(uint128).max);

        string memory benchmarkId = vm.envString("SCDBENCH_ID");
        bytes memory originalInit = vm.envBytes("SCDBENCH_ORIGINAL_INIT");
        address original = deploy(originalInit);
        Console.log(string.concat("SCDBENCH_DEPLOY benchmark_id=", benchmarkId, " target=original addr=", addressHex(original)));

        uint256 hasPredicted = vm.envOr("SCDBENCH_HAS_PREDICTED", uint256(0));
        address predicted = address(0);
        if (hasPredicted != 0) {
            bytes memory predictedInit = vm.envBytes("SCDBENCH_PREDICTED_INIT");
            predicted = deploy(predictedInit);
            Console.log(string.concat("SCDBENCH_DEPLOY benchmark_id=", benchmarkId, " target=predicted addr=", addressHex(predicted)));
        }

        uint256 calls = vm.envUint("SCDBENCH_CALLS");
        for (uint256 i = 0; i < calls; i++) {
            observeOne(benchmarkId, "original", original, i);
            if (predicted != address(0)) {
                observeOne(benchmarkId, "predicted", predicted, i);
            }
        }
    }

    function observeOne(string memory benchmarkId, string memory targetName, address target, uint256 index) internal {
        string memory idx = uintToString(index);
        string memory caseId = vm.envString(string.concat("SCDBENCH_CASE_ID_", idx));
        string memory signature = vm.envString(string.concat("SCDBENCH_SIG_", idx));
        bytes memory data = vm.envBytes(string.concat("SCDBENCH_CALLDATA_", idx));
        uint256 value = vm.envOr(string.concat("SCDBENCH_VALUE_", idx), uint256(0));

        uint256 outerSnapshotId = vm.snapshotState();
        applySetupCalls(target, index);
        bytes32[] memory writeSlots = discoverWriteSlots(target, data, value);
        bytes32[] memory beforeValues = readSlots(target, writeSlots);
        (bool success, bytes memory returnData, bytes32[] memory afterValues, Vm.Log[] memory logs) =
            executeAndRead(target, data, value, writeSlots);

        emitCallLine(benchmarkId, caseId, targetName, signature, success, returnData);
        emitStorageLines(benchmarkId, caseId, targetName, writeSlots, beforeValues, afterValues);
        emitLogLines(benchmarkId, caseId, targetName, target, logs);
        vm.revertToState(outerSnapshotId);
    }

    function applySetupCalls(address target, uint256 index) internal {
        string memory idx = uintToString(index);
        uint256 setupCount = vm.envOr(string.concat("SCDBENCH_SETUP_COUNT_", idx), uint256(0));
        for (uint256 j = 0; j < setupCount; j++) {
            string memory setupIdx = uintToString(j);
            bytes memory setupData = vm.envBytes(string.concat("SCDBENCH_SETUP_CALLDATA_", idx, "_", setupIdx));
            uint256 setupValue = vm.envOr(string.concat("SCDBENCH_SETUP_VALUE_", idx, "_", setupIdx), uint256(0));
            target.call{value: setupValue, gas: TARGET_CALL_GAS_LIMIT}(setupData);
        }
    }

    function discoverWriteSlots(address target, bytes memory data, uint256 value) internal returns (bytes32[] memory) {
        uint256 snapshotId = vm.snapshotState();
        vm.record();
        target.call{value: value, gas: TARGET_CALL_GAS_LIMIT}(data);
        (, bytes32[] memory writeSlots) = vm.accesses(target);
        vm.revertToState(snapshotId);
        return unique(writeSlots);
    }

    function readSlots(address target, bytes32[] memory slots) internal view returns (bytes32[] memory values) {
        values = new bytes32[](slots.length);
        for (uint256 i = 0; i < slots.length; i++) {
            values[i] = vm.load(target, slots[i]);
        }
    }

    function executeAndRead(address target, bytes memory data, uint256 value, bytes32[] memory slots)
        internal
        returns (bool success, bytes memory returnData, bytes32[] memory afterValues, Vm.Log[] memory logs)
    {
        uint256 snapshotId = vm.snapshotState();
        vm.recordLogs();
        (success, returnData) = target.call{value: value, gas: TARGET_CALL_GAS_LIMIT}(data);
        logs = vm.getRecordedLogs();
        afterValues = readSlots(target, slots);
        vm.revertToState(snapshotId);
    }

    function emitCallLine(
        string memory benchmarkId,
        string memory caseId,
        string memory targetName,
        string memory signature,
        bool success,
        bytes memory returnData
    ) internal view {
        Console.log(
            string.concat(
                "SCDBENCH_CALL benchmark_id=",
                benchmarkId,
                " case_id=",
                caseId,
                " target=",
                targetName,
                " sig=",
                signature,
                " success=",
                success ? "1" : "0",
                " returndata=",
                bytesHex(returnData)
            )
        );
    }

    function emitStorageLines(
        string memory benchmarkId,
        string memory caseId,
        string memory targetName,
        bytes32[] memory slots,
        bytes32[] memory beforeValues,
        bytes32[] memory afterValues
    ) internal view {
        for (uint256 i = 0; i < slots.length; i++) {
            Console.log(
                string.concat(
                    "SCDBENCH_STORAGE benchmark_id=",
                    benchmarkId,
                    " case_id=",
                    caseId,
                    " target=",
                    targetName,
                    " slot=",
                    bytes32Hex(slots[i]),
                    " before=",
                    bytes32Hex(beforeValues[i]),
                    " after=",
                    bytes32Hex(afterValues[i])
                )
            );
        }
    }

    function emitLogLines(
        string memory benchmarkId,
        string memory caseId,
        string memory targetName,
        address target,
        Vm.Log[] memory logs
    ) internal view {
        for (uint256 i = 0; i < logs.length; i++) {
            string memory emitter = logs[i].emitter == target ? "<target>" : addressHex(logs[i].emitter);
            Console.log(
                string.concat(
                    "SCDBENCH_LOG benchmark_id=",
                    benchmarkId,
                    " case_id=",
                    caseId,
                    " target=",
                    targetName,
                    " log_index=",
                    uintToString(i),
                    " emitter=",
                    emitter,
                    " topics=",
                    topicsHex(logs[i].topics),
                    " data=",
                    bytesHex(logs[i].data)
                )
            );
        }
    }

    function deploy(bytes memory initCode) internal returns (address deployed) {
        assembly {
            deployed := create(0, add(initCode, 0x20), mload(initCode))
        }
        require(deployed != address(0), "SCDBench deploy failed");
    }

    function unique(bytes32[] memory values) internal pure returns (bytes32[] memory out) {
        bytes32[] memory scratch = new bytes32[](values.length);
        uint256 count = 0;
        for (uint256 i = 0; i < values.length; i++) {
            bool seen = false;
            for (uint256 j = 0; j < count; j++) {
                if (scratch[j] == values[i]) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                scratch[count] = values[i];
                count++;
            }
        }
        out = new bytes32[](count);
        for (uint256 i = 0; i < count; i++) {
            out[i] = scratch[i];
        }
    }

    function uintToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits = 0;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }

    function addressHex(address value) internal pure returns (string memory) {
        return fixedHex(abi.encodePacked(value));
    }

    function bytes32Hex(bytes32 value) internal pure returns (string memory) {
        return fixedHex(abi.encodePacked(value));
    }

    function bytesHex(bytes memory data) internal pure returns (string memory) {
        return fixedHex(data);
    }

    function topicsHex(bytes32[] memory topics) internal pure returns (string memory) {
        string memory out = "";
        for (uint256 i = 0; i < topics.length; i++) {
            if (i == 0) {
                out = bytes32Hex(topics[i]);
            } else {
                out = string.concat(out, ",", bytes32Hex(topics[i]));
            }
        }
        return out;
    }

    function fixedHex(bytes memory data) internal pure returns (string memory) {
        bytes memory out = new bytes(2 + data.length * 2);
        out[0] = "0";
        out[1] = "x";
        for (uint256 i = 0; i < data.length; i++) {
            out[2 + i * 2] = HEX[uint8(data[i] >> 4)];
            out[3 + i * 2] = HEX[uint8(data[i] & 0x0f)];
        }
        return string(out);
    }
}
