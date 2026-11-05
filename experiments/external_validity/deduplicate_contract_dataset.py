#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from array import array
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


SOLIDITY_KEYWORDS = {
    "abstract",
    "address",
    "after",
    "alias",
    "anonymous",
    "as",
    "assembly",
    "assert",
    "bool",
    "break",
    "bytes",
    "calldata",
    "case",
    "catch",
    "constant",
    "constructor",
    "continue",
    "contract",
    "default",
    "delete",
    "do",
    "else",
    "emit",
    "enum",
    "error",
    "event",
    "external",
    "fallback",
    "false",
    "for",
    "from",
    "function",
    "global",
    "if",
    "immutable",
    "import",
    "indexed",
    "interface",
    "internal",
    "is",
    "let",
    "library",
    "mapping",
    "memory",
    "modifier",
    "new",
    "override",
    "payable",
    "pragma",
    "private",
    "public",
    "pure",
    "receive",
    "return",
    "returns",
    "revert",
    "storage",
    "string",
    "struct",
    "switch",
    "true",
    "try",
    "type",
    "uint",
    "using",
    "var",
    "view",
    "virtual",
    "while",
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\b")
STRING_RE = re.compile(r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\'', re.DOTALL)
WS_RE = re.compile(r"\s+")
SPDX_RE = re.compile(r"^\s*//\s*SPDX-License-Identifier:.*$", re.MULTILINE)
PRAGMA_RE = re.compile(r"^\s*pragma\s+[^;]+;\s*$", re.MULTILINE)
FUNC_SIG_RE = re.compile(
    r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:[^;{]*)",
    re.DOTALL,
)
EVENT_SIG_RE = re.compile(r"\bevent\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*;", re.DOTALL)
IMPORT_ALIAS_RE = re.compile(r"^\s*import\s+.*?;\s*$", re.MULTILINE)

SIMPLIFIED_SOURCE_KEY = "simplified_code"
SIMPLIFIED_BYTECODE_KEY = "simplified_bytecode"
ABI_KEYS = ("abi", "ABI")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CHUNK_CHARS = 1200
EMBED_CHUNK_OVERLAP = 200
EMBED_MAX_CHUNKS_PER_CONTRACT = 4
EMBED_SOURCE_VIEW_MAX_CHARS = 1200
EMBED_SOURCE_VIEW_DECL_LIMIT = 96

OPCODES: Dict[int, str] = {
    0x00: "STOP",
    0x01: "ADD",
    0x02: "MUL",
    0x03: "SUB",
    0x04: "DIV",
    0x05: "SDIV",
    0x06: "MOD",
    0x07: "SMOD",
    0x08: "ADDMOD",
    0x09: "MULMOD",
    0x0A: "EXP",
    0x0B: "SIGNEXTEND",
    0x10: "LT",
    0x11: "GT",
    0x12: "SLT",
    0x13: "SGT",
    0x14: "EQ",
    0x15: "ISZERO",
    0x16: "AND",
    0x17: "OR",
    0x18: "XOR",
    0x19: "NOT",
    0x1A: "BYTE",
    0x1B: "SHL",
    0x1C: "SHR",
    0x1D: "SAR",
    0x20: "SHA3",
    0x30: "ADDRESS",
    0x31: "BALANCE",
    0x32: "ORIGIN",
    0x33: "CALLER",
    0x34: "CALLVALUE",
    0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY",
    0x38: "CODESIZE",
    0x39: "CODECOPY",
    0x3A: "GASPRICE",
    0x3B: "EXTCODESIZE",
    0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY",
    0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH",
    0x41: "COINBASE",
    0x42: "TIMESTAMP",
    0x43: "NUMBER",
    0x44: "PREVRANDAO",
    0x45: "GASLIMIT",
    0x46: "CHAINID",
    0x47: "SELFBALANCE",
    0x48: "BASEFEE",
    0x50: "POP",
    0x51: "MLOAD",
    0x52: "MSTORE",
    0x53: "MSTORE8",
    0x54: "SLOAD",
    0x55: "SSTORE",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x58: "PC",
    0x59: "MSIZE",
    0x5A: "GAS",
    0x5B: "JUMPDEST",
    0x5F: "PUSH0",
    0xA0: "LOG0",
    0xA1: "LOG1",
    0xA2: "LOG2",
    0xA3: "LOG3",
    0xA4: "LOG4",
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF3: "RETURN",
    0xF4: "DELEGATECALL",
    0xF5: "CREATE2",
    0xFA: "STATICCALL",
    0xFD: "REVERT",
    0xFE: "INVALID",
    0xFF: "SELFDESTRUCT",
}


def stable_hash(text: str, size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_comments_preserving_strings(code: str) -> str:
    out: List[str] = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                cur = code[i]
                out.append(cur)
                if cur == "\\" and i + 1 < n:
                    out.append(code[i + 1])
                    i += 2
                    continue
                if cur == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            i += 2
            while i < n and code[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            i += 2
            while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def training_source_view(source: str) -> str:
    text = normalize_newlines(source)
    text = SPDX_RE.sub("", text)
    text = IMPORT_ALIAS_RE.sub("", text)
    text = strip_comments_preserving_strings(text)
    lines = [line.rstrip() for line in text.split("\n")]
    compact: List[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        compact.append(line)
        previous_blank = blank
    return "\n".join(compact).strip()


def _replace_literals(text: str) -> str:
    text = STRING_RE.sub(" STR ", text)
    text = NUM_RE.sub(" NUM ", text)
    return text


def similarity_source_tokens(source: str) -> List[str]:
    text = training_source_view(source)
    text = PRAGMA_RE.sub("", text)
    text = _replace_literals(text)
    tokens: List[str] = []
    idx = 0
    while idx < len(text):
        match = IDENT_RE.match(text, idx)
        if match:
            word = match.group(0)
            lowered = word.lower()
            tokens.append(word if lowered in SOLIDITY_KEYWORDS else "ID")
            idx = match.end()
            continue
        ch = text[idx]
        if ch.isspace():
            idx += 1
            continue
        tokens.append(ch)
        idx += 1
    return tokens


def shingled_tokens(tokens: Sequence[str], n: int) -> List[str]:
    if len(tokens) <= n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def simhash64(features: Sequence[str]) -> int:
    if not features:
        return 0
    weights = [0] * 64
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            if value & (1 << bit):
                weights[bit] += 1
            else:
                weights[bit] -= 1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def chunk_text_for_embedding(text: str, max_chars: int = EMBED_CHUNK_CHARS, overlap: int = EMBED_CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    step = max(1, max_chars - overlap)
    while start < text_len:
        end = min(text_len, start + max_chars)
        if end < text_len:
            newline = text.rfind("\n", start + max_chars // 2, end)
            if newline > start:
                end = newline
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(start + 1, end - overlap)
    if len(chunks) <= EMBED_MAX_CHUNKS_PER_CONTRACT:
        return chunks
    if EMBED_MAX_CHUNKS_PER_CONTRACT == 1:
        return [chunks[0]]
    selected: List[str] = []
    last_idx = len(chunks) - 1
    for pos in range(EMBED_MAX_CHUNKS_PER_CONTRACT):
        idx = round(pos * last_idx / (EMBED_MAX_CHUNKS_PER_CONTRACT - 1))
        chunk = chunks[idx]
        if not selected or selected[-1] != chunk:
            selected.append(chunk)
    return selected


def embedding_source_view(source: str) -> str:
    text = training_source_view(source)
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    decl_pattern = re.compile(
        r"^(abstract\s+contract|contract|interface|library|struct|enum|event|error|function|modifier|constructor|fallback|receive)\b"
    )
    selected: List[str] = []
    seen = set()
    for line in lines:
        if decl_pattern.match(line):
            if line not in seen:
                selected.append(line)
                seen.add(line)
            if len(selected) >= EMBED_SOURCE_VIEW_DECL_LIMIT:
                break
    if not selected:
        selected = lines[: min(len(lines), 80)]
    text = "\n".join(selected)
    if len(text) <= EMBED_SOURCE_VIEW_MAX_CHARS:
        return text
    head = text[: EMBED_SOURCE_VIEW_MAX_CHARS // 2].rstrip()
    tail = text[-(EMBED_SOURCE_VIEW_MAX_CHARS // 3) :].lstrip()
    return (head + "\n...\n" + tail)[:EMBED_SOURCE_VIEW_MAX_CHARS]


@lru_cache(maxsize=4)
def load_source_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=True)


def representative_source_text(input_path: Path, feature: "FeatureRecord") -> str:
    raw = read_line_at_offset(input_path, feature.offset)
    row = json.loads(raw)
    source = row.get(SIMPLIFIED_SOURCE_KEY) if isinstance(row.get(SIMPLIFIED_SOURCE_KEY), str) else ""
    return embedding_source_view(source)


def build_representative_source_embeddings(
    input_path: Path,
    features: List["FeatureRecord"],
    representative_indices: Sequence[int],
    model_name: str,
    batch_size: int = 64,
) -> Dict[int, object]:
    import numpy as np

    if not representative_indices:
        return {}
    model = load_source_embedding_model(model_name)
    chunk_texts: List[str] = []
    owners: List[int] = []
    for idx in representative_indices:
        text = representative_source_text(input_path, features[idx])
        chunks = chunk_text_for_embedding(text)
        if not chunks:
            continue
        for chunk in chunks:
            chunk_texts.append(chunk)
            owners.append(idx)
    if not chunk_texts:
        return {}
    vectors = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    grouped: Dict[int, List[object]] = defaultdict(list)
    for idx, vec in zip(owners, vectors):
        grouped[idx].append(vec)
    result: Dict[int, object] = {}
    for idx, vecs in grouped.items():
        mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        result[idx] = mean_vec
    return result


def hamming_distance_u64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def clean_hex_bytecode(bytecode: str) -> str:
    text = bytecode.strip()
    if text.startswith("0x"):
        text = text[2:]
    text = re.sub(r"__[^_]{1,64}__", "00" * 20, text)
    text = re.sub(r"[^0-9a-fA-F]", "", text)
    if len(text) % 2 == 1:
        text = text[:-1]
    return text.lower()


def strip_solc_metadata(hexcode: str) -> str:
    if len(hexcode) < 4:
        return hexcode
    try:
        meta_len = int(hexcode[-4:], 16)
    except ValueError:
        return hexcode
    total_meta_hex = meta_len * 2 + 4
    if total_meta_hex >= len(hexcode):
        return hexcode
    start = len(hexcode) - total_meta_hex
    if start < 0:
        return hexcode
    header_byte = int(hexcode[start : start + 2], 16)
    if 0xA0 <= header_byte <= 0xBF:
        return hexcode[:start]
    return hexcode


def normalized_opcode_tokens(bytecode: str) -> List[str]:
    hexcode = strip_solc_metadata(clean_hex_bytecode(bytecode))
    tokens: List[str] = []
    i = 0
    while i + 2 <= len(hexcode):
        opcode = int(hexcode[i : i + 2], 16)
        i += 2
        if 0x60 <= opcode <= 0x7F:
            push_len = opcode - 0x5F
            data = hexcode[i : i + push_len * 2]
            i += push_len * 2
            if push_len <= 4:
                tokens.append(f"PUSH{push_len}:{data.lstrip('0') or '0'}")
            elif push_len == 20:
                tokens.append("PUSH20:ADDR")
            else:
                tokens.append(f"PUSH{push_len}:DATA")
            continue
        if 0x80 <= opcode <= 0x8F:
            tokens.append(f"DUP{opcode - 0x7F}")
            continue
        if 0x90 <= opcode <= 0x9F:
            tokens.append(f"SWAP{opcode - 0x8F}")
            continue
        if 0xA0 <= opcode <= 0xA4:
            tokens.append(f"LOG{opcode - 0xA0}")
            continue
        tokens.append(OPCODES.get(opcode, f"OP_{opcode:02x}"))
    return tokens


def detect_wrapper_type(source: str, bytecode: str) -> Optional[str]:
    clean = strip_solc_metadata(clean_hex_bytecode(bytecode))
    if re.search(r"363d3d373d3d3d363d73[0-9a-f]{40}5af43d82803e903d9160[0-9a-f]{2}57fd5bf3", clean):
        return "minimal_proxy"
    lower = source.lower()
    if "delegatecall" in lower and "fallback" in lower:
        return "proxy"
    if "fallback" in lower and "call(" in lower:
        return "forwarder"
    return None


def parse_abi_signatures(row: dict) -> FrozenSet[str]:
    raw = None
    for key in ABI_KEYS:
        if key in row:
            raw = row[key]
            break
    abi_obj = raw
    if isinstance(raw, str):
        try:
            abi_obj = json.loads(raw)
        except Exception:
            abi_obj = None
    signatures: Set[str] = set()
    if not isinstance(abi_obj, list):
        return frozenset()
    for entry in abi_obj:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        name = entry.get("name")
        if kind not in {"function", "event", "error"} or not isinstance(name, str):
            continue
        inputs = entry.get("inputs")
        arg_types: List[str] = []
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict) and isinstance(item.get("type"), str):
                    arg_types.append(item["type"])
        signatures.add(f"{kind}:{name}({','.join(arg_types)})")
    return frozenset(sorted(signatures))


def source_signatures(source: str) -> FrozenSet[str]:
    signatures: Set[str] = set()
    for match in FUNC_SIG_RE.finditer(source):
        name = match.group(1)
        args = collapse_arg_types(match.group(2))
        signatures.add(f"function:{name}({args})")
    for match in EVENT_SIG_RE.finditer(source):
        name = match.group(1)
        args = collapse_arg_types(match.group(2))
        signatures.add(f"event:{name}({args})")
    return frozenset(sorted(signatures))


def collapse_arg_types(arg_text: str) -> str:
    if not arg_text.strip():
        return ""
    chunks = []
    depth = 0
    current: List[str] = []
    for ch in arg_text:
        if ch == "," and depth == 0:
            chunks.append("".join(current).strip())
            current = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        current.append(ch)
    if current:
        chunks.append("".join(current).strip())
    arg_types: List[str] = []
    for chunk in chunks:
        parts = chunk.split()
        if not parts:
            continue
        arg_types.append(parts[0])
    return ",".join(arg_types)


def interface_signatures(row: dict, source: str) -> FrozenSet[str]:
    abi_sigs = parse_abi_signatures(row)
    if abi_sigs:
        return abi_sigs
    return source_signatures(source)


def interface_hash(signatures: FrozenSet[str]) -> str:
    if not signatures:
        return "noiface"
    return stable_hash("\n".join(sorted(signatures)))


def jaccard_similarity(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


ERC20_SIGS = {
    "function:totalSupply()",
    "function:balanceOf(address)",
    "function:transfer(address,uint256)",
    "function:transferFrom(address,address,uint256)",
    "function:approve(address,uint256)",
    "function:allowance(address,address)",
}
ERC721_SIGS = {
    "function:ownerOf(uint256)",
    "function:balanceOf(address)",
    "function:approve(address,uint256)",
    "function:getApproved(uint256)",
    "function:isApprovedForAll(address,address)",
    "function:setApprovalForAll(address,bool)",
}
ERC1155_SIGS = {
    "function:balanceOf(address,uint256)",
    "function:balanceOfBatch(address[],uint256[])",
    "function:safeTransferFrom(address,address,uint256,uint256,bytes)",
    "function:safeBatchTransferFrom(address,address,uint256[],uint256[],bytes)",
    "function:setApprovalForAll(address,bool)",
    "function:isApprovedForAll(address,address)",
}
ERC4626_SIGS = {
    "function:asset()",
    "function:totalAssets()",
    "function:deposit(uint256,address)",
    "function:mint(uint256,address)",
    "function:withdraw(uint256,address,address)",
    "function:redeem(uint256,address,address)",
}
AMM_PAIR_SIGS = {
    "function:swap(uint256,uint256,address,bytes)",
    "function:mint(address)",
    "function:burn(address)",
    "function:getReserves()",
}
GOVERNANCE_SIGS = {
    "function:propose(address[],uint256[],bytes[],string)",
    "function:castVote(uint256,uint8)",
    "function:quorum(uint256)",
}

def signature_function_names(signatures: FrozenSet[str]) -> FrozenSet[str]:
    names: Set[str] = set()
    for sig in signatures:
        if not sig.startswith("function:"):
            continue
        body = sig[len("function:") :]
        name, _, _ = body.partition("(")
        if name:
            names.add(name)
    return frozenset(names)


def has_any(names: FrozenSet[str], expected: Set[str]) -> bool:
    return bool(names & expected)


def has_at_least(names: FrozenSet[str], expected: Set[str], threshold: int) -> bool:
    return len(names & expected) >= threshold


ACCESS_CONTROL_FNS = {"grantRole", "revokeRole", "renounceRole", "hasRole", "getRoleAdmin"}
PAUSABLE_FNS = {"pause", "unpause", "paused"}
TIMELOCK_FNS = {"schedule", "execute", "cancel", "getMinDelay"}
STAKING_FNS = {"stake", "withdraw", "exit", "earned", "getReward", "rewardPerToken"}
ROUTER_FNS = {
    "swapExactTokensForTokens",
    "swapTokensForExactTokens",
    "swapExactETHForTokens",
    "swapTokensForExactETH",
    "swapExactTokensForETH",
    "exactInput",
    "exactOutput",
    "exactInputSingle",
    "exactOutputSingle",
    "multicall",
}
FACTORY_FNS = {
    "create",
    "create2",
    "createPair",
    "createPool",
    "createVault",
    "createAccount",
    "deploy",
    "deployDeterministic",
    "clone",
    "cloneDeterministic",
}
VAULT_FNS = {"deposit", "withdraw", "mint", "redeem", "asset", "totalAssets"}
ORACLE_FNS = {"latestAnswer", "latestRoundData", "decimals", "description", "consult", "getPrice"}
BRIDGE_FNS = {
    "bridge",
    "bridgeToken",
    "send",
    "sendFrom",
    "lzReceive",
    "quoteSend",
    "estimateSendFee",
    "depositETHTo",
    "finalizeWithdrawal",
}
SALE_FNS = {"buy", "buyTokens", "purchase", "claim", "rate", "saleActive"}
MARKET_FNS = {
    "fulfillOrder",
    "fulfillBasicOrder",
    "matchOrders",
    "cancel",
    "cancelOrder",
    "validate",
    "acceptOffer",
}


def infer_family_label(
    signatures: FrozenSet[str], wrapper_type: Optional[str], contract_name: str
) -> str:
    if wrapper_type:
        return wrapper_type
    fn_names = signature_function_names(signatures)
    name = contract_name.lower()
    if ERC20_SIGS <= signatures:
        return "erc20_like"
    if len(ERC721_SIGS & signatures) >= 5:
        return "erc721_like"
    if len(ERC1155_SIGS & signatures) >= 5:
        return "erc1155_like"
    if len(ERC4626_SIGS & signatures) >= 4:
        return "erc4626_like"
    if len(AMM_PAIR_SIGS & signatures) >= 3:
        return "amm_pair_like"
    if len(GOVERNANCE_SIGS & signatures) >= 2:
        return "governance_like"
    if has_at_least(fn_names, ACCESS_CONTROL_FNS, 3):
        return "access_control_like"
    if "function:implementation()" in signatures or "function:upgradeTo(address)" in signatures:
        return "upgradeable_proxy"
    if has_at_least(fn_names, TIMELOCK_FNS, 3) or "timelock" in name:
        return "timelock_like"
    if has_at_least(fn_names, STAKING_FNS, 3) or (
        ("staking" in name or "chef" in name) and has_at_least(fn_names, STAKING_FNS, 2)
    ):
        return "staking_like"
    if has_at_least(fn_names, ROUTER_FNS, 2) or (
        "router" in name and has_any(fn_names, ROUTER_FNS)
    ):
        return "router_like"
    if has_at_least(fn_names, FACTORY_FNS, 2) or (
        ("factory" in name or "deployer" in name) and has_any(fn_names, FACTORY_FNS)
    ):
        return "factory_like"
    if has_at_least(fn_names, BRIDGE_FNS, 2) or (
        ("bridge" in name or "oft" in name) and has_any(fn_names, BRIDGE_FNS)
    ):
        return "bridge_like"
    if has_at_least(fn_names, ORACLE_FNS, 2) or ("oracle" in name and has_any(fn_names, ORACLE_FNS)):
        return "oracle_like"
    if len(VAULT_FNS & fn_names) >= 3 or ("vault" in name and len(VAULT_FNS & fn_names) >= 2):
        return "vault_like"
    if has_at_least(fn_names, SALE_FNS, 2) or (
        any(token in name for token in ("sale", "presale", "crowdsale")) and has_any(fn_names, SALE_FNS)
    ):
        return "sale_like"
    if has_at_least(fn_names, MARKET_FNS, 2) or (
        any(token in name for token in ("market", "exchange", "orderbook")) and has_any(fn_names, MARKET_FNS)
    ):
        return "market_like"
    if has_at_least(fn_names, PAUSABLE_FNS, 2):
        return "pausable_like"
    if "function:owner()" in signatures and "function:transferOwnership(address)" in signatures:
        return "ownable_misc"
    return "misc"


def get_first_key(d: dict, keys: Sequence[str]) -> Tuple[Optional[str], Optional[object]]:
    for key in keys:
        if key in d:
            return key, d[key]
    return None, None


def stable_bucket_keys(value: int, prefix: str) -> List[str]:
    return [f"{prefix}:{idx}:{(value >> (16 * idx)) & 0xFFFF:04x}" for idx in range(4)]


def quality_score(row: dict, source: str, bytecode: str, signatures: FrozenSet[str]) -> Tuple[int, int, int, int]:
    compiler_version = row.get("compiler_version")
    has_compiler = 1 if isinstance(compiler_version, str) and compiler_version else 0
    has_abi = 1 if signatures else 0
    return (has_compiler, has_abi, len(bytecode), len(source))


@dataclass
class FeatureRecord:
    line_no: int
    offset: int
    address: str
    contract_name: str
    compiler_version: str
    compile_success: bool
    exact_key: str
    source_train_hash: str
    source_similarity_hash: str
    bytecode_logic_hash: str
    source_simhash: int
    bytecode_simhash: int
    bytecode_token_count: int
    interface_hash: str
    signatures: FrozenSet[str]
    wrapper_type: str
    family_label: str
    score: Tuple[int, int, int, int]
    retained_exact: bool = False
    exact_group_id: int = -1
    near_cluster_id: int = -1
    family_cluster_id: str = ""
    split: str = ""
    retained_final: bool = False


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def read_line_at_offset(path: Path, offset: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        return f.readline().rstrip("\n")


def iter_offsets(path: Path, limit: Optional[int]) -> Iterator[Tuple[int, int]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lineno = 0
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            lineno += 1
            if limit is not None and lineno > limit:
                break
            yield lineno, offset


def compute_features(row: dict, line_no: int, offset: int) -> FeatureRecord:
    source = row.get(SIMPLIFIED_SOURCE_KEY) if isinstance(row.get(SIMPLIFIED_SOURCE_KEY), str) else ""
    bytecode = (
        row.get(SIMPLIFIED_BYTECODE_KEY)
        if isinstance(row.get(SIMPLIFIED_BYTECODE_KEY), str)
        else ""
    )

    compile_success = bool(source and bytecode)
    training_view = training_source_view(source) if source else ""
    similarity_tokens = similarity_source_tokens(source) if source else []
    similarity_view = " ".join(similarity_tokens)
    source_shingles = shingled_tokens(similarity_tokens, 5)
    bytecode_tokens = normalized_opcode_tokens(bytecode) if bytecode else []
    opcode_shingles = shingled_tokens(bytecode_tokens, 4)
    signatures = interface_signatures(row, source) if source else frozenset()
    wrapper_type = detect_wrapper_type(source, bytecode) or ""
    family_label = ""
    source_train_hash = stable_hash(training_view) if training_view else "nosource"
    source_similarity_hash = stable_hash(similarity_view) if similarity_view else "nosource"
    bytecode_logic_hash = stable_hash(" ".join(bytecode_tokens)) if bytecode_tokens else "nobytecode"
    exact_key = bytecode_logic_hash if bytecode_tokens else source_train_hash
    return FeatureRecord(
        line_no=line_no,
        offset=offset,
        address=str(row.get("address") or ""),
        contract_name=str(row.get("contract_name") or ""),
        compiler_version=str(row.get("compiler_version") or ""),
        compile_success=compile_success,
        exact_key=exact_key,
        source_train_hash=source_train_hash,
        source_similarity_hash=source_similarity_hash,
        bytecode_logic_hash=bytecode_logic_hash,
        source_simhash=simhash64(source_shingles),
        bytecode_simhash=simhash64(opcode_shingles),
        bytecode_token_count=len(bytecode_tokens),
        interface_hash=interface_hash(signatures),
        signatures=signatures,
        wrapper_type=wrapper_type or "none",
        family_label=family_label,
        score=quality_score(row, source, bytecode, signatures),
    )


def is_better_representative(candidate: FeatureRecord, current: FeatureRecord) -> bool:
    if candidate.score != current.score:
        return candidate.score > current.score
    return candidate.line_no < current.line_no


def build_exact_groups(features: List[FeatureRecord]) -> Tuple[Dict[str, int], Dict[int, List[int]], Dict[int, int]]:
    key_to_group: Dict[str, int] = {}
    group_members: Dict[int, List[int]] = defaultdict(list)
    group_representative: Dict[int, int] = {}
    for idx, record in enumerate(features):
        if not record.compile_success:
            continue
        group_id = key_to_group.get(record.exact_key)
        if group_id is None:
            group_id = len(key_to_group)
            key_to_group[record.exact_key] = group_id
            group_representative[group_id] = idx
        else:
            current_rep_idx = group_representative[group_id]
            if is_better_representative(record, features[current_rep_idx]):
                group_representative[group_id] = idx
        record.exact_group_id = group_id
        group_members[group_id].append(idx)
    for rep_idx in group_representative.values():
        features[rep_idx].retained_exact = True
    return key_to_group, group_members, group_representative


def near_duplicate_match(a: FeatureRecord, b: FeatureRecord) -> bool:
    byte_hd = hamming_distance_u64(a.bytecode_simhash, b.bytecode_simhash)
    source_hd = hamming_distance_u64(a.source_simhash, b.source_simhash)
    max_tokens = max(a.bytecode_token_count, b.bytecode_token_count, 1)
    length_gap = abs(a.bytecode_token_count - b.bytecode_token_count)
    length_close = length_gap <= max(8, int(max_tokens * 0.1))
    if a.interface_hash != "noiface" and a.interface_hash == b.interface_hash and length_close and byte_hd <= 2:
        return True
    if a.interface_hash != "noiface" and a.interface_hash == b.interface_hash and length_close and byte_hd <= 4 and source_hd <= 6:
        return True
    if (
        a.wrapper_type != "none"
        and a.wrapper_type == b.wrapper_type
        and length_close
        and byte_hd <= 2
    ):
        return True
    if (
        a.interface_hash == "noiface"
        and b.interface_hash == "noiface"
        and length_close
        and byte_hd <= 1
        and source_hd <= 4
    ):
        return True
    return False


def build_near_duplicate_clusters(features: List[FeatureRecord], representative_indices: Sequence[int]) -> Dict[int, List[int]]:
    if not representative_indices:
        return {}
    uf = UnionFind(len(features))
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in representative_indices:
        record = features[idx]
        keys: List[str] = []
        if record.interface_hash != "noiface":
            keys.append(f"iface:{record.interface_hash}")
        if record.wrapper_type != "none":
            keys.append(f"wrap:{record.wrapper_type}")
        keys.extend(stable_bucket_keys(record.bytecode_simhash, "bc"))
        if record.interface_hash == "noiface":
            keys.extend(stable_bucket_keys(record.source_simhash, "src"))
        compared: Set[int] = set()
        for key in keys:
            candidates = buckets[key]
            for other_idx in candidates[-128:]:
                if other_idx in compared:
                    continue
                compared.add(other_idx)
                if near_duplicate_match(record, features[other_idx]):
                    uf.union(idx, other_idx)
            buckets[key].append(idx)

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in representative_indices:
        components[uf.find(idx)].append(idx)
    return components


def similarity_family_match(a: FeatureRecord, b: FeatureRecord, src_cos: float) -> bool:
    iface_j = jaccard_similarity(a.signatures, b.signatures)
    byte_hd = hamming_distance_u64(a.bytecode_simhash, b.bytecode_simhash)
    if a.interface_hash != "noiface" and a.interface_hash == b.interface_hash and src_cos >= 0.60:
        return True
    if a.wrapper_type != "none" and a.wrapper_type == b.wrapper_type and src_cos >= 0.68:
        return True
    if a.interface_hash == "noiface" and b.interface_hash == "noiface" and src_cos >= 0.86 and byte_hd <= 8:
        return True
    if a.interface_hash == "noiface" and b.interface_hash == "noiface" and src_cos >= 0.82 and iface_j >= 0.70 and byte_hd <= 10:
        return True
    return False


def similarity_family_bucket(feature: FeatureRecord) -> str:
    if feature.interface_hash != "noiface":
        return f"iface:{feature.interface_hash}"
    if feature.wrapper_type != "none":
        return f"wrap:{feature.wrapper_type}"
    sig_band = min(len(feature.signatures), 24)
    byte_band = min(feature.bytecode_token_count // 200, 20)
    return f"noiface:{sig_band}:{byte_band}"


def build_similarity_family_clusters(
    input_path: Path,
    features: List[FeatureRecord],
    near_clusters: Dict[int, List[int]],
    embedding_model: str,
) -> Tuple[Dict[int, List[int]], Dict[int, object]]:
    if not near_clusters:
        return {}, {}
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    roots = sorted(near_clusters)
    root_pos = {root: pos for pos, root in enumerate(roots)}
    reps = {root: cluster_representative(features, members) for root, members in near_clusters.items()}
    rep_indices = [reps[root] for root in roots]
    print(f"[families] embedding_representatives={len(rep_indices)}", flush=True)
    rep_embeddings = build_representative_source_embeddings(input_path, features, rep_indices, embedding_model)
    usable_roots = [root for root in roots if reps[root] in rep_embeddings]
    if not usable_roots:
        return ({root: [root] for root in roots}, {})
    print(f"[families] usable_embeddings={len(usable_roots)}", flush=True)
    uf = UnionFind(len(roots))
    buckets: Dict[str, List[int]] = defaultdict(list)
    for root in usable_roots:
        buckets[similarity_family_bucket(features[reps[root]])].append(root)
    for bucket_roots in buckets.values():
        if len(bucket_roots) < 2:
            continue
        usable_matrix = np.stack([rep_embeddings[reps[root]] for root in bucket_roots], axis=0)
        neighbor_count = min(32, len(bucket_roots))
        nn = NearestNeighbors(metric="cosine", n_neighbors=neighbor_count)
        nn.fit(usable_matrix)
        distances, indices = nn.kneighbors(usable_matrix)
        for pos, root in enumerate(bucket_roots):
            rep = features[reps[root]]
            for dist, neigh_idx in zip(distances[pos], indices[pos]):
                other_root = bucket_roots[neigh_idx]
                if other_root == root:
                    continue
                other_rep = features[reps[other_root]]
                src_cos = 1.0 - float(dist)
                if similarity_family_match(rep, other_rep, src_cos):
                    uf.union(root_pos[root], root_pos[other_root])

    families: Dict[int, List[int]] = defaultdict(list)
    for root in roots:
        families[roots[uf.find(root_pos[root])]].append(root)
    return families, rep_embeddings


def feature_distance(a: FeatureRecord, b: FeatureRecord, src_cos: float) -> float:
    byte = hamming_distance_u64(a.bytecode_simhash, b.bytecode_simhash) / 64.0
    source = 1.0 - src_cos
    iface = 1.0 - jaccard_similarity(a.signatures, b.signatures)
    return 0.25 * byte + 0.55 * source + 0.20 * iface


def diverse_select_cluster_roots(
    features: List[FeatureRecord],
    entries: List[Tuple[int, List[int], int]],
    keep_limit: int,
    representative_embeddings: Dict[int, object],
) -> Set[int]:
    if keep_limit >= len(entries):
        return {root for root, _, _ in entries}
    ranked = sorted(
        entries,
        key=lambda item: (len(item[1]), features[item[2]].score, -features[item[2]].line_no),
        reverse=True,
    )
    selected: List[Tuple[int, List[int], int]] = [ranked[0]]
    selected_roots: Set[int] = {ranked[0][0]}
    while len(selected) < keep_limit:
        best_entry = None
        best_score = None
        for entry in ranked:
            root, _, rep_idx = entry
            if root in selected_roots:
                continue
            rep = features[rep_idx]
            rep_vec = representative_embeddings.get(rep_idx)
            if rep_vec is not None:
                dist_candidates = [
                    feature_distance(
                        rep,
                        features[sel_rep_idx],
                        float(rep_vec @ representative_embeddings[sel_rep_idx]),
                    )
                    for _, _, sel_rep_idx in selected
                    if sel_rep_idx in representative_embeddings
                ]
                if dist_candidates:
                    min_dist = min(dist_candidates)
                else:
                    min_dist = min(
                        hamming_distance_u64(rep.bytecode_simhash, features[sel_rep_idx].bytecode_simhash) / 64.0
                        for _, _, sel_rep_idx in selected
                    )
            else:
                min_dist = min(
                    hamming_distance_u64(rep.bytecode_simhash, features[sel_rep_idx].bytecode_simhash) / 64.0
                    for _, _, sel_rep_idx in selected
                )
            score = (min_dist, len(entry[1]), features[rep_idx].score, -features[rep_idx].line_no)
            if best_score is None or score > best_score:
                best_score = score
                best_entry = entry
        assert best_entry is not None
        selected.append(best_entry)
        selected_roots.add(best_entry[0])
    return selected_roots


def cluster_representative(features: List[FeatureRecord], member_indices: Sequence[int]) -> int:
    best = member_indices[0]
    for idx in member_indices[1:]:
        if is_better_representative(features[idx], features[best]):
            best = idx
    return best


def balance_similarity_families(
    features: List[FeatureRecord],
    near_clusters: Dict[int, List[int]],
    similarity_families: Dict[int, List[int]],
    representative_embeddings: Dict[int, object],
    family_cap: int,
    small_family_keep_all: int,
    max_representatives_per_cluster: int,
) -> Dict[int, List[int]]:
    retained: Dict[int, List[int]] = {}
    for family_root, cluster_roots in similarity_families.items():
        entries = [
            (root, list(near_clusters[root]), cluster_representative(features, near_clusters[root]))
            for root in cluster_roots
        ]
        if len(entries) <= small_family_keep_all:
            for root, members, _ in entries:
                retained[root] = select_cluster_representatives(
                    features, members, max_representatives_per_cluster
                )
            continue
        keep_limit = min(family_cap, max(small_family_keep_all, int(math.ceil(math.sqrt(len(entries)) * 2.0))))
        selected_roots = diverse_select_cluster_roots(
            features, entries, keep_limit, representative_embeddings
        )
        for root, members, _ in entries:
            if root not in selected_roots:
                continue
            retained[root] = select_cluster_representatives(
                features, members, max_representatives_per_cluster
            )
    return retained


def select_cluster_representatives(
    features: List[FeatureRecord], member_indices: Sequence[int], max_representatives_per_cluster: int
) -> List[int]:
    ordered = sorted(
        member_indices,
        key=lambda idx: (features[idx].score, -len(features[idx].signatures), -features[idx].line_no),
        reverse=True,
    )
    return ordered[: max(1, max_representatives_per_cluster)]


def assign_splits_by_cluster(
    features: List[FeatureRecord],
    clusters: Dict[int, List[int]],
    retained: Dict[int, List[int]],
    split_weights: Dict[str, float],
) -> Dict[int, str]:
    split_weights = {name: weight for name, weight in split_weights.items() if weight > 0}
    if not split_weights:
        raise ValueError("at least one split must have positive weight")
    totals = {name: 0 for name in split_weights}
    cluster_order = sorted(
        clusters.items(),
        key=lambda item: (
            len(retained.get(item[0], [])),
            len(item[1]),
            stable_hash(str(item[0])),
        ),
        reverse=True,
    )
    assignments: Dict[int, str] = {}
    total_weight = sum(split_weights.values())
    for root, members in cluster_order:
        cluster_size = max(1, len(retained.get(root, [])))
        best_split = None
        best_score = None
        for split_name, target in split_weights.items():
            projected = totals[split_name] + cluster_size
            score = projected / max(target, 1e-9)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assert best_split is not None
        totals[best_split] += cluster_size
        assignments[root] = best_split
    return assignments


def write_outputs(
    input_path: Path,
    features: List[FeatureRecord],
    group_members: Dict[int, List[int]],
    near_clusters: Dict[int, List[int]],
    similarity_families: Dict[int, List[int]],
    retained: Dict[int, List[int]],
    split_assignments: Dict[int, str],
    metadata_output: Path,
    retained_output: Path,
    summary_output: Path,
) -> None:
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    retained_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    root_to_cluster_id = {root: idx for idx, root in enumerate(sorted(near_clusters))}
    row_to_cluster_root: Dict[int, int] = {}
    for root, member_indices in near_clusters.items():
        for rep_idx in member_indices:
            row_to_cluster_root[rep_idx] = root

    exact_group_to_cluster_root: Dict[int, int] = {}
    for root, member_indices in near_clusters.items():
        for rep_idx in member_indices:
            exact_group_to_cluster_root[features[rep_idx].exact_group_id] = root

    near_root_to_family_root: Dict[int, int] = {}
    for family_root, roots in similarity_families.items():
        for root in roots:
            near_root_to_family_root[root] = family_root

    retained_line_numbers: Set[int] = set()
    cluster_sizes: Dict[int, int] = {root: len(member_indices) for root, member_indices in near_clusters.items()}
    family_counts: Counter[str] = Counter()
    for root, selected in retained.items():
        if not selected:
            continue
        family_root = near_root_to_family_root.get(root, root)
        family_counts[f"simfam:{family_root}"] += len(selected)
        for idx in selected:
            retained_line_numbers.add(features[idx].line_no)
            features[idx].retained_final = True

    with metadata_output.open("w", encoding="utf-8") as meta_f:
        for record in features:
            root = exact_group_to_cluster_root.get(record.exact_group_id)
            near_cluster = root_to_cluster_id[root] if root is not None else -1
            split = split_assignments.get(root, "")
            record.near_cluster_id = near_cluster
            family_root = near_root_to_family_root.get(root)
            record.family_cluster_id = f"simfam:{family_root}" if family_root is not None else ""
            record.split = split
            meta = {
                "line_no": record.line_no,
                "address": record.address,
                "contract_name": record.contract_name,
                "compile_success": record.compile_success,
                "exact_duplicate_group": record.exact_group_id,
                "exact_duplicate_group_size": len(group_members.get(record.exact_group_id, [])),
                "exact_representative": record.retained_exact,
                "near_duplicate_cluster": near_cluster,
                "near_duplicate_cluster_size": cluster_sizes.get(root, 0),
                "family_cluster": record.family_cluster_id,
                "split": split,
                "retained": record.retained_final,
                "source_train_hash": record.source_train_hash,
                "source_similarity_hash": record.source_similarity_hash,
                "bytecode_logic_hash": record.bytecode_logic_hash,
                "interface_hash": record.interface_hash,
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    with retained_output.open("w", encoding="utf-8") as out_f:
        for record in features:
            if record.line_no not in retained_line_numbers:
                continue
            raw = read_line_at_offset(input_path, record.offset)
            row = json.loads(raw)
            root = exact_group_to_cluster_root.get(record.exact_group_id)
            near_cluster = root_to_cluster_id[root] if root is not None else -1
            row["exact_duplicate_group"] = record.exact_group_id
            row["near_duplicate_cluster"] = near_cluster
            row["family_cluster"] = record.family_cluster_id
            row["split"] = split_assignments.get(root, "")
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_family_clusters = [
        {"family_cluster": cluster_id, "retained_rows": count}
        for cluster_id, count in family_counts.most_common(20)
    ]
    summary = {
        "rows_seen": len(features),
        "compile_success_rows": sum(1 for record in features if record.compile_success),
        "exact_duplicate_groups": len(group_members),
        "near_duplicate_clusters": len(near_clusters),
        "similarity_family_clusters": len(similarity_families),
        "retained_rows": len(retained_line_numbers),
        "top_retained_family_clusters": top_family_clusters,
        "split_counts_retained": dict(
            Counter(features[idx].split for selected in retained.values() for idx in selected).most_common()
        ),
    }
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    input_path: Path,
    metadata_output: Path,
    retained_output: Path,
    summary_output: Path,
    limit: Optional[int],
    family_cap: int,
    small_family_keep_all: int,
    max_representatives_per_cluster: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    log_every: int,
    embedding_model: str,
) -> None:
    features: List[FeatureRecord] = []
    compile_success_rows = 0
    for line_no, offset in iter_offsets(input_path, limit):
        raw = read_line_at_offset(input_path, offset)
        row = json.loads(raw)
        feature = compute_features(row, line_no=line_no, offset=offset)
        features.append(feature)
        if feature.compile_success:
            compile_success_rows += 1
        if log_every > 0 and line_no % log_every == 0:
            print(
                f"[features] processed={line_no} compile_success={compile_success_rows}",
                flush=True,
            )

    _, group_members, group_representative = build_exact_groups(features)
    if log_every > 0:
        print(
            f"[exact] groups={len(group_members)} representatives={len(group_representative)}",
            flush=True,
        )
    representative_indices = sorted(group_representative.values())
    near_clusters = build_near_duplicate_clusters(features, representative_indices)
    if log_every > 0:
        print(f"[near] clusters={len(near_clusters)}", flush=True)
    similarity_families, representative_embeddings = build_similarity_family_clusters(
        input_path, features, near_clusters, embedding_model
    )
    if log_every > 0:
        print(f"[families] clusters={len(similarity_families)}", flush=True)
    retained = balance_similarity_families(
        features,
        near_clusters,
        similarity_families,
        representative_embeddings,
        family_cap=family_cap,
        small_family_keep_all=small_family_keep_all,
        max_representatives_per_cluster=max_representatives_per_cluster,
    )
    if log_every > 0:
        print(
            f"[balance] retained_clusters={len(retained)} retained_rows={sum(len(v) for v in retained.values())}",
            flush=True,
        )
    split_assignments = assign_splits_by_cluster(
        features,
        near_clusters,
        retained,
        split_weights={"train": train_ratio, "val": val_ratio, "test": test_ratio},
    )
    write_outputs(
        input_path=input_path,
        features=features,
        group_members=group_members,
        near_clusters=near_clusters,
        similarity_families=similarity_families,
        retained=retained,
        split_assignments=split_assignments,
        metadata_output=metadata_output,
        retained_output=retained_output,
        summary_output=summary_output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate and debias a smart-contract source/bytecode dataset using exact and near-duplicate clustering."
    )
    parser.add_argument("--input", type=Path, default=Path("data/contracts_simplified_solc_v1.jsonl"))
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/contracts_simplified_solc_v1.dedup_metadata.jsonl"),
    )
    parser.add_argument(
        "--retained-output",
        type=Path,
        default=Path("data/contracts_simplified_solc_v1.dedup_retained.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/contracts_simplified_solc_v1.dedup_summary.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--family-cap", type=int, default=200)
    parser.add_argument("--small-family-keep-all", type=int, default=100)
    parser.add_argument("--max-representatives-per-cluster", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.family_cap <= 0:
        raise SystemExit("--family-cap must be > 0")
    if args.small_family_keep_all <= 0:
        raise SystemExit("--small-family-keep-all must be > 0")
    if args.max_representatives_per_cluster <= 0:
        raise SystemExit("--max-representatives-per-cluster must be > 0")
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise SystemExit("split ratios must sum to 1.0")

    run_pipeline(
        input_path=args.input,
        metadata_output=args.metadata_output,
        retained_output=args.retained_output,
        summary_output=args.summary_output,
        limit=args.limit,
        family_cap=args.family_cap,
        small_family_keep_all=args.small_family_keep_all,
        max_representatives_per_cluster=args.max_representatives_per_cluster,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        log_every=args.log_every,
        embedding_model=args.embedding_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
