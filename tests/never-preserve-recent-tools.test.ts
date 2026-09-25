import { test } from "node:test";
import assert from "node:assert/strict";
import { createCore, createInitialState, defaultConfig, coveredMessageIds } from "acp-kernel";
import { anthropicToCore, type AnthropicRequestBody } from "acp-kernel/wire";
import { parseCompressSettings } from "../src/config.ts";
import { mergeCompress, resolveRequestConfig } from "../src/compress-settings.ts";

// --- Config plumbing (#1277) ----------------------------------------------

test("parseCompressSettings accepts a valid neverPreserveRecentTools array (empty = escape hatch)", () => {
    const s = parseCompressSettings({ neverPreserveRecentTools: [" decompress ", "read*"] });
    assert.deepEqual(s?.neverPreserveRecentTools, ["decompress", "read*"]);
    // Unlike protectedTools/protectedLatestTools, an explicit [] is VALID:
    // it excludes nothing, giving fresh read results recent-zone protection
    // (the #1198 read-loop escape hatch).
    assert.deepEqual(parseCompressSettings({ neverPreserveRecentTools: [] })?.neverPreserveRecentTools, []);
});

test("parseCompressSettings rejects malformed neverPreserveRecentTools", () => {
    assert.equal(parseCompressSettings({ neverPreserveRecentTools: "read" }), undefined);
    assert.equal(parseCompressSettings({ neverPreserveRecentTools: [42] }), undefined);
    assert.equal(parseCompressSettings({ neverPreserveRecentTools: [""] }), undefined);
    assert.equal(parseCompressSettings({ neverPreserveRecentTools: ["read", null] }), undefined);
});

test("mergeCompress: neverPreserveRecentTools deepest level wins, [] replaces rather than clears", () => {
    const merged = mergeCompress(
        { neverPreserveRecentTools: ["read", "bash"], tiers: true },
        { neverPreserveRecentTools: ["decompress", "search_context", "bash"] },
        { tiers: false },
    );
    assert.deepEqual(merged.neverPreserveRecentTools, ["decompress", "search_context", "bash"]);
    assert.equal(merged.tiers, false);
    // An explicit [] at a deeper level must NOT fall back to the shallower
    // list — [] is a meaningful value (protect everything in the recent zone).
    assert.deepEqual(
        mergeCompress({ neverPreserveRecentTools: ["read"] }, { neverPreserveRecentTools: [] }, undefined).neverPreserveRecentTools,
        [],
    );
    assert.equal(mergeCompress(undefined, undefined, undefined).neverPreserveRecentTools, undefined);
});

test("resolveRequestConfig passes neverPreserveRecentTools onto the kernel Config verbatim", () => {
    const base = defaultConfig(200000);
    assert.deepEqual(
        resolveRequestConfig(base, {}, undefined, "claude-test", 200000, {
            neverPreserveRecentTools: ["decompress", "search_context", "bash"],
        }).neverPreserveRecentTools,
        ["decompress", "search_context", "bash"],
    );
    assert.deepEqual(
        resolveRequestConfig(base, {}, undefined, "claude-test", 200000, { neverPreserveRecentTools: [] }).neverPreserveRecentTools,
        [],
    );
    // Unset stays undefined so the kernel built-in default list governs.
    assert.equal(resolveRequestConfig(base, {}, undefined, "claude-test", 200000, {}).neverPreserveRecentTools, undefined);
});

// --- Kernel end-to-end: recent-zone membership flips with the list ---------

function buildBody(): AnthropicRequestBody {
    const body: AnthropicRequestBody = { model: "claude-test", messages: [] };
    const push = (role: "user" | "assistant", content: unknown): void => {
        body.messages.push({ role, content: content as never });
    };
    push("user", "message 0 start of a long working session");
    push("assistant", [{ type: "tool_use", id: "read-1", name: "read", input: { path: "src/server.ts" } }]);
    push("user", [{ type: "tool_result", tool_use_id: "read-1", content: `file body ${"y".repeat(400)}` }]);
    for (let i = 0; i < 8; i++) {
        push(i % 2 === 0 ? "user" : "assistant", `tail message ${i} ${"x".repeat(200)}`);
    }
    return body;
}

function foldReadTurn(neverList: string[] | undefined): { errors: string[]; blocksCreated: number; readCovered: boolean } {
    const core = createCore();
    const state = createInitialState();
    const config = {
        ...defaultConfig(200000),
        preserveRecentMessages: 12,
        preserveRecentTokens: 0,
        compress: { ...defaultConfig(200000).compress, minCompressRange: 0 },
        ...(neverList !== undefined ? { neverPreserveRecentTools: neverList } : {}),
    };
    const { msgs } = anthropicToCore(buildBody());
    const turn = core.processTurn({ messages: msgs, state, config, tokenCount: 9999, renderTags: "text-only" });
    const readCall = msgs.find((m) => m.contentType === "tool-call" && m.toolCallId === "read-1")!;
    const readResult = msgs.find((m) => m.contentType === "tool-result" && m.toolCallId === "read-1")!;
    const res = core.applyCompression({
        ranges: [{
            startRef: turn.state.messageRefs.byRaw[readCall.id]!,
            endRef: turn.state.messageRefs.byRaw[readResult.id]!,
            summary: "fold the freshly-read file body away".repeat(3),
        }],
        state: turn.state,
        config,
        messages: turn.messages,
    });
    const covered = coveredMessageIds(res.state);
    return {
        errors: res.result.errors,
        blocksCreated: res.result.blocksCreated,
        readCovered: covered.has(readResult.id),
    };
}

test("kernel default list: fresh read inside the recent window folds immediately (#1198 mechanism)", () => {
    const out = foldReadTurn(undefined);
    assert.equal(out.errors.length, 0, `no errors: ${out.errors.join("; ")}`);
    assert.ok(out.readCovered, "read result folded — excluded from the recent zone by the kernel default");
});

test("read removed from the list: the fresh read pair gains recent-zone protection (#1277)", () => {
    const out = foldReadTurn(["decompress", "search_context", "bash"]);
    assert.equal(out.blocksCreated, 0, "nothing folds — the whole range is protected");
    assert.match(out.errors[0] ?? "", /protected/i);
    assert.ok(!out.readCovered, "read result untouched");
});

test("empty list []: same protection as removing read (max-protection escape hatch)", () => {
    const out = foldReadTurn([]);
    assert.equal(out.blocksCreated, 0);
    assert.match(out.errors[0] ?? "", /protected/i);
    assert.ok(!out.readCovered);
});
