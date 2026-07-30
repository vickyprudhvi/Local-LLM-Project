#!/usr/bin/env node
// Phase F test fixture: a REAL stdio JSON-RPC MCP server exposing the filesystem
// tool surface, used so installation/validation tests exercise a genuine child
// process without any network access. Allowed roots come from argv (like the
// official server). It never reads or writes outside those roots.
"use strict";

const fs = require("fs");
const path = require("path");
const readline = require("readline");

const PROTOCOL_VERSION = "2024-11-05";
const allowedRoots = process.argv.slice(2).map((p) => path.resolve(p));

const OBJECT_SCHEMA = (props, required) => ({
  type: "object",
  properties: props || {},
  required: required || [],
});

// Deliberately advertises permissions the LOCAL policy must ignore.
const TOOLS = [
  { name: "list_allowed_directories", description: "List allowed roots.",
    inputSchema: OBJECT_SCHEMA({}), annotations: { permission: "read" } },
  { name: "list_directory", description: "List a directory.",
    inputSchema: OBJECT_SCHEMA({ path: { type: "string" } }, ["path"]),
    annotations: { permission: "read" } },
  { name: "read_text_file", description: "Read a text file.",
    inputSchema: OBJECT_SCHEMA({ path: { type: "string" } }, ["path"]),
    annotations: { permission: "read" } },
  { name: "write_file", description: "Write a text file.",
    inputSchema: OBJECT_SCHEMA({ path: { type: "string" }, content: { type: "string" } }, ["path"]),
    // Advertised as read-only on purpose: local policy must still make it WRITE.
    annotations: { permission: "read" } },
  { name: "move_file", description: "Move a file.",
    inputSchema: OBJECT_SCHEMA({ source: { type: "string" }, destination: { type: "string" } }),
    annotations: { permission: "read" } },
  { name: "edit_file", description: "Edit a file.",
    inputSchema: OBJECT_SCHEMA({ path: { type: "string" } }),
    annotations: { permission: "read" } },
  { name: "get_file_info", description: "Stat a file.",
    inputSchema: OBJECT_SCHEMA({ path: { type: "string" } }),
    annotations: { permission: "read" } },
  { name: "search_files", description: "Search for files.",
    inputSchema: OBJECT_SCHEMA({ path: { type: "string" }, pattern: { type: "string" } }),
    annotations: { permission: "read" } },
  { name: "undocumented_extra_tool", description: "Not in the catalog policy.",
    inputSchema: OBJECT_SCHEMA({}), annotations: { permission: "write" } },
];

function inAllowedRoot(target) {
  const resolved = path.resolve(target);
  return allowedRoots.some(
    (root) => resolved === root || resolved.startsWith(root + path.sep)
  );
}

function ok(structured) {
  return {
    content: [{ type: "text", text: JSON.stringify(structured) }],
    structuredContent: structured,
    isError: false,
  };
}

function toolError(message) {
  return { content: [{ type: "text", text: message }], isError: true };
}

function callTool(name, args) {
  args = args || {};
  if (name === "list_allowed_directories") {
    return ok({ directories: allowedRoots });
  }
  if (name === "list_directory") {
    if (!inAllowedRoot(args.path)) return toolError("Path outside allowed roots.");
    return ok({ entries: fs.readdirSync(path.resolve(args.path)) });
  }
  if (name === "read_text_file") {
    if (!inAllowedRoot(args.path)) return toolError("Path outside allowed roots.");
    return ok({ content: fs.readFileSync(path.resolve(args.path), "utf8") });
  }
  if (name === "write_file") {
    if (!inAllowedRoot(args.path)) return toolError("Path outside allowed roots.");
    fs.writeFileSync(path.resolve(args.path), args.content == null ? "" : args.content, "utf8");
    return ok({ written: true, path: args.path });
  }
  if (name === "get_file_info") {
    if (!inAllowedRoot(args.path)) return toolError("Path outside allowed roots.");
    const stat = fs.statSync(path.resolve(args.path));
    return ok({ size: stat.size, is_directory: stat.isDirectory() });
  }
  if (name === "search_files") {
    if (!inAllowedRoot(args.path)) return toolError("Path outside allowed roots.");
    return ok({ matches: [] });
  }
  if (name === "move_file" || name === "edit_file" || name === "undocumented_extra_tool") {
    // Reachable only if a local policy wrongly enabled it — tests assert it is not.
    return ok({ performed: name });
  }
  return null;
}

function handle(message) {
  const id = message.id;
  const params = message.params || {};
  switch (message.method) {
    case "initialize":
      return { jsonrpc: "2.0", id, result: {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: { name: "fake-filesystem-server", version: "0.0.0-test" },
      } };
    case "notifications/initialized":
      return null;
    case "tools/list":
      return { jsonrpc: "2.0", id, result: { tools: TOOLS } };
    case "tools/call": {
      let result;
      try {
        result = callTool(params.name, params.arguments);
      } catch (e) {
        return { jsonrpc: "2.0", id, result: toolError(String(e && e.message)) };
      }
      if (result === null) {
        return { jsonrpc: "2.0", id, error: { code: -32601, message: "Unknown tool: " + params.name } };
      }
      return { jsonrpc: "2.0", id, result };
    }
    default:
      if (id === undefined || id === null) return null;
      return { jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found" } };
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let message;
  try {
    message = JSON.parse(trimmed);
  } catch (e) {
    return;
  }
  const response = handle(message);
  if (response !== null) {
    process.stdout.write(JSON.stringify(response) + "\n");
  }
});
