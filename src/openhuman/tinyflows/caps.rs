//! The capability seam: five adapters implementing `tinyflows::caps` traits
//! over real OpenHuman services.
//!
//! Each tinyflows integration node hands its **whole** `node.config` to the
//! matching trait method — the adapter interprets a free-form JSON value the
//! flow author wrote, pulling a connection ref out of `config["connection_ref"]`
//! where relevant. See `my_docs/ohxtf/b1-engine-seam-domain/04-capability-seam.md`
//! for the source-verified node → trait contract this mirrors.
//!
//! All host errors are mapped to `tinyflows::error::EngineError::Capability`,
//! per the crate's contract (`caps` traits return `tinyflows::error::Result`).

use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use serde_json::{json, Value};
use tinyagents::graph::SqliteCheckpointer;
use tinyflows::caps::{
    Capabilities, CodeLanguage, CodeRunner, HttpClient, LlmProvider, StateStore, ToolInvoker,
};
use tinyflows::error::{EngineError, Result};

use crate::openhuman::agent::harness::definition::SandboxMode;
use crate::openhuman::composio::client::{
    create_composio_client, direct_execute, ComposioClientKind,
};
use crate::openhuman::config::{Config, HttpRequestConfig};
use crate::openhuman::flows;
use crate::openhuman::inference::provider::{
    create_chat_provider, ChatMessage, ChatRequest, UsageInfo,
};
use crate::openhuman::sandbox::{execute_in_sandbox, resolve_sandbox_policy};
use crate::openhuman::security::SecurityPolicy;
use crate::openhuman::tools::traits::Tool as _;
use crate::openhuman::tools::HttpRequestTool;

/// Maps a `UsageInfo` (not `Serialize`) into a JSON value field-by-field, so
/// [`OpenHumanLlm::complete`] can surface it in its response `Value` without
/// requiring an upstream `Serialize` impl change.
fn usage_to_json(usage: &Option<UsageInfo>) -> Value {
    match usage {
        None => Value::Null,
        Some(u) => json!({
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "context_window": u.context_window,
            "cached_input_tokens": u.cached_input_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
            "reasoning_tokens": u.reasoning_tokens,
            "charged_amount_usd": u.charged_amount_usd,
        }),
    }
}

/// [`LlmProvider`] adapter over OpenHuman's inference stack
/// (`src/openhuman/inference/provider/`).
///
/// The `agent` node is single-completion in tinyflows 0.2 (no tool-calling
/// loop, no sub-ports), so `complete` performs exactly one `provider.chat`
/// call and returns its result — no agent loop is driven here.
pub struct OpenHumanLlm {
    pub config: Arc<Config>,
}

#[async_trait]
impl LlmProvider for OpenHumanLlm {
    async fn complete(&self, request: Value, conn: Option<&str>) -> Result<Value> {
        if let Some(c) = conn {
            // B1 does not resolve `connection_ref` to a specific BYOK account —
            // `create_chat_provider` picks the configured provider for `role`.
            tracing::debug!(target: "flows", conn = %c, "[flows] llm conn (not resolved in B1)");
        }

        let role = request
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("summarization");
        let temperature = request
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.7);
        let max_tokens = request
            .get("max_tokens")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok());

        let messages: Vec<ChatMessage> = match request.get("messages").and_then(Value::as_array) {
            Some(entries) if !entries.is_empty() => entries
                .iter()
                .filter_map(|entry| {
                    let content = entry.get("content").and_then(Value::as_str)?.to_string();
                    let role = entry.get("role").and_then(Value::as_str).unwrap_or("user");
                    Some(match role {
                        "system" => ChatMessage::system(content),
                        "assistant" => ChatMessage::assistant(content),
                        "tool" => ChatMessage::tool(content),
                        _ => ChatMessage::user(content),
                    })
                })
                .collect(),
            _ => {
                let prompt = request
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                vec![ChatMessage::user(prompt)]
            }
        };

        tracing::debug!(
            target: "flows",
            role,
            message_count = messages.len(),
            "[flows] llm.complete: dispatching agent-node completion"
        );

        let (provider, model) = create_chat_provider(role, &self.config)
            .map_err(|e| EngineError::Capability(e.to_string()))?;

        let response = provider
            .chat(
                ChatRequest {
                    messages: &messages,
                    tools: None,
                    stream: None,
                    max_tokens,
                },
                &model,
                temperature,
            )
            .await
            .map_err(|e| EngineError::Capability(e.to_string()))?;

        Ok(json!({
            "text": response.text,
            "tool_calls": response.tool_calls,
            "usage": usage_to_json(&response.usage),
            "reasoning_content": response.reasoning_content,
        }))
    }
}

/// Parses a `"composio:<toolkit>:<connection_id>"` `connection_ref` (see the
/// node catalog, `my_docs/ohxtf/commons/12-node-catalog-0.2.md`) and returns
/// the trailing connection id segment. Values that don't match this shape
/// return `None` — the caller logs and falls back to the ambient session
/// account (only Direct mode can actually forward the id today; see
/// [`OpenHumanTools::invoke`]'s doc for the Backend-mode gap this leaves
/// open).
pub(crate) fn composio_connection_id(conn: &str) -> Option<&str> {
    let rest = conn.strip_prefix("composio:")?;
    let id = rest.rsplit(':').next()?;
    (!id.is_empty()).then_some(id)
}

/// Parses a `"http_cred:<name>"` `connection_ref` for [`OpenHumanHttp`]. No
/// host-side HTTP credential store exists yet — this only extracts the name
/// so the adapter can log a clear, actionable warning instead of silently
/// ignoring the reference. See [`OpenHumanHttp::request`]'s doc.
pub(crate) fn http_cred_name(conn: &str) -> Option<&str> {
    let name = conn.strip_prefix("http_cred:")?.trim();
    (!name.is_empty()).then_some(name)
}

/// Strict, deny-by-default curation check for flow `tool_call` nodes (issue
/// B2 finding #2).
///
/// This is intentionally **stricter** than
/// `memory_sync::composio::providers::is_action_visible_with_pref` — the
/// helper the normal agent tool-call loop uses. That helper is permissive by
/// design for a toolkit it doesn't recognize: it falls back to the
/// `classify_unknown` heuristic and lets the slug through (scope-gated), and
/// treats a prefix-less slug as unconditionally visible. That's safe in the
/// agent loop because the model only ever sees slugs the *backend itself*
/// returned from live tool discovery (`composio_list_tools`) — there is no
/// path for the model to invent a slug that reaches this check. A flow's
/// `tool_call.slug`, by contrast, is a free-form string the flow *author*
/// typed when building the graph; it never round-trips through Composio
/// discovery before `invoke` is called. So here a slug is allowed **only**
/// if it resolves to a real, known toolkit AND is present in that toolkit's
/// curated catalog:
/// - `toolkit_from_slug` fails to extract anything (empty/blank slug) → reject.
/// - the extracted toolkit has no registered provider curated list AND no
///   static `catalog_for_toolkit` entry (i.e. it isn't one of OpenHuman's
///   known/curated toolkits at all — including a made-up prefix like
///   `madeupkit`, or a prefix-less slug like `noop` which `toolkit_from_slug`
///   degrades to treating as its own single-segment "toolkit") → reject.
/// - the toolkit has a catalog but `slug` isn't one of its entries → reject.
/// - otherwise, apply the same per-user read/write/admin scope preference
///   the agent loop uses (`UserScopePref::allows`).
///
/// // TODO(0.3): this hard-rejects any *real* Composio toolkit that simply
/// // isn't in the static `catalog_for_toolkit` map yet (there is no
/// // host-side, offline way to ask "is this actually a valid Composio
/// // toolkit/action" beyond the curated catalogs OpenHuman ships). That's
/// // an accepted trade-off for a genuine allowlist rather than a residual
/// // gap to silently work around — extending `catalog_for_toolkit` (or, if
/// // a live catalog lookup becomes available, consulting it here) is how a
/// // newly-supported toolkit gets flow tool-call support.
async fn is_curated_flow_tool(slug: &str) -> bool {
    use crate::openhuman::memory_sync::composio::providers::{
        catalog_for_toolkit, find_curated, get_provider, load_user_scope_or_default,
        toolkit_from_slug,
    };

    let Some(toolkit) = toolkit_from_slug(slug) else {
        return false;
    };
    let catalog = get_provider(&toolkit)
        .and_then(|p| p.curated_tools())
        .or_else(|| catalog_for_toolkit(&toolkit));
    let Some(catalog) = catalog else {
        return false;
    };
    let Some(curated) = find_curated(catalog, slug) else {
        return false;
    };
    let pref = load_user_scope_or_default(&toolkit).await;
    pref.allows(curated.scope)
}

/// [`ToolInvoker`] adapter over Composio (`src/openhuman/composio/client.rs`).
///
/// **B2 (closes two B1 deviations, see
/// `my_docs/ohxtf/b2-triggers-trust/01-triggers-and-trust.md` §4-5):**
/// - **Curation + scope (hard allowlist)**: every call is checked against
///   [`is_curated_flow_tool`] — a deny-by-default gate that only allows a
///   slug resolving to a *known, curated* toolkit action, unlike the general
///   agent tool-call path's more permissive
///   `memory_sync::composio::providers::is_action_visible_with_pref` (see
///   [`is_curated_flow_tool`]'s doc for why the two differ). A non-curated /
///   unrecognized / out-of-scope slug is rejected with
///   `EngineError::Capability("tool not permitted: <slug>")` before any
///   Composio call. **As of tinyflows 0.3 this is load-bearing, not merely
///   defense-in-depth**: integration-node config (including `slug`) is now
///   `=`-expression evaluated against upstream/trigger data before `invoke`,
///   so a trigger payload *can* influence which tool a `=`-derived slug
///   resolves to. The curation gate runs on the **resolved** slug (verified:
///   a `=item.tool`-derived unknown slug is rejected here before Composio),
///   constraining any data-derived tool to the user's curated, in-scope,
///   connected set — and it still closes the case where an author hand-types
///   an arbitrary/typo'd slug.
/// - **connection_ref**: `conn` (`"composio:<toolkit>:<connection_id>"`) is
///   now parsed and forwarded to `direct_execute` (Composio Direct mode).
///   Backend mode's `execute_tool` still has no per-call account-scoping
///   path — that's a backend API gap, not something this seam can close
///   alone — so a `connection_ref` under Backend mode logs a warning and
///   falls back to the ambient signed-in account (documented stub; see
///   `composio_connection_id`).
/// - **Trust gate**: invocation is also routed through the OpenHuman
///   `ApprovalGate` (mirrors `tinyagents/middleware.rs::ApprovalSecurityMiddleware`)
///   before dispatch, closing the Codex P1 finding that flow tool nodes
///   bypassed the Network/tool approval gate entirely. `ops::flows_run` /
///   `flows_resume` scope a `TrustedAutomation { Workflow }` origin around
///   the whole run, so the gate either auto-allows (pre-declared trust root)
///   or — when the flow's `require_approval` is set — parks for a real
///   decision. No gate installed (unit tests, some hosts) means no gating,
///   same as the existing agent tool-loop middleware.
///
/// // SECURITY NOTE (tinyflows 0.3, now the pinned version): integration nodes
/// // `=`-resolve config from upstream/trigger data, so a trigger-driven flow
/// // whose `slug`/`url` is `=`-derived lets untrusted trigger data pick *which*
/// // curated + in-scope + connected tool/endpoint runs (blast radius bounded by
/// // the curation + scope + connection checks above and the approval gate).
/// // For such flows authors should set `require_approval`. FOLLOW-UP: auto-force
/// // approval when a trigger-driven run's tool/http config contains `=`-exprs.
pub struct OpenHumanTools {
    pub config: Arc<Config>,
}

#[async_trait]
impl ToolInvoker for OpenHumanTools {
    async fn invoke(&self, slug: &str, args: Value, conn: Option<&str>) -> Result<Value> {
        // Curation + scope gate — hard allowlist (see [`is_curated_flow_tool`]'s
        // doc for why this differs from the general agent tool-call path).
        // Runs before anything else — a rejected slug never reaches the
        // composio client at all.
        if !is_curated_flow_tool(slug).await {
            tracing::warn!(
                target: "flows",
                %slug,
                "[flows] tool_call: rejected — not a recognized curated toolkit action, or out \
                 of the user's configured scope"
            );
            return Err(EngineError::Capability(format!(
                "tool not permitted: {slug}"
            )));
        }

        // Approval gate (see the struct doc). Mirrors
        // `tinyagents/middleware.rs::ApprovalSecurityMiddleware::wrap_tool`'s
        // shape exactly: compute summary/redacted args only when a gate is
        // installed, deny short-circuits before any composio call, allow
        // records an audit id to close out after the call resolves.
        let mut audit_id: Option<String> = None;
        if let Some(gate) = crate::openhuman::approval::ApprovalGate::try_global() {
            let summary = crate::openhuman::approval::summarize_action(slug, &args);
            let redacted = crate::openhuman::approval::redact_args(&args);
            let (outcome, request_id) = gate.intercept_audited(slug, &summary, redacted).await;
            match outcome {
                crate::openhuman::approval::GateOutcome::Deny { reason } => {
                    return Err(EngineError::Capability(reason));
                }
                crate::openhuman::approval::GateOutcome::Allow => audit_id = request_id,
            }
        }

        let kind = create_composio_client(&self.config)
            .map_err(|e| EngineError::Capability(e.to_string()))?;
        let args_opt = if args.is_null() { None } else { Some(args) };
        let connection_id = conn.and_then(composio_connection_id);

        tracing::debug!(
            target: "flows",
            %slug,
            mode = kind.mode(),
            has_connection_ref = connection_id.is_some(),
            "[flows] tool_call: invoking composio tool"
        );

        let response = match kind {
            ComposioClientKind::Backend(client) => {
                if connection_id.is_some() {
                    tracing::warn!(
                        target: "flows",
                        %slug,
                        "[flows] tool_call: connection_ref set but backend mode has no per-call \
                         account-scoping path yet — using the ambient session account \
                         (documented stub, see caps.rs's OpenHumanTools doc)"
                    );
                }
                client
                    .execute_tool(slug, args_opt)
                    .await
                    .map_err(|e| EngineError::Capability(e.to_string()))
            }
            ComposioClientKind::Direct(tool) => direct_execute(
                &tool,
                slug,
                args_opt,
                &self.config.composio.entity_id,
                connection_id,
            )
            .await
            .map_err(|e| EngineError::Capability(e.to_string())),
        };

        if let Some(id) = audit_id {
            if let Some(gate) = crate::openhuman::approval::ApprovalGate::try_global() {
                let exec = if response.is_ok() {
                    crate::openhuman::approval::ExecutionOutcome::Success
                } else {
                    crate::openhuman::approval::ExecutionOutcome::Failure
                };
                gate.record_execution(
                    &id,
                    exec,
                    response.as_ref().err().map(ToString::to_string).as_deref(),
                );
            }
        }

        serde_json::to_value(response?).map_err(|e| EngineError::Capability(e.to_string()))
    }
}

/// [`HttpClient`] adapter over `HttpRequestTool`
/// (`src/openhuman/tools/impl/network/http_request.rs`). Allowlist + DNS-rebind
/// guard live inside `execute`, so this adapter gets them for free.
///
/// **B2:** also routes through the OpenHuman `ApprovalGate` before dispatch
/// (same rationale/shape as [`OpenHumanTools::invoke`] — closes the Codex P1
/// finding that flow HTTP nodes bypassed the Network approval gate). A
/// `"http_cred:<name>"` `connection_ref` is parsed but there is no HTTP
/// credential store to resolve it against yet (documented stub, see
/// `http_cred_name`) — the request proceeds without injecting stored
/// credentials.
pub struct OpenHumanHttp {
    pub security: Arc<SecurityPolicy>,
    pub http_config: HttpRequestConfig,
}

#[async_trait]
impl HttpClient for OpenHumanHttp {
    async fn request(&self, request: Value, conn: Option<&str>) -> Result<Value> {
        const TOOL_NAME: &str = "flows_http_request";

        let mut audit_id: Option<String> = None;
        if let Some(gate) = crate::openhuman::approval::ApprovalGate::try_global() {
            let summary = crate::openhuman::approval::summarize_action(TOOL_NAME, &request);
            let redacted = crate::openhuman::approval::redact_args(&request);
            let (outcome, request_id) = gate.intercept_audited(TOOL_NAME, &summary, redacted).await;
            match outcome {
                crate::openhuman::approval::GateOutcome::Deny { reason } => {
                    return Err(EngineError::Capability(reason));
                }
                crate::openhuman::approval::GateOutcome::Allow => audit_id = request_id,
            }
        }

        if let Some(name) = conn.and_then(http_cred_name) {
            tracing::warn!(
                target: "flows",
                cred = %name,
                "[flows] http_request: connection_ref names an http_cred secret, but no HTTP \
                 credential store exists yet — proceeding WITHOUT injecting stored credentials \
                 (documented stub, see caps.rs's OpenHumanHttp doc)"
            );
        } else if let Some(c) = conn {
            tracing::debug!(target: "flows", conn = %c, "[flows] http conn: unrecognized connection_ref prefix (expected `http_cred:<name>`) — ignoring");
        }

        let tool = HttpRequestTool::new(
            self.security.clone(),
            self.http_config.allowed_domains.clone(),
            self.http_config.max_response_size,
            self.http_config.timeout_secs,
        );

        tracing::debug!(
            target: "flows",
            method = ?request.get("method"),
            url = ?request.get("url"),
            "[flows] http_request: dispatching outbound request"
        );

        // `request` is already `{ method, url, headers?, body? }` — the node's
        // config is the request descriptor; `HttpRequestTool::execute` reads
        // only those keys and ignores the rest (e.g. `connection_ref`,
        // `on_error`), so passing the whole config through is safe.
        let result = tool.execute(request).await;

        let outcome: Result<Value> = match result {
            Ok(result) if result.is_error => {
                // `HttpRequestTool::execute` always returns `Ok`, using
                // `is_error` to signal a failed request (non-2xx, DNS/allowlist
                // rejection, timeout, …) — surface that as a capability error
                // so the engine's `on_error`/`retry` policy can act on it.
                Err(EngineError::Capability(result.text()))
            }
            Ok(result) => Ok(json!({ "text": result.text() })),
            Err(e) => Err(EngineError::Capability(e.to_string())),
        };

        if let Some(id) = audit_id {
            if let Some(gate) = crate::openhuman::approval::ApprovalGate::try_global() {
                let exec = if outcome.is_ok() {
                    crate::openhuman::approval::ExecutionOutcome::Success
                } else {
                    crate::openhuman::approval::ExecutionOutcome::Failure
                };
                gate.record_execution(
                    &id,
                    exec,
                    outcome.as_ref().err().map(ToString::to_string).as_deref(),
                );
            }
        }

        outcome
    }
}

/// [`CodeRunner`] adapter running sandboxed user code via
/// `src/openhuman/sandbox/ops.rs` (`resolve_sandbox_policy` +
/// `execute_in_sandbox`), modeled on
/// `src/openhuman/tools/impl/system/node_exec.rs::run_sandboxed`.
///
/// **Mismatch handled here:** the sandbox runs a shell command string, not a
/// `(language, source, input)` triple. `source` is treated as a function body
/// receiving the serialized `input` items array and returning the node's
/// output — this convention is a B1 design choice (not specified by the
/// crate), matching the mock's "function body" tests
/// (`tinyflows::nodes::integration::code` — e.g. `"source": "return 1;"`).
///
/// Requires `node`/`python3` on the `PATH` the sandbox backend runs under;
/// there is no managed toolchain wiring here (unlike `node_exec`'s
/// `NodeBootstrap`).
pub struct OpenHumanCode {
    pub config: Arc<Config>,
}

const CODE_RUN_TIMEOUT_SECS: u64 = 60;

#[async_trait]
impl CodeRunner for OpenHumanCode {
    async fn run(&self, language: CodeLanguage, source: &str, input: Value) -> Result<Value> {
        let policy = resolve_sandbox_policy(
            SandboxMode::Sandboxed,
            &self.config.action_dir,
            &self.config.runtime,
            false,
        );

        // Work dir lives under `action_dir` (the sandbox workspace root). We keep
        // its path *relative* to `action_dir` so the run command works on every
        // backend: for Local, `execute_in_sandbox`'s `working_dir` is the host
        // cwd; for Docker, `action_dir` is bind-mounted at `/workspace` with
        // `-w /workspace`. Host-absolute paths would not exist inside the
        // container, so we pass `action_dir` as the working dir and reference the
        // script/input by their `action_dir`-relative paths.
        let rel_dir = std::path::Path::new(".flows_code").join(uuid::Uuid::new_v4().to_string());
        let work_dir = self.config.action_dir.join(&rel_dir);
        tokio::fs::create_dir_all(&work_dir)
            .await
            .map_err(|e| EngineError::Capability(format!("failed to create code work dir: {e}")))?;

        let (script_name, interpreter, script_body) = match language {
            CodeLanguage::JavaScript => ("script.js", "node", js_harness(source)),
            CodeLanguage::Python => ("script.py", "python3", python_harness(source)),
        };
        let script_path = work_dir.join(script_name);
        let input_path = work_dir.join("input.json");

        let input_json = serde_json::to_string(&input)
            .map_err(|e| EngineError::Capability(format!("failed to serialize code input: {e}")))?;
        tokio::fs::write(&script_path, script_body)
            .await
            .map_err(|e| EngineError::Capability(format!("failed to write code script: {e}")))?;
        tokio::fs::write(&input_path, input_json)
            .await
            .map_err(|e| EngineError::Capability(format!("failed to write code input: {e}")))?;

        // Backend-agnostic, `action_dir`-relative command paths (see above).
        let rel_script = rel_dir.join(script_name);
        let rel_input = rel_dir.join("input.json");
        let command = format!(
            "{} {} {}",
            shell_quote(interpreter),
            shell_quote(&rel_script.to_string_lossy()),
            shell_quote(&rel_input.to_string_lossy()),
        );

        let mut extra_env = std::collections::HashMap::new();
        if let Ok(host_path) = std::env::var("PATH") {
            extra_env.insert("PATH".to_string(), host_path);
        }

        tracing::debug!(
            target: "flows",
            ?language,
            work_dir = %work_dir.display(),
            "[flows] code: running sandboxed script"
        );

        let exec_result = execute_in_sandbox(
            &policy,
            &command,
            &self.config.action_dir,
            extra_env,
            std::time::Duration::from_secs(CODE_RUN_TIMEOUT_SECS),
        )
        .await;

        // Always clean up the work dir — even when `execute_in_sandbox` itself
        // errors (e.g. a spawn failure) — so temp scripts never leak.
        if let Err(e) = tokio::fs::remove_dir_all(&work_dir).await {
            tracing::debug!(target: "flows", error = %e, "[flows] code: failed to clean up work dir (non-fatal)");
        }

        let result = exec_result
            .map_err(|e| EngineError::Capability(format!("sandbox execution failed: {e}")))?;

        if !result.success() {
            return Err(EngineError::Capability(format!(
                "code node exited non-zero (timed_out={}): {}",
                result.timed_out, result.stderr
            )));
        }

        serde_json::from_str(result.stdout.trim())
            .map_err(|e| EngineError::Capability(format!("code output was not valid JSON: {e}")))
    }
}

/// Wraps user `source` as a function body receiving `input`, executed by Node,
/// printing the JSON result (or `null`) to stdout.
fn js_harness(source: &str) -> String {
    format!(
        "const fs = require('fs');\n\
         const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));\n\
         const __result__ = (function(input) {{\n{source}\n}})(input);\n\
         process.stdout.write(JSON.stringify(__result__ === undefined ? null : __result__));\n"
    )
}

/// Wraps user `source` as a function body receiving `input`, executed by
/// Python, printing the JSON result (or `null`) to stdout.
fn python_harness(source: &str) -> String {
    let indented: String = if source.trim().is_empty() {
        "    pass".to_string()
    } else {
        source
            .lines()
            .map(|line| format!("    {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "import sys, json\n\
         with open(sys.argv[1]) as __f__:\n    input = json.load(__f__)\n\
         def __user_fn__(input):\n{indented}\n    return None\n\
         __result__ = __user_fn__(input)\n\
         print(json.dumps(__result__))\n"
    )
}

/// POSIX single-quote shell escaping, mirroring
/// `tools/impl/system/node_exec.rs::shell_quote`.
fn shell_quote(s: &str) -> String {
    let escaped = s.replace('\'', "'\\''");
    format!("'{escaped}'")
}

/// [`StateStore`] adapter over the `flows::` domain's `flow_state` KV table.
pub struct FlowStateStore {
    pub config: Arc<Config>,
    pub namespace: String,
}

#[async_trait]
impl StateStore for FlowStateStore {
    async fn load(&self, key: &str) -> Result<Option<Value>> {
        flows::kv_get(&self.config, &self.namespace, key)
            .map_err(|e| EngineError::Capability(e.to_string()))
    }

    async fn store(&self, key: &str, value: Value) -> Result<()> {
        flows::kv_set(&self.config, &self.namespace, key, &value)
            .map_err(|e| EngineError::Capability(e.to_string()))
    }
}

/// Builds the [`Capabilities`] bundle for one run, wiring each of the five
/// host-injected traits to a real OpenHuman adapter (see each adapter above for
/// its contract).
///
/// `state_namespace` scopes the [`FlowStateStore`] KV so two saved flows that
/// use the same state key never read or overwrite each other — callers pass a
/// per-flow namespace (e.g. `"flow:<id>"`).
pub fn build_capabilities(config: Arc<Config>, state_namespace: impl Into<String>) -> Capabilities {
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
        &config.action_dir,
    ));
    let http_config = config.http_request.clone();

    Capabilities {
        llm: Arc::new(OpenHumanLlm {
            config: config.clone(),
        }),
        tools: Arc::new(OpenHumanTools {
            config: config.clone(),
        }),
        http: Arc::new(OpenHumanHttp {
            security,
            http_config,
        }),
        code: Arc::new(OpenHumanCode {
            config: config.clone(),
        }),
        state: Arc::new(FlowStateStore {
            config,
            namespace: state_namespace.into(),
        }),
    }
}

/// Opens the durable, cross-process checkpointer a `flows_run` uses via
/// `tinyflows::engine::run_with_checkpointer` — the crate's own
/// `tinyagents::graph::SqliteCheckpointer`, stored under
/// `<workspace_dir>/flows/checkpoints.db`.
///
/// Deliberately **not** a bespoke checkpointer: the crate ships its own
/// SQLite-backed `Checkpointer<State>` impl (feature `sqlite`, already enabled
/// on the `tinyagents` dependency), so the seam just opens it — mirrors the
/// construction in `src/openhuman/agent_orchestration/delegation.rs`.
pub fn open_flow_checkpointer(
    config: &Config,
) -> anyhow::Result<Arc<dyn tinyflows::engine::Checkpointer<serde_json::Value>>> {
    let db_path = config.workspace_dir.join("flows").join("checkpoints.db");
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create flows directory: {}", parent.display()))?;
    }
    tracing::debug!(target: "flows", db = %db_path.display(), "[flows] opening checkpointer");
    Ok(Arc::new(
        SqliteCheckpointer::<serde_json::Value>::open(&db_path)
            .with_context(|| format!("Failed to open flows checkpointer: {}", db_path.display()))?,
    ))
}
