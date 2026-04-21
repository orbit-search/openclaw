import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { ImageContent } from "../agents/command/types.js";
import { createDefaultDeps } from "../cli/deps.js";
import { agentCommandFromIngress } from "../commands/agent.js";
import type { GatewayHttpChatCompletionsConfig } from "../config/types.gateway.js";
import { emitAgentEvent, onAgentEvent } from "../infra/agent-events.js";
import { logWarn } from "../logger.js";
import { estimateBase64DecodedBytes } from "../media/base64.js";
import {
  DEFAULT_INPUT_IMAGE_MAX_BYTES,
  DEFAULT_INPUT_IMAGE_MIMES,
  DEFAULT_INPUT_MAX_REDIRECTS,
  DEFAULT_INPUT_TIMEOUT_MS,
  extractImageContentFromSource,
  normalizeMimeList,
  type InputImageLimits,
  type InputImageSource,
} from "../media/input-files.js";
import { defaultRuntime } from "../runtime.js";
import { isInternalMessageChannel } from "../utils/message-channel.js";
import { resolveAssistantStreamDeltaText } from "./agent-event-assistant-text.js";
import {
  buildAgentMessageFromConversationEntries,
  type ConversationEntry,
} from "./agent-prompt.js";
import type { AuthRateLimiter } from "./auth-rate-limit.js";
import type { ResolvedGatewayAuth } from "./auth.js";
import { sendJson, setSseHeaders, writeDone } from "./http-common.js";
import { handleGatewayPostJsonEndpoint } from "./http-endpoint-helpers.js";
import {
  getHeader,
  resolveGatewayRequestContext,
  resolveOpenAiCompatModelOverride,
  resolveOpenAiCompatibleHttpOperatorScopes,
  resolveOpenAiCompatibleHttpSenderIsOwner,
  resolveWorkspaceOverride,
} from "./http-utils.js";
import { normalizeInputHostnameAllowlist } from "./input-allowlist.js";

const CTX_HEADER_PREFIX = "x-openclaw-ctx-";

/**
 * Extract all `X-OpenClaw-Ctx-*` request headers, strip the prefix, and return
 * them as a plain key→value map.  Header names are lower-cased after stripping
 * so that `X-OpenClaw-Ctx-ReplyToId` becomes `replytoid` in the map.
 */
function extractCtxHeaders(req: IncomingMessage): Record<string, string> | undefined {
  let ctx: Record<string, string> | undefined;
  for (const [key, value] of Object.entries(req.headers)) {
    if (!key.startsWith(CTX_HEADER_PREFIX)) {
      continue;
    }
    const strippedKey = key.slice(CTX_HEADER_PREFIX.length);
    if (!strippedKey) {
      continue;
    }
    const strValue =
      typeof value === "string" ? value : Array.isArray(value) ? value[0] : undefined;
    if (strValue != null) {
      ctx ??= {};
      ctx[strippedKey] = strValue;
    }
  }
  return ctx;
}

type OpenAiHttpOptions = {
  auth: ResolvedGatewayAuth;
  config?: GatewayHttpChatCompletionsConfig;
  maxBodyBytes?: number;
  trustedProxies?: string[];
  allowRealIpFallback?: boolean;
  rateLimiter?: AuthRateLimiter;
};

type OpenAiChatMessage = {
  role?: unknown;
  content?: unknown;
  name?: unknown;
};

type OpenAiChatCompletionRequest = {
  model?: unknown;
  stream?: unknown;
  messages?: unknown;
  user?: unknown;
};

const DEFAULT_OPENAI_CHAT_COMPLETIONS_BODY_BYTES = 20 * 1024 * 1024;
const IMAGE_ONLY_USER_MESSAGE = "User sent image(s) with no text.";
const DEFAULT_OPENAI_MAX_IMAGE_PARTS = 8;
const DEFAULT_OPENAI_MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024;
const DEFAULT_OPENAI_IMAGE_LIMITS: InputImageLimits = {
  allowUrl: false,
  allowedMimes: new Set(DEFAULT_INPUT_IMAGE_MIMES),
  maxBytes: DEFAULT_INPUT_IMAGE_MAX_BYTES,
  maxRedirects: DEFAULT_INPUT_MAX_REDIRECTS,
  timeoutMs: DEFAULT_INPUT_TIMEOUT_MS,
};

type ResolvedOpenAiChatCompletionsLimits = {
  maxBodyBytes: number;
  maxImageParts: number;
  maxTotalImageBytes: number;
  images: InputImageLimits;
};

function resolveOpenAiChatCompletionsLimits(
  config: GatewayHttpChatCompletionsConfig | undefined,
): ResolvedOpenAiChatCompletionsLimits {
  const imageConfig = config?.images;
  return {
    maxBodyBytes: config?.maxBodyBytes ?? DEFAULT_OPENAI_CHAT_COMPLETIONS_BODY_BYTES,
    maxImageParts:
      typeof config?.maxImageParts === "number"
        ? Math.max(0, Math.floor(config.maxImageParts))
        : DEFAULT_OPENAI_MAX_IMAGE_PARTS,
    maxTotalImageBytes:
      typeof config?.maxTotalImageBytes === "number"
        ? Math.max(1, Math.floor(config.maxTotalImageBytes))
        : DEFAULT_OPENAI_MAX_TOTAL_IMAGE_BYTES,
    images: {
      allowUrl: imageConfig?.allowUrl ?? DEFAULT_OPENAI_IMAGE_LIMITS.allowUrl,
      urlAllowlist: normalizeInputHostnameAllowlist(imageConfig?.urlAllowlist),
      allowedMimes: normalizeMimeList(imageConfig?.allowedMimes, DEFAULT_INPUT_IMAGE_MIMES),
      maxBytes: imageConfig?.maxBytes ?? DEFAULT_INPUT_IMAGE_MAX_BYTES,
      maxRedirects: imageConfig?.maxRedirects ?? DEFAULT_INPUT_MAX_REDIRECTS,
      timeoutMs: imageConfig?.timeoutMs ?? DEFAULT_INPUT_TIMEOUT_MS,
    },
  };
}

function writeSse(res: ServerResponse, data: unknown) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function buildAgentCommandInput(params: {
  prompt: { message: string; extraSystemPrompt?: string; images?: ImageContent[] };
  modelOverride?: string;
  sessionKey: string;
  runId: string;
  messageChannel: string;
  senderIsOwner: boolean;
  workspaceDir?: string;
  deliveryCtx?: Record<string, string>;
  replyToId?: string;
  deliver?: boolean;
  channel?: string;
}) {
  return {
    message: params.prompt.message,
    extraSystemPrompt: params.prompt.extraSystemPrompt,
    images: params.prompt.images,
    model: params.modelOverride,
    sessionKey: params.sessionKey,
    runId: params.runId,
    deliver: params.deliver ?? false,
    messageChannel: params.messageChannel,
    bestEffortDeliver: false as const,
    senderIsOwner: params.senderIsOwner,
    allowModelOverride: true as const,
    workspaceDir: params.workspaceDir,
    deliveryCtx: params.deliveryCtx,
    replyToId: params.replyToId,
    channel: params.channel,
  };
}

function writeAssistantRoleChunk(res: ServerResponse, params: { runId: string; model: string }) {
  writeSse(res, {
    id: params.runId,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: params.model,
    choices: [{ index: 0, delta: { role: "assistant" } }],
  });
}

function writeAssistantContentChunk(
  res: ServerResponse,
  params: { runId: string; model: string; content: string; finishReason: "stop" | null },
) {
  writeSse(res, {
    id: params.runId,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: params.model,
    choices: [
      {
        index: 0,
        delta: { content: params.content },
        finish_reason: params.finishReason,
      },
    ],
  });
}

function asMessages(val: unknown): OpenAiChatMessage[] {
  return Array.isArray(val) ? (val as OpenAiChatMessage[]) : [];
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (!part || typeof part !== "object") {
          return "";
        }
        const type = (part as { type?: unknown }).type;
        const text = (part as { text?: unknown }).text;
        const inputText = (part as { input_text?: unknown }).input_text;
        if (type === "text" && typeof text === "string") {
          return text;
        }
        if (type === "input_text" && typeof text === "string") {
          return text;
        }
        if (typeof inputText === "string") {
          return inputText;
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function resolveImageUrlPart(part: unknown): string | undefined {
  if (!part || typeof part !== "object") {
    return undefined;
  }
  const imageUrl = (part as { image_url?: unknown }).image_url;
  if (typeof imageUrl === "string") {
    const trimmed = imageUrl.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  if (!imageUrl || typeof imageUrl !== "object") {
    return undefined;
  }
  const rawUrl = (imageUrl as { url?: unknown }).url;
  if (typeof rawUrl !== "string") {
    return undefined;
  }
  const trimmed = rawUrl.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function extractImageUrls(content: unknown): string[] {
  if (!Array.isArray(content)) {
    return [];
  }
  const urls: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") {
      continue;
    }
    if ((part as { type?: unknown }).type !== "image_url") {
      continue;
    }
    const url = resolveImageUrlPart(part);
    if (url) {
      urls.push(url);
    }
  }
  return urls;
}

type ActiveTurnContext = {
  activeTurnIndex: number;
  activeUserMessageIndex: number;
  urls: string[];
};

function parseImageUrlToSource(url: string): InputImageSource {
  const dataUriMatch = /^data:([^,]*?),(.*)$/is.exec(url);
  if (dataUriMatch) {
    const metadata = dataUriMatch[1]?.trim() ?? "";
    const data = dataUriMatch[2] ?? "";
    const metadataParts = metadata
      .split(";")
      .map((part) => part.trim())
      .filter(Boolean);
    const isBase64 = metadataParts.some((part) => part.toLowerCase() === "base64");
    if (!isBase64) {
      throw new Error("image_url data URI must be base64 encoded");
    }
    if (!data.trim()) {
      throw new Error("image_url data URI is missing payload data");
    }
    const mediaTypeRaw = metadataParts.find((part) => part.includes("/"));
    return {
      type: "base64",
      mediaType: mediaTypeRaw,
      data,
    };
  }
  return { type: "url", url };
}

function resolveActiveTurnContext(messagesUnknown: unknown): ActiveTurnContext {
  const messages = asMessages(messagesUnknown);
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") {
      continue;
    }
    const role = typeof msg.role === "string" ? msg.role.trim() : "";
    const normalizedRole = role === "function" ? "tool" : role;
    if (normalizedRole !== "user" && normalizedRole !== "tool") {
      continue;
    }
    return {
      activeTurnIndex: i,
      activeUserMessageIndex: normalizedRole === "user" ? i : -1,
      urls: normalizedRole === "user" ? extractImageUrls(msg.content) : [],
    };
  }
  return { activeTurnIndex: -1, activeUserMessageIndex: -1, urls: [] };
}

async function resolveImagesForRequest(
  activeTurnContext: Pick<ActiveTurnContext, "urls">,
  limits: ResolvedOpenAiChatCompletionsLimits,
): Promise<ImageContent[]> {
  const urls = activeTurnContext.urls;
  if (urls.length === 0) {
    return [];
  }
  if (urls.length > limits.maxImageParts) {
    throw new Error(`Too many image_url parts (${urls.length}; limit ${limits.maxImageParts})`);
  }

  const images: ImageContent[] = [];
  let totalBytes = 0;
  for (const url of urls) {
    const source = parseImageUrlToSource(url);
    if (source.type === "base64") {
      const sourceBytes = estimateBase64DecodedBytes(source.data);
      if (totalBytes + sourceBytes > limits.maxTotalImageBytes) {
        throw new Error(
          `Total image payload too large (${totalBytes + sourceBytes}; limit ${limits.maxTotalImageBytes})`,
        );
      }
    }

    const image = await extractImageContentFromSource(source, limits.images);
    totalBytes += estimateBase64DecodedBytes(image.data);
    if (totalBytes > limits.maxTotalImageBytes) {
      throw new Error(
        `Total image payload too large (${totalBytes}; limit ${limits.maxTotalImageBytes})`,
      );
    }
    images.push(image);
  }
  return images;
}

export const __testOnlyOpenAiHttp = {
  resolveImagesForRequest,
  resolveOpenAiChatCompletionsLimits,
  resolveAgentResponseText,
};

function buildAgentPrompt(
  messagesUnknown: unknown,
  activeUserMessageIndex: number,
): {
  message: string;
  extraSystemPrompt?: string;
} {
  const messages = asMessages(messagesUnknown);

  const systemParts: string[] = [];
  const conversationEntries: ConversationEntry[] = [];

  for (const [i, msg] of messages.entries()) {
    if (!msg || typeof msg !== "object") {
      continue;
    }
    const role = typeof msg.role === "string" ? msg.role.trim() : "";
    const content = extractTextContent(msg.content).trim();
    const hasImage = extractImageUrls(msg.content).length > 0;
    if (!role) {
      continue;
    }
    if (role === "system" || role === "developer") {
      if (content) {
        systemParts.push(content);
      }
      continue;
    }

    const normalizedRole = role === "function" ? "tool" : role;
    if (normalizedRole !== "user" && normalizedRole !== "assistant" && normalizedRole !== "tool") {
      continue;
    }

    // Keep the image-only placeholder scoped to the active user turn so we don't
    // mention historical image-only turns whose bytes are intentionally not replayed.
    const messageContent =
      normalizedRole === "user" && !content && hasImage && i === activeUserMessageIndex
        ? IMAGE_ONLY_USER_MESSAGE
        : content;
    if (!messageContent) {
      continue;
    }

    const name = typeof msg.name === "string" ? msg.name.trim() : "";
    const sender =
      normalizedRole === "assistant"
        ? "Assistant"
        : normalizedRole === "user"
          ? "User"
          : name
            ? `Tool:${name}`
            : "Tool";

    conversationEntries.push({
      role: normalizedRole,
      entry: { sender, body: messageContent },
    });
  }

  const message = buildAgentMessageFromConversationEntries(conversationEntries);

  return {
    message,
    extraSystemPrompt: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
  };
}

function coerceRequest(val: unknown): OpenAiChatCompletionRequest {
  if (!val || typeof val !== "object") {
    return {};
  }
  return val as OpenAiChatCompletionRequest;
}

const DEFAULT_YIELD_WAIT_TIMEOUT_MS = 5 * 60 * 1000;
const MAX_YIELD_ROUNDS = 10;

function resultYielded(result: unknown): boolean {
  return (result as { meta?: { yieldDetected?: boolean } } | null)?.meta?.yieldDetected === true;
}

function waitForFollowUpResponse(params: {
  sessionKey: string;
  timeoutMs: number;
  req: IncomingMessage;
}): Promise<string | null> {
  return new Promise((resolve) => {
    console.log("[openai-http] waitForFollowUpResponse entered, sessionKey:", params.sessionKey);

    let settled = false;
    const assistantChunks: string[] = [];
    let followUpRunId: string | undefined;
    let yieldRounds = 0;

    const finish = (text: string | null) => {
      if (settled) {
        return;
      }
      console.log("[openai-http] waitForFollowUpResponse finish, text length:", text?.length ?? 0);
      settled = true;
      clearTimeout(timer);
      unsubscribe();
      params.req.removeListener("close", onClose);
      resolve(text);
    };

    const unsubscribe = onAgentEvent((evt) => {
      if (settled) {
        return;
      }
      console.log("[openai-http] event:", evt.stream, evt.sessionKey, evt.runId);
      if (evt.sessionKey !== params.sessionKey) {
        return;
      }

      if (evt.stream === "assistant") {
        if (!followUpRunId) {
          followUpRunId = evt.runId;
        }
        if (evt.runId === followUpRunId) {
          const text = resolveAssistantStreamDeltaText(evt) ?? "";
          if (text) {
            assistantChunks.push(text);
          }
        }
      }

      if (evt.stream === "lifecycle") {
        const phase = evt.data?.phase;
        if (phase === "error") {
          finish(assistantChunks.length > 0 ? assistantChunks.join("") : null);
          return;
        }

        if (phase === "end") {
          const yielded = evt.data?.yieldDetected === true;
          if (yielded) {
            yieldRounds += 1;
            if (yieldRounds >= MAX_YIELD_ROUNDS) {
              finish(assistantChunks.length > 0 ? assistantChunks.join("") : null);
              return;
            }
            followUpRunId = undefined;
            assistantChunks.length = 0;
            return;
          }
          finish(assistantChunks.length > 0 ? assistantChunks.join("") : null);
        }
      }
    });

    const timer = setTimeout(() => finish(null), params.timeoutMs);
    const onClose = () => finish(null);
    params.req.once("close", onClose);
  });
}

function resolveAgentResponseText(result: unknown, opts?: { finalResponseOnly?: boolean }): string {
  const payloads = (result as { payloads?: Array<{ text?: string; isReasoning?: boolean }> } | null)
    ?.payloads;
  if (!Array.isArray(payloads) || payloads.length === 0) {
    return "No response from OpenClaw.";
  }

  if (opts?.finalResponseOnly) {
    const lastText = [...payloads]
      .toReversed()
      .find((p) => typeof p.text === "string" && p.text.trim() && p.isReasoning !== true);
    return lastText?.text?.trim() || "No response from OpenClaw.";
  }

  const content = payloads
    .map((p) => (typeof p.text === "string" ? p.text : ""))
    .filter(Boolean)
    .join("\n\n");
  return content || "No response from OpenClaw.";
}

export async function handleOpenAiHttpRequest(
  req: IncomingMessage,
  res: ServerResponse,
  opts: OpenAiHttpOptions,
): Promise<boolean> {
  const limits = resolveOpenAiChatCompletionsLimits(opts.config);
  const handled = await handleGatewayPostJsonEndpoint(req, res, {
    pathname: "/v1/chat/completions",
    requiredOperatorMethod: "chat.send",
    // Compat HTTP uses a different scope model from generic HTTP helpers:
    // shared-secret bearer auth is treated as full operator access here.
    resolveOperatorScopes: resolveOpenAiCompatibleHttpOperatorScopes,
    auth: opts.auth,
    trustedProxies: opts.trustedProxies,
    allowRealIpFallback: opts.allowRealIpFallback,
    rateLimiter: opts.rateLimiter,
    maxBodyBytes: opts.maxBodyBytes ?? limits.maxBodyBytes,
  });
  if (handled === false) {
    return false;
  }
  if (!handled) {
    return true;
  }
  // On the compat surface, shared-secret bearer auth is also treated as an
  // owner sender so owner-only tool policy matches the documented contract.
  const senderIsOwner = resolveOpenAiCompatibleHttpSenderIsOwner(req, handled.requestAuth);

  const payload = coerceRequest(handled.body);
  const stream = Boolean(payload.stream);
  const model = typeof payload.model === "string" ? payload.model : "openclaw";
  const user = typeof payload.user === "string" ? payload.user : undefined;

  const { agentId, sessionKey, messageChannel } = resolveGatewayRequestContext({
    req,
    model,
    user,
    sessionPrefix: "openai",
    defaultMessageChannel: "webchat",
    useMessageChannelHeader: true,
  });
  const { modelOverride, errorMessage: modelError } = await resolveOpenAiCompatModelOverride({
    req,
    agentId,
    model,
  });
  if (modelError) {
    sendJson(res, 400, {
      error: { message: modelError, type: "invalid_request_error" },
    });
    return true;
  }
  const activeTurnContext = resolveActiveTurnContext(payload.messages);
  const prompt = buildAgentPrompt(payload.messages, activeTurnContext.activeUserMessageIndex);
  let images: ImageContent[] = [];
  try {
    images = await resolveImagesForRequest(activeTurnContext, limits);
  } catch (err) {
    logWarn(`openai-compat: invalid image_url content: ${String(err)}`);
    sendJson(res, 400, {
      error: {
        message: "Invalid image_url content in `messages`.",
        type: "invalid_request_error",
      },
    });
    return true;
  }

  if (!prompt.message && images.length === 0) {
    sendJson(res, 400, {
      error: {
        message: "Missing user message in `messages`.",
        type: "invalid_request_error",
      },
    });
    return true;
  }

  const runId = `chatcmpl_${randomUUID()}`;
  const deps = createDefaultDeps();
  const workspaceOverride = resolveWorkspaceOverride(req);
  const deliveryCtx = extractCtxHeaders(req);
  const replyToId = deliveryCtx?.replytoid;
  const channelRouted = !isInternalMessageChannel(messageChannel);
  const commandInput = buildAgentCommandInput({
    prompt: {
      message: prompt.message,
      extraSystemPrompt: prompt.extraSystemPrompt,
      images: images.length > 0 ? images : undefined,
    },
    modelOverride,
    sessionKey,
    runId,
    messageChannel,
    senderIsOwner,
    workspaceDir: workspaceOverride,
    deliveryCtx,
    replyToId,
    deliver: channelRouted,
    channel: channelRouted ? messageChannel : undefined,
  });

  // Channel-routed request: fire-and-forget, let the channel pipeline deliver.
  if (channelRouted) {
    void agentCommandFromIngress(commandInput, defaultRuntime, deps).catch((err) => {
      logWarn(`openai-compat: channel-routed agent run failed: ${String(err)}`);
    });
    sendJson(res, 200, {
      id: runId,
      object: "chat.completion",
      dispatched: true,
      message: "Response will be delivered via channel",
    });
    return true;
  }

  const finalResponseOnly = getHeader(req, "x-openclaw-final-response-only") === "true";

  if (!stream) {
    try {
      const result = await agentCommandFromIngress(commandInput, defaultRuntime, deps);

      console.log(
        "[openai-http] result meta:",
        JSON.stringify((result as { meta?: unknown } | null)?.meta ?? {}),
      );
      console.log("[openai-http] yieldDetected:", resultYielded(result));

      let content: string;
      if (resultYielded(result) && !req.destroyed) {
        const followUpText = await waitForFollowUpResponse({
          sessionKey,
          timeoutMs: DEFAULT_YIELD_WAIT_TIMEOUT_MS,
          req,
        });
        content = followUpText || resolveAgentResponseText(result, { finalResponseOnly });
      } else {
        content = resolveAgentResponseText(result, { finalResponseOnly });
      }

      sendJson(res, 200, {
        id: runId,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    } catch (err) {
      logWarn(`openai-compat: chat completion failed: ${String(err)}`);
      sendJson(res, 500, {
        error: { message: "internal error", type: "api_error" },
      });
    }
    return true;
  }

  setSseHeaders(res);

  let wroteRole = false;
  let sawAssistantDelta = false;
  let closed = false;
  let waitingForFollowUp = false;
  let followUpRunId: string | undefined;
  let yieldRounds = 0;

  const finishSse = () => {
    if (closed) {
      return;
    }
    closed = true;
    unsubscribe();
    writeDone(res);
    res.end();
  };

  const matchesCurrentRun = (evt: { runId: string; sessionKey?: string }) => {
    if (evt.runId === runId) {
      return true;
    }
    if (!waitingForFollowUp) {
      return false;
    }
    if (evt.sessionKey !== sessionKey) {
      return false;
    }
    if (!followUpRunId) {
      followUpRunId = evt.runId;
    }
    return evt.runId === followUpRunId;
  };

  const unsubscribe = onAgentEvent((evt) => {
    if (closed) {
      return;
    }
    if (!matchesCurrentRun(evt)) {
      return;
    }

    if (evt.stream === "assistant") {
      const content = resolveAssistantStreamDeltaText(evt) ?? "";
      if (!content) {
        return;
      }

      if (!wroteRole) {
        wroteRole = true;
        writeAssistantRoleChunk(res, { runId, model });
      }

      sawAssistantDelta = true;
      writeAssistantContentChunk(res, {
        runId,
        model,
        content,
        finishReason: null,
      });
      return;
    }

    if (evt.stream === "lifecycle") {
      const phase = evt.data?.phase;
      if (phase === "end") {
        const yielded = evt.data?.yieldDetected === true;
        if (yielded && yieldRounds < MAX_YIELD_ROUNDS) {
          yieldRounds += 1;
          waitingForFollowUp = true;
          followUpRunId = undefined;
          return;
        }
      }

      if (phase === "end" || phase === "error") {
        finishSse();
      }
    }
  });

  req.on("close", () => {
    closed = true;
    unsubscribe();
  });

  const yieldTimeoutTimer = setTimeout(() => {
    if (waitingForFollowUp && !closed) {
      finishSse();
    }
  }, DEFAULT_YIELD_WAIT_TIMEOUT_MS);

  void (async () => {
    try {
      const result = await agentCommandFromIngress(commandInput, defaultRuntime, deps);

      if (closed) {
        return;
      }

      if (resultYielded(result)) {
        waitingForFollowUp = true;
        followUpRunId = undefined;
        return;
      }

      if (!sawAssistantDelta) {
        if (!wroteRole) {
          wroteRole = true;
          writeAssistantRoleChunk(res, { runId, model });
        }

        const content = resolveAgentResponseText(result, { finalResponseOnly });

        sawAssistantDelta = true;
        writeAssistantContentChunk(res, {
          runId,
          model,
          content,
          finishReason: null,
        });
      }
    } catch (err) {
      logWarn(`openai-compat: streaming chat completion failed: ${String(err)}`);
      if (closed) {
        return;
      }
      writeAssistantContentChunk(res, {
        runId,
        model,
        content: "Error: internal error",
        finishReason: "stop",
      });
      emitAgentEvent({
        runId,
        stream: "lifecycle",
        data: { phase: "error" },
      });
    } finally {
      clearTimeout(yieldTimeoutTimer);
      if (!closed && !waitingForFollowUp) {
        finishSse();
      }
    }
  })();

  return true;
}
