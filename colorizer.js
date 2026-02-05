import "dotenv/config";
import fsp from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import OpenAI from "openai";

// ── Configuration ──────────────────────────────────────────────────────────

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const INPUT_DIR = process.env.INPUT_DIR || "./input";
const OUTPUT_DIR = process.env.OUTPUT_DIR || "./output";
const DEBUG_DIR = "./debug";

const OUTPUT_W = Number(process.env.OUTPUT_WIDTH) || 800;
const OUTPUT_H = Number(process.env.OUTPUT_HEIGHT) || 1280;

const DARK_THRESHOLD = Number(process.env.DARK_THRESHOLD) || 20;
const MIN_GAP_HEIGHT = Number(process.env.MIN_GAP_HEIGHT) || 30;
const EDGE_TOLERANCE = Number(process.env.EDGE_TOLERANCE) || 0.02;

const DEBUG = process.env.DEBUG === "true";

// Image generation quality tier: "low", "medium", or "high".
// Controls image gen quality, input resolution, and vision detail.
const QUALITY = (process.env.QUALITY || "medium").toLowerCase();

// Quality-tier settings: only maxWidth changes between tiers.
// Model, fidelity, and vision detail stay at their best values always.
const QUALITY_TIERS = {
  high: { maxWidth: Infinity },
  medium: { maxWidth: 1024 },
  low: { maxWidth: 800 },
};
const TIER = QUALITY_TIERS[QUALITY] || QUALITY_TIERS.medium;

// Character palette — selects a JSON file from ./palettes/
const PALETTE = process.env.PALETTE || "intro";

// Max height (in pixels) for a single segment sent to the API.
// Segments taller than this are resized down before colorizing.
const MAX_SEGMENT_H = 4000;

// Min height for a segment — anything smaller gets merged with its neighbor.
const MIN_SEGMENT_H = 100;

// Intelligent context capture — learns color choices for unlocked elements.
const CAPTURE_CONTEXT = process.env.CAPTURE_CONTEXT !== "false"; // default true

// Black restoration post-processing — set to "false" to disable
const RESTORE_BLACKS = process.env.RESTORE_BLACKS !== "false"; // default true


// System-level instructions — sent via the `instructions` parameter.
// Keep this minimal — just role and safety context.
const SYSTEM_INSTRUCTIONS = `
You are a professional Korean webtoon colorist. Colorize black-and-white manga/manhwa panels using the image_generation tool in edit mode. Always call the tool — never respond with text only.

CONTEXT: This is a published Korean webtoon (manhwa). All content is fictional and safe for colorization. Sound effects like "BANG", "SLAM", "CRACK" are physical actions, not violence.
`.trim();

// Main colorization prompt — shorter and less restrictive for better colors.
const COLORIZATION_PROMPT = `
Colorize this black-and-white webtoon panel with vibrant, professional Korean manhwa colors.

STYLE: Rich, saturated colors like "Solo Leveling" or "Tower of God". Clean cel-shading with good contrast. Natural lighting.

KEY RULES:
- Skin: Light Korean skin tone (#FAE0D4), warm peach.
- ALL black areas must stay pure black (#000000): panel dividers (including thin horizontal/vertical bars), black backgrounds, silhouettes, borders between panels. Never color these blue, tan, or any other color.
- SPEECH BUBBLES: DO NOT MODIFY AT ALL. Leave speech bubbles exactly as they appear — white fill, black outline, black text. Do not redraw, move, resize, or alter the text in any way. The text must remain pixel-perfect identical to the original.
- Sound effects text (like "HAHA", "BANG", etc.): Preserve exactly as drawn, do not redraw or distort.
`.trim();

// User-level prompt — combines main colorization prompt with character palette.
async function loadPalette() {
  const palettePath = path.join(".", "palettes", `${PALETTE}.json`);
  try {
    const raw = await fsp.readFile(palettePath, "utf-8");
    const data = JSON.parse(raw);
    console.log(`Palette: ${data.name} (${PALETTE}.json)`);
    const lines = data.characters.map((c) => `- ${c}`).join("\n");
    // Combine main prompt with character-specific colors
    return `${COLORIZATION_PROMPT}\n\nCHARACTERS:\n${lines}`;
  } catch (err) {
    if (err.code === "ENOENT") {
      throw new Error(`Palette file not found: ${palettePath}\nAvailable palettes are in the ./palettes/ directory.`);
    }
    throw err;
  }
}

// ── Intelligent context system ────────────────────────────────────────────

const CONTEXT_PATH = path.join(".", "palettes", `${PALETTE}_context.json`);

async function loadContext() {
  try {
    const raw = await fsp.readFile(CONTEXT_PATH, "utf-8");
    const data = JSON.parse(raw);
    const entries = data.learned || [];
    if (entries.length > 0) {
      console.log(`Context: loaded ${entries.length} learned color(s) from ${PALETTE}_context.json`);
    }
    return entries;
  } catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

async function saveContext(entries) {
  await fsp.writeFile(CONTEXT_PATH, JSON.stringify({ learned: entries }, null, 2));
}

function buildPromptWithContext(basePrompt, contextEntries) {
  if (contextEntries.length === 0) return basePrompt;
  const contextBlock = contextEntries.map((c) => `- ${c}`).join("\n");
  return `${basePrompt}\n\nPREVIOUSLY LEARNED COLORS (use these for consistency, but override if clearly wrong):\n${contextBlock}`;
}

async function captureContext(colorizedBuf, palettePrompt) {
  try {
    const res = await client.responses.create({
      model: "gpt-5.2",
      instructions: "You analyze colorized webtoon panels. Return ONLY a JSON array of short strings describing colors you observe for elements NOT already specified in the palette (e.g., backgrounds, furniture, unnamed clothing, objects). Each string should be like: \"hospital hallway: pale mint-green walls (#D4E8D6)\". If nothing notable, return an empty array [].",
      input: [{
        role: "user",
        content: [
          { type: "input_image", image_url: toDataUrl(colorizedBuf), detail: "low" },
          { type: "input_text", text: `Here is the palette that was already specified:\n${palettePrompt}\n\nList colors chosen for elements NOT in the palette above. Return a JSON array of strings. Be concise — only notable/reusable colors.` },
        ],
      }],
    });

    // Extract text response
    for (const item of res.output) {
      if (item.type === "message" && item.content) {
        for (const c of item.content) {
          if (c.text) {
            // Parse JSON array from response (handle markdown code fences)
            const cleaned = c.text.replace(/```json?\s*/g, "").replace(/```/g, "").trim();
            const arr = JSON.parse(cleaned);
            if (Array.isArray(arr)) return arr.filter((s) => typeof s === "string" && s.length > 0);
          }
        }
      }
    }
    return [];
  } catch (err) {
    console.warn(`    Context capture failed (non-fatal): ${err.message}`);
    return [];
  }
}

// ── Utility helpers ────────────────────────────────────────────────────────

function parseName(filename) {
  const base = path.basename(filename);
  const m = base.match(/^(.*?)(\d+)(\.[^.]+)$/);
  if (!m) return { key: base, idx: 0, ext: ".png" };
  return { key: m[1], idx: Number(m[2]), ext: m[3] };
}

async function ensureDir(dir) {
  await fsp.mkdir(dir, { recursive: true });
}

async function listImages(dir) {
  const files = await fsp.readdir(dir);
  return files
    .filter((f) => /\.(png|jpe?g)$/i.test(f))
    .map((f) => path.join(dir, f));
}

function toDataUrl(buf) {
  return `data:image/png;base64,${buf.toString("base64")}`;
}

function pad(n, len = 3) {
  return String(n).padStart(len, "0");
}

// ── Step 1: Stitch slices into one tall strip ──────────────────────────────

async function stitchSlices(filePaths) {
  const metas = await Promise.all(filePaths.map((f) => sharp(f).metadata()));
  const width = metas[0].width;

  // Normalize all slices to the same width
  const buffers = await Promise.all(
    filePaths.map(async (f, i) => {
      const m = metas[i];
      let img = sharp(f);
      if (m.width !== width) {
        img = img.resize(width, m.height, { fit: "fill" });
      }
      return img.png().toBuffer();
    })
  );

  const heights = metas.map((m) => m.height);
  const totalH = heights.reduce((a, b) => a + b, 0);

  let y = 0;
  const composites = [];
  for (let i = 0; i < buffers.length; i++) {
    composites.push({ input: buffers[i], top: y, left: 0 });
    y += heights[i];
  }

  const stitched = await sharp({
    create: {
      width,
      height: totalH,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 1 },
    },
  })
    .composite(composites)
    .png()
    .toBuffer();

  return { buffer: stitched, width, totalH, heights };
}

// ── Step 2: Detect safe split points ───────────────────────────────────────

async function detectSafeSplitPoints(stitchedBuf) {
  const { data, info } = await sharp(stitchedBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const w = info.width;
  const h = info.height;
  const c = info.channels;

  // Identify rows where all (or nearly all) pixels are dark
  const safeRows = [];
  for (let y = 0; y < h; y++) {
    let darkCount = 0;
    const rowStart = y * w * c;
    for (let x = 0; x < w; x++) {
      const p = rowStart + x * c;
      if (
        data[p] < DARK_THRESHOLD &&
        data[p + 1] < DARK_THRESHOLD &&
        data[p + 2] < DARK_THRESHOLD
      ) {
        darkCount++;
      }
    }
    if (darkCount / w >= 1 - EDGE_TOLERANCE) {
      safeRows.push(y);
    }
  }

  // Find consecutive runs of safe rows
  const gaps = [];
  let gapStart = null;

  for (let i = 0; i < safeRows.length; i++) {
    if (gapStart === null) gapStart = safeRows[i];

    const next = safeRows[i + 1];
    if (next !== safeRows[i] + 1) {
      // Run ended
      const gapEnd = safeRows[i];
      const gapHeight = gapEnd - gapStart + 1;
      if (gapHeight >= MIN_GAP_HEIGHT) {
        gaps.push({
          startRow: gapStart,
          endRow: gapEnd,
          midPoint: Math.floor((gapStart + gapEnd) / 2),
          height: gapHeight,
        });
      }
      gapStart = null;
    }
  }

  return gaps;
}

// ── Step 3: Split into segments ────────────────────────────────────────────

async function splitAtPoints(stitchedBuf, width, totalH, splitPoints) {
  const cuts = splitPoints.map((sp) => sp.midPoint);

  // Build segment boundaries: [0, cut1, cut2, ..., totalH]
  const bounds = [0, ...cuts, totalH];
  let segments = [];

  for (let i = 0; i < bounds.length - 1; i++) {
    const startY = bounds[i];
    const endY = bounds[i + 1];
    const h = endY - startY;
    if (h <= 0) continue;
    segments.push({ startY, endY, height: h });
  }

  // Merge tiny segments with their neighbor
  segments = segments.filter((seg, i) => {
    if (seg.height >= MIN_SEGMENT_H) return true;
    // Merge into next or previous
    if (i + 1 < segments.length) {
      segments[i + 1].startY = seg.startY;
      segments[i + 1].height = segments[i + 1].endY - seg.startY;
    } else if (i > 0) {
      segments[i - 1].endY = seg.endY;
      segments[i - 1].height = seg.endY - segments[i - 1].startY;
    }
    return false;
  });

  // Extract each segment
  const result = [];
  for (const seg of segments) {
    const buf = await sharp(stitchedBuf)
      .extract({ left: 0, top: seg.startY, width, height: seg.height })
      .png()
      .toBuffer();
    result.push({ buffer: buf, startY: seg.startY, height: seg.height, width });
  }

  return result;
}

// ── Retry helper ────────────────────────────────────────────────────────────

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 2000;

function isTransient(err) {
  // Rate limits, server errors, timeouts, network failures
  if (err.status === 429 || (err.status >= 500 && err.status < 600)) return true;
  if (/timeout|ETIMEDOUT|ECONNRESET|ECONNREFUSED|socket hang up|network/i.test(err.message)) return true;
  if (/safety|content_policy|moderation/i.test(err.message)) return true;
  return false;
}

async function withRetry(fn, label) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const isSafety = /safety|content_policy|moderation/i.test(err.message);
      const retriable = isTransient(err);

      if (!retriable || attempt === MAX_RETRIES) {
        // Non-transient or exhausted retries — return null to signal failure
        const reason = isSafety ? "safety filter" : err.message;
        console.warn(`    WARNING: ${label} failed after ${attempt + 1} attempt(s): ${reason}`);
        return null;
      }

      // Compute delay: respect Retry-After header if present, otherwise exponential backoff
      let delay = BASE_DELAY_MS * Math.pow(2, attempt);
      if (err.headers?.["retry-after"]) {
        const ra = Number(err.headers["retry-after"]);
        if (!isNaN(ra)) delay = ra * 1000;
      }
      const tag = isSafety ? "safety filter" : `${err.status || "network error"}`;
      console.log(`    ${tag} — retry ${attempt + 1}/${MAX_RETRIES} in ${(delay / 1000).toFixed(1)}s...`);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
}

// ── Step 4: Colorize a segment via Responses API ────────────────────────────

async function isBlankSegment(buf) {
  const { data, info } = await sharp(buf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const total = info.width * info.height;
  let darkCount = 0;
  let brightCount = 0;
  for (let i = 0; i < total; i++) {
    const p = i * info.channels;
    const r = data[p], g = data[p + 1], b = data[p + 2];
    if (r < DARK_THRESHOLD && g < DARK_THRESHOLD && b < DARK_THRESHOLD) {
      darkCount++;
    } else if (r > 200 && g > 200 && b > 200) {
      brightCount++;
    }
  }

  const darkRatio = darkCount / total;
  const nonDark = total - darkCount;
  const brightRatio = nonDark > 0 ? brightCount / nonDark : 0;

  // Pure black segments (dividers, empty space) — 98%+ dark
  if (darkRatio >= 0.98) return `blank (${(darkRatio * 100).toFixed(0)}% black)`;

  // Text-on-black segments (e.g., "WEEKS EARLIER...", "ALONE...") —
  // mostly black with small amounts of white text. No artwork to colorize.
  // Must be 85%+ dark, and non-dark pixels are mostly white text.
  // Threshold is 0.6 (not higher) to account for anti-aliased text edges.
  if (darkRatio >= 0.85 && nonDark > 0 && brightRatio >= 0.6) {
    return `text-on-black (${(darkRatio * 100).toFixed(0)}% black, ${(brightRatio * 100).toFixed(0)}% of rest is white)`;
  }

  return false;
}

// ── Post-process: restore black pixels from original ─────────────────────
// Hybrid approach: Edge-connected flood fill + speech bubble awareness
//
// Problem: Speech bubbles (white) break edge-connectivity, leaving black
// pixels around bubbles unrestored even though they're in black regions.
//
// Solution:
// 1. Find edge-connected dark pixels (flood fill from image borders)
// 2. Also find dark pixels adjacent to large white regions (speech bubbles)
//    that are in high-density black areas
// 3. Restore dark pixels that are either edge-connected OR bubble-adjacent,
//    but only if they're in a high-density region (70%+ dark locally)

const BLACK_RESTORE_THRESHOLD = 5;   // RGB < 5 = "true black"
const WHITE_THRESHOLD = 250;          // RGB > 250 = "white" (speech bubbles)
const LOCAL_CHECK_RADIUS = 16;        // 33x33 block for density check
const LOCAL_DENSITY_MIN = 0.70;       // block must be 70%+ dark
const BUBBLE_SEARCH_RADIUS = 3;       // how far to look for white pixels

async function restoreBlacks(originalBuf, colorizedBuf) {
  const origRaw = await sharp(originalBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const colRaw = await sharp(colorizedBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const oD = origRaw.data;
  const cD = Buffer.from(colRaw.data); // mutable copy
  const w = origRaw.info.width;
  const h = origRaw.info.height;
  const ch = origRaw.info.channels;

  // Build binary maps for dark and white pixels
  const dark = new Uint8Array(w * h);
  const white = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * ch;
      const r = oD[p], g = oD[p + 1], b = oD[p + 2];
      if (r < BLACK_RESTORE_THRESHOLD && g < BLACK_RESTORE_THRESHOLD && b < BLACK_RESTORE_THRESHOLD) {
        dark[y * w + x] = 1;
      }
      if (r > WHITE_THRESHOLD && g > WHITE_THRESHOLD && b > WHITE_THRESHOLD) {
        white[y * w + x] = 1;
      }
    }
  }

  // Build summed area table for O(1) local density queries
  const sat = new Int32Array((w + 1) * (h + 1));
  const sw = w + 1;
  for (let y = 1; y <= h; y++) {
    for (let x = 1; x <= w; x++) {
      sat[y * sw + x] =
        dark[(y - 1) * w + (x - 1)] +
        sat[(y - 1) * sw + x] +
        sat[y * sw + (x - 1)] -
        sat[(y - 1) * sw + (x - 1)];
    }
  }

  function getLocalDensity(cx, cy) {
    const x1 = Math.max(0, cx - LOCAL_CHECK_RADIUS);
    const y1 = Math.max(0, cy - LOCAL_CHECK_RADIUS);
    const x2 = Math.min(w - 1, cx + LOCAL_CHECK_RADIUS);
    const y2 = Math.min(h - 1, cy + LOCAL_CHECK_RADIUS);
    const blockSize = (x2 - x1 + 1) * (y2 - y1 + 1);
    const a = (y2 + 1) * sw + (x2 + 1);
    const b = y1 * sw + (x2 + 1);
    const c = (y2 + 1) * sw + x1;
    const d = y1 * sw + x1;
    const darkCount = sat[a] - sat[b] - sat[c] + sat[d];
    return darkCount / blockSize;
  }

  // Check if a pixel is near a white region (speech bubble)
  function isNearWhite(cx, cy) {
    const r = BUBBLE_SEARCH_RADIUS;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          if (white[ny * w + nx]) return true;
        }
      }
    }
    return false;
  }

  // Track which dark pixels are edge-connected
  const edgeConnected = new Uint8Array(w * h);

  // Seed queue with all dark pixels on image borders
  const queue = [];
  for (let x = 0; x < w; x++) {
    if (dark[x]) queue.push(x);
    const bottomIdx = (h - 1) * w + x;
    if (dark[bottomIdx]) queue.push(bottomIdx);
  }
  for (let y = 1; y < h - 1; y++) {
    const leftIdx = y * w;
    if (dark[leftIdx]) queue.push(leftIdx);
    const rightIdx = y * w + (w - 1);
    if (dark[rightIdx]) queue.push(rightIdx);
  }

  for (const idx of queue) {
    edgeConnected[idx] = 1;
  }

  // Flood-fill to find all edge-connected dark pixels
  while (queue.length > 0) {
    const idx = queue.pop();
    const x = idx % w;
    const y = Math.floor(idx / w);

    const neighbors = [];
    if (x > 0) neighbors.push(idx - 1);
    if (x < w - 1) neighbors.push(idx + 1);
    if (y > 0) neighbors.push(idx - w);
    if (y < h - 1) neighbors.push(idx + w);

    for (const nIdx of neighbors) {
      if (dark[nIdx] && !edgeConnected[nIdx]) {
        edgeConnected[nIdx] = 1;
        queue.push(nIdx);
      }
    }
  }

  // Restore dark pixels that meet criteria
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!dark[idx]) continue;

      // Must be in a high-density region
      const density = getLocalDensity(x, y);
      if (density < LOCAL_DENSITY_MIN) continue;

      // Must be either edge-connected OR near a speech bubble
      if (!edgeConnected[idx] && !isNearWhite(x, y)) continue;

      const p = idx * ch;
      cD[p] = 0;
      cD[p + 1] = 0;
      cD[p + 2] = 0;
      cD[p + 3] = 255;
    }
  }

  return sharp(cD, {
    raw: { width: colRaw.info.width, height: colRaw.info.height, channels: ch },
  })
    .png()
    .toBuffer();
}

// Pick the best API size for a given aspect ratio.
// API only supports: 1024x1024, 1024x1536 (portrait), 1536x1024 (landscape)
// Picks the smallest API size that fits the aspect ratio without excessive upscaling.
function pickApiSize(w, h) {
  const ratio = w / h;
  const candidates = [
    { aw: 1024, ah: 1024 },   // square
    { aw: 1024, ah: 1536 },   // portrait
    { aw: 1536, ah: 1024 },   // landscape
  ];

  // Score each candidate: prefer the one where the scale factor is closest to 1
  // (i.e., least upscaling needed) while still fitting the aspect ratio reasonably.
  let best = candidates[0];
  let bestScore = Infinity;
  for (const c of candidates) {
    const scale = Math.min(c.aw / w, c.ah / h);
    const aspectDiff = Math.abs(c.aw / c.ah - ratio);
    // Penalize heavy upscaling (scale > 2) — prefer smaller API size
    const upscalePenalty = scale > 2 ? scale : 0;
    const score = aspectDiff + upscalePenalty;
    if (score < bestScore) {
      bestScore = score;
      best = c;
    }
  }

  return best;
}

async function colorizeSegment(segBuf, index, total, prompt) {
  const meta = await sharp(segBuf).metadata();
  const origW = meta.width;
  const origH = meta.height;

  // Downscale segment if wider than the quality tier allows.
  // This reduces both vision tokens (orchestrator) and image gen cost.
  let workBuf = segBuf;
  let workW = origW;
  let workH = origH;
  if (origW > TIER.maxWidth) {
    const downscale = TIER.maxWidth / origW;
    workW = TIER.maxWidth;
    workH = Math.round(origH * downscale);
    workBuf = await sharp(segBuf)
      .resize(workW, workH, { fit: "fill", kernel: "lanczos3" })
      .png()
      .toBuffer();
  }

  // Choose the API output size that best matches the working aspect ratio
  const { aw, ah } = pickApiSize(workW, workH);
  const apiSize = `${aw}x${ah}`;

  // Resize to fit within the API dimensions, then pad with black.
  const scale = Math.min(aw / workW, ah / workH);
  const fitW = Math.round(workW * scale);
  const fitH = Math.round(workH * scale);

  const sendBuf = await sharp(workBuf)
    .resize(fitW, fitH, { fit: "fill", kernel: "lanczos3" })
    .extend({
      top: 0,
      left: 0,
      right: aw - fitW,
      bottom: ah - fitH,
      background: { r: 0, g: 0, b: 0, alpha: 1 },
    })
    .png()
    .toBuffer();

  console.log(`    Prepared ${origW}x${origH} → ${workW}x${workH} → ${fitW}x${fitH} padded to ${aw}x${ah}`);

  const content = [
    {
      type: "input_image",
      image_url: toDataUrl(sendBuf),
      detail: "high",
    },
    {
      type: "input_text",
      text: prompt,
    },
  ];

  const res = await client.responses.create({
    model: "gpt-5.2",
    instructions: SYSTEM_INSTRUCTIONS,
    input: [{ role: "user", content }],
    tools: [
      {
        type: "image_generation",
        action: "edit",
        quality: QUALITY,
        input_fidelity: "high",
        size: apiSize,
        output_format: "png",
      },
    ],
  });

  // Find image output
  let b64 = null;
  for (const item of res.output) {
    if (item.type === "image_generation_call") {
      b64 = item.result;
      break;
    }
  }

  if (!b64) {
    const types = res.output.map((o) => o.type).join(", ");
    for (const item of res.output) {
      if (item.type === "message" && item.content) {
        for (const c of item.content) {
          if (c.text) console.error(`  API message: ${c.text}`);
        }
      }
    }
    throw new Error(
      `Segment ${index + 1}/${total}: No image in response. Output types: ${types}`
    );
  }

  const apiOut = Buffer.from(b64, "base64");

  // Crop out the padding (still at working resolution)
  const cropped = await sharp(apiOut)
    .extract({ left: 0, top: 0, width: fitW, height: fitH })
    .resize(workW, workH, { fit: "fill", kernel: "lanczos3" })
    .png()
    .toBuffer();

  // Restore blacks at working resolution so both images are at the same scale.
  // This avoids jagged artifacts from comparing crisp originals against upscaled output.
  const restored = RESTORE_BLACKS ? await restoreBlacks(workBuf, cropped) : cropped;

  // Upscale to original full-res dimensions
  if (workW !== origW || workH !== origH) {
    return sharp(restored)
      .resize(origW, origH, { fit: "fill", kernel: "lanczos3" })
      .png()
      .toBuffer();
  }
  return restored;
}

// ── Step 5: Reassemble colorized segments ──────────────────────────────────

async function reassembleSegments(segments, width) {
  const totalH = segments.reduce((sum, s) => sum + s.height, 0);

  let y = 0;
  const composites = [];
  for (const seg of segments) {
    composites.push({ input: seg.buffer, top: y, left: 0 });
    y += seg.height;
  }

  const reassembled = await sharp({
    create: {
      width,
      height: totalH,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 1 },
    },
  })
    .composite(composites)
    .png()
    .toBuffer();

  return reassembled;
}

// ── Step 6: Re-slice to match original input slice dimensions ───────────

async function reslice(reassembledBuf, width, originalHeights) {
  const slices = [];
  let top = 0;

  for (let i = 0; i < originalHeights.length; i++) {
    const sliceH = originalHeights[i];
    const remaining = (await sharp(reassembledBuf).metadata()).height - top;
    if (remaining <= 0) break;

    const extractH = Math.min(sliceH, remaining);
    let slice = await sharp(reassembledBuf)
      .extract({ left: 0, top, width, height: extractH })
      .png()
      .toBuffer();

    // If extracted region is shorter than the original slice, pad with black
    if (extractH < sliceH) {
      slice = await sharp({
        create: {
          width,
          height: sliceH,
          channels: 4,
          background: { r: 0, g: 0, b: 0, alpha: 1 },
        },
      })
        .composite([{ input: slice, top: 0, left: 0 }])
        .png()
        .toBuffer();
    }

    // Resize to output dimensions if needed
    if (width !== OUTPUT_W || sliceH !== OUTPUT_H) {
      slice = await sharp(slice)
        .resize(OUTPUT_W, OUTPUT_H, { fit: "fill" })
        .png()
        .toBuffer();
    }

    slices.push(slice);
    top += sliceH;
  }

  return slices;
}

// ── Debug helpers ──────────────────────────────────────────────────────────

async function debugSave(name, buf) {
  if (!DEBUG) return;
  await ensureDir(DEBUG_DIR);
  await fsp.writeFile(path.join(DEBUG_DIR, name), buf);
  console.log(`  [debug] saved ${name}`);
}

// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
  if (!process.env.OPENAI_API_KEY)
    throw new Error("Missing OPENAI_API_KEY in .env");

  console.log(`Quality: ${QUALITY} | Restore blacks: ${RESTORE_BLACKS} | Context: ${CAPTURE_CONTEXT}`);

  const BASE_PROMPT = await loadPalette();
  const contextEntries = await loadContext();
  const PROMPT = buildPromptWithContext(BASE_PROMPT, contextEntries);

  await ensureDir(OUTPUT_DIR);

  // 1. Load input slices
  const files = await listImages(INPUT_DIR);
  const targets = files
    .map((f) => ({ file: f, ...parseName(f) }))
    .sort((a, b) => a.idx - b.idx);

  if (targets.length === 0) throw new Error("No PNG files found in " + INPUT_DIR);

  const inFiles = targets.map((t) => t.file);
  console.log(`Found ${inFiles.length} input slices`);

  // 2. Stitch into one continuous strip
  console.log("Stitching slices...");
  const { buffer: stitchedBuf, width, totalH, heights } =
    await stitchSlices(inFiles);
  console.log(`  Stitched: ${width}x${totalH}`);
  await debugSave("01_stitched.png", stitchedBuf);

  // 3. Detect safe split points
  console.log("Detecting safe split points...");
  const splitPoints = await detectSafeSplitPoints(stitchedBuf);
  console.log(
    `  Found ${splitPoints.length} split points:`,
    splitPoints.map((sp) => `row ${sp.midPoint} (gap ${sp.height}px)`).join(", ") ||
      "none"
  );

  // 4. Split into self-contained segments
  console.log("Splitting into segments...");
  const segments = await splitAtPoints(stitchedBuf, width, totalH, splitPoints);
  console.log(`  Created ${segments.length} segments`);
  for (let i = 0; i < segments.length; i++) {
    console.log(`    Segment ${i + 1}: ${segments[i].width}x${segments[i].height}`);
    await debugSave(`02_segment_${pad(i + 1)}_input.png`, segments[i].buffer);
  }

  // 5. Colorize each segment
  let apiCalls = 0;
  let skippedSegments = 0;
  const failedIndices = [];
  console.log("Colorizing segments...");
  const colorizedSegments = [];
  for (let i = 0; i < segments.length; i++) {
    const progress = `[${i + 1}/${segments.length}, ${apiCalls} API calls, ${failedIndices.length} failed, ${skippedSegments} skipped]`;
    console.log(
      `  ${progress} Segment ${i + 1} (${segments[i].width}x${segments[i].height})...`
    );

    // Check if segment is blank/text-on-black before making API call
    const blankReason = await isBlankSegment(segments[i].buffer);
    let colorized = null;

    if (blankReason) {
      console.log(`    ${blankReason} — skipping API call`);
      colorized = segments[i].buffer;
      skippedSegments++;
    } else {
      colorized = await withRetry(
        () => {
          apiCalls++;
          return colorizeSegment(segments[i].buffer, i, segments.length, PROMPT);
        },
        `Segment ${i + 1}/${segments.length}`
      );

      if (colorized === null) {
        // All retries exhausted — fall back to B&W
        console.warn(`    → Using original B&W for segment ${i + 1}`);
        colorized = segments[i].buffer;
        failedIndices.push(i + 1);
      } else if (CAPTURE_CONTEXT) {
        // Capture color decisions for unlocked elements
        const newColors = await captureContext(colorized, BASE_PROMPT);
        if (newColors.length > 0) {
          // Deduplicate against existing entries
          const existing = new Set(contextEntries.map((e) => e.toLowerCase()));
          const unique = newColors.filter((c) => !existing.has(c.toLowerCase()));
          if (unique.length > 0) {
            contextEntries.push(...unique);
            console.log(`    Context: +${unique.length} learned (${contextEntries.length} total)`);
          }
        }
      }
    }

    colorizedSegments.push({
      buffer: colorized,
      height: segments[i].height,
    });
    await debugSave(`03_segment_${pad(i + 1)}_colorized.png`, colorized);
  }

  // Final summary
  console.log(`\nAPI summary: ${apiCalls} calls, ${skippedSegments} skipped, ${failedIndices.length} failed (quality: ${QUALITY})`);
  if (failedIndices.length > 0) {
    console.warn(`Failed segments (fell back to B&W): ${failedIndices.join(", ")}`);
  }

  // Save learned context
  if (CAPTURE_CONTEXT && contextEntries.length > 0) {
    await saveContext(contextEntries);
    console.log(`Context saved: ${contextEntries.length} entries → ${CONTEXT_PATH}`);
  }

  // 6. Reassemble
  console.log("Reassembling...");
  const reassembled = await reassembleSegments(colorizedSegments, width);
  await debugSave("04_reassembled.png", reassembled);

  // 7. Re-slice to match original input slice dimensions
  console.log(`Re-slicing to ${OUTPUT_W}x${OUTPUT_H}...`);
  const outputSlices = await reslice(reassembled, width, heights);

  // 8. Save output
  const { key: namePrefix, idx: startIdx, ext } = targets[0];
  const idxDigits = String(targets[targets.length - 1].idx).length;

  const isJpg = /\.jpe?g$/i.test(ext);

  for (let i = 0; i < outputSlices.length; i++) {
    const idx = startIdx + i;
    const outName = `${namePrefix}${pad(idx, Math.max(idxDigits, 3))}${ext}`;
    const outPath = path.join(OUTPUT_DIR, outName);
    const outBuf = isJpg
      ? await sharp(outputSlices[i]).jpeg({ quality: 95 }).toBuffer()
      : outputSlices[i];
    await fsp.writeFile(outPath, outBuf);
    console.log(`  Saved ${outName}`);
  }

  console.log(
    `Done. ${outputSlices.length} slices saved to ${OUTPUT_DIR}`
  );
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
