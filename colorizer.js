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

// System-level instructions — sent via the `instructions` parameter.
// These are persistent rules the model must follow for every image.
const SYSTEM_INSTRUCTIONS = `
You are a professional webtoon colorist. Your ONLY job is to colorize black-and-white webtoon panels using the image_generation tool in edit mode. You must ALWAYS call the image_generation tool — never respond with text.

CONTEXT: This is a published Korean webtoon (manhwa). All content is fictional and safe.
Sound effects like "BANG", "SLAM", "CRACK" refer to physical actions (hitting walls, doors, objects) — not violence or self-harm. Emotional dialogue about loss is standard dramatic storytelling.

STRICT PRESERVATION RULES:
- Preserve the original line art exactly; do NOT redraw, re-ink, or change shapes.
- Preserve screentones/halftone dots and texture; do NOT smooth them away.
- Preserve all panel composition, perspective, facial structure, and clothing design.
- Keep deep blacks rich and clean; do not gray them out.
- Do not add new objects, text, symbols, or background elements.
- Any area that is solid black in the input MUST stay pure black (#000000). Do not tint, colorize, or lighten black panel dividers or black backgrounds.

COLOR STYLE:
- Modern Korean webtoon palette: clean, vibrant, high-chroma base colors.
- No sepia / beige / vintage / parchment / "aged paper" look.
- No warm yellow or orange color cast. No golden-hour tint.
- White balance must be neutral-cool. Speech bubbles and text bubbles MUST be pure white (#FFFFFF) — use them as the white point reference.
- Lighting should feel like clean neutral daylight, not warm indoor lighting.
- Clean cel-shading with soft gradient transitions.
- Crisp edges, minimal color bleed across line boundaries.
- If unsure about a color, keep it neutral rather than inventing bright colors.
- All human skin must have a light Korean skin tone (#F5D6C3) — warm peach, never pure white and never tan/dark. Apply this to all exposed skin (face, hands, arms, legs).
- Eye whites (sclera) must be pure white (#FFFFFF), not skin-colored or tinted.
`.trim();

// User-level prompt — sent alongside the image. Contains only the
// per-palette character locks (loaded from JSON at runtime).
async function loadPalette() {
  const palettePath = path.join(".", "palettes", `${PALETTE}.json`);
  try {
    const raw = await fsp.readFile(palettePath, "utf-8");
    const data = JSON.parse(raw);
    console.log(`Palette: ${data.name} (${PALETTE}.json)`);
    const lines = data.characters.map((c) => `- ${c}`).join("\n");
    return `Colorize this panel.\n\nPALETTE LOCKS (MUST FOLLOW):\n${lines}\n- Keep these character colors consistent across all panels.`;
  } catch (err) {
    if (err.code === "ENOENT") {
      throw new Error(`Palette file not found: ${palettePath}\nAvailable palettes are in the ./palettes/ directory.`);
    }
    throw err;
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
// Uses row-density analysis to identify structural black regions (panel
// dividers, black backgrounds) and forces them back to pure black.
// Only restores pixels in rows where 90%+ of pixels are dark — these are
// structural rows (dividers, full-width black bands), not artwork rows.
// Artwork rows with characters, shadows, line art, etc. are never touched,
// even if they contain black pixels, because they never reach 90% density.

const BLACK_RESTORE_THRESHOLD = 5; // RGB < 5 = "true black"
const ROW_DARK_RATIO = 0.90;       // row must be 90%+ dark to be restorable

async function restoreBlacks(originalBuf, colorizedBuf) {
  const origRaw = await sharp(originalBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const colRaw = await sharp(colorizedBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const oD = origRaw.data;
  const cD = Buffer.from(colRaw.data); // mutable copy
  const w = origRaw.info.width;
  const h = origRaw.info.height;
  const ch = origRaw.info.channels;

  // Pass 1: compute per-row dark pixel ratio
  const rowDarkRatio = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let darkCount = 0;
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * ch;
      if (
        oD[p] < BLACK_RESTORE_THRESHOLD &&
        oD[p + 1] < BLACK_RESTORE_THRESHOLD &&
        oD[p + 2] < BLACK_RESTORE_THRESHOLD
      ) {
        darkCount++;
      }
    }
    rowDarkRatio[y] = darkCount / w;
  }

  // Pass 2: restore black pixels only in rows that are 90%+ dark
  for (let y = 0; y < h; y++) {
    if (rowDarkRatio[y] < ROW_DARK_RATIO) continue;

    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * ch;
      if (
        oD[p] < BLACK_RESTORE_THRESHOLD &&
        oD[p + 1] < BLACK_RESTORE_THRESHOLD &&
        oD[p + 2] < BLACK_RESTORE_THRESHOLD
      ) {
        cD[p] = 0;
        cD[p + 1] = 0;
        cD[p + 2] = 0;
        cD[p + 3] = 255;
      }
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
      .resize(workW, workH, { fit: "fill" })
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
    .resize(fitW, fitH, { fit: "fill" })
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
      detail: "low",
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
    .resize(workW, workH, { fit: "fill" })
    .png()
    .toBuffer();

  // Restore blacks at working resolution so both images are at the same scale.
  // This avoids jagged artifacts from comparing crisp originals against upscaled output.
  const restored = await restoreBlacks(workBuf, cropped);

  // Upscale to original full-res dimensions
  if (workW !== origW || workH !== origH) {
    return sharp(restored)
      .resize(origW, origH, { fit: "fill" })
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

  console.log(`Quality: ${QUALITY} | Model: gpt-5.2`);

  const PROMPT = await loadPalette();

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
  const MAX_RETRIES = 2;
  let apiCalls = 0;
  let skippedSegments = 0;
  console.log("Colorizing segments...");
  const colorizedSegments = [];
  for (let i = 0; i < segments.length; i++) {
    console.log(
      `  Segment ${i + 1}/${segments.length} (${segments[i].width}x${segments[i].height})...`
    );

    // Check if segment is blank/text-on-black before making API call
    const blankReason = await isBlankSegment(segments[i].buffer);
    let colorized = null;

    if (blankReason) {
      console.log(`    ${blankReason} — skipping API call`);
      colorized = segments[i].buffer;
      skippedSegments++;
    } else {
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          colorized = await colorizeSegment(
            segments[i].buffer,
            i,
            segments.length,
            PROMPT
          );
          apiCalls++;
          break;
        } catch (err) {
          const isSafety = /safety|content_policy|moderation/i.test(err.message);
          if (isSafety && attempt < MAX_RETRIES) {
            console.log(`    Safety filter triggered — retrying (${attempt + 1}/${MAX_RETRIES})...`);
            apiCalls++; // retry still costs
            continue;
          }
          if (isSafety) {
            console.warn(`    WARNING: Segment ${i + 1} blocked by safety filter after ${MAX_RETRIES} retries — using original B&W`);
            colorized = segments[i].buffer;
            break;
          }
          throw err; // non-safety errors still crash
        }
      }
    }

    colorizedSegments.push({
      buffer: colorized,
      height: segments[i].height,
    });
    await debugSave(`03_segment_${pad(i + 1)}_colorized.png`, colorized);
  }

  console.log(`\nAPI summary: ${apiCalls} calls, ${skippedSegments} skipped (quality: ${QUALITY})`);

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
