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

// Credit saver mode: uses cheaper orchestrator model.
// Image generation quality (gpt-image-1.5) stays the same.
const CREDIT_SAVER = false; // do not change to true

// Character palette — selects a JSON file from ./palettes/
const PALETTE = process.env.PALETTE || "intro";

// Max height (in pixels) for a single segment sent to the API.
// Segments taller than this are resized down before colorizing.
const MAX_SEGMENT_H = 4000;

// Min height for a segment — anything smaller gets merged with its neighbor.
const MIN_SEGMENT_H = 100;

const PROMPT_BASE = `
COLORIZE THIS BLACK-AND-WHITE WEBTOON PANEL.

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
`.trim();

async function loadPalette() {
  const palettePath = path.join(".", "palettes", `${PALETTE}.json`);
  try {
    const raw = await fsp.readFile(palettePath, "utf-8");
    const data = JSON.parse(raw);
    console.log(`Palette: ${data.name} (${PALETTE}.json)`);
    const lines = data.characters.map((c) => `- ${c}`).join("\n");
    return `${PROMPT_BASE}\n\nPALETTE LOCKS (MUST FOLLOW):\n${lines}\n- Keep these character colors consistent across all panels.`;
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

async function listPngs(dir) {
  const files = await fsp.readdir(dir);
  return files
    .filter((f) => /\.png$/i.test(f))
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
  for (let i = 0; i < total; i++) {
    const p = i * info.channels;
    if (
      data[p] < DARK_THRESHOLD &&
      data[p + 1] < DARK_THRESHOLD &&
      data[p + 2] < DARK_THRESHOLD
    ) {
      darkCount++;
    }
  }
  return darkCount / total >= 0.98;
}

// ── Post-process: restore black pixels from original ─────────────────────
// Uses a strict threshold (RGB < 5) to only restore truly black pixels
// (panel dividers, solid black backgrounds) without affecting dark artwork
// shadows or screentone that the AI has meaningfully colored.
// A pixel is restored if it was near-pure-black in the original AND is NOT
// adjacent to colorful artwork (detected by checking if neighbors in the
// original were also black — isolated dark pixels in art are left alone).

const BLACK_RESTORE_THRESHOLD = 5; // Much stricter than DARK_THRESHOLD (20)

async function restoreBlacks(originalBuf, colorizedBuf) {
  const origRaw = await sharp(originalBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const colRaw = await sharp(colorizedBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const oD = origRaw.data;
  const cD = Buffer.from(colRaw.data); // mutable copy
  const w = origRaw.info.width;
  const h = origRaw.info.height;
  const c = origRaw.info.channels;

  // First pass: mark which pixels are "true black" in the original
  const isBlack = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const p = i * c;
    if (
      oD[p] < BLACK_RESTORE_THRESHOLD &&
      oD[p + 1] < BLACK_RESTORE_THRESHOLD &&
      oD[p + 2] < BLACK_RESTORE_THRESHOLD
    ) {
      isBlack[i] = 1;
    }
  }

  // Second pass: only restore a black pixel if enough of its neighbors
  // (in a 5x5 area) are also black. This ensures we restore large black
  // regions (dividers, backgrounds) but leave isolated dark pixels in
  // artwork (shadows, screentone, line art edges) untouched.
  const RADIUS = 2;
  const MIN_NEIGHBORS = 15; // out of 25 (5x5 area) = 60%

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!isBlack[idx]) continue;

      // Count black neighbors
      let blackNeighbors = 0;
      for (let dy = -RADIUS; dy <= RADIUS; dy++) {
        for (let dx = -RADIUS; dx <= RADIUS; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            if (isBlack[ny * w + nx]) blackNeighbors++;
          }
        }
      }

      if (blackNeighbors >= MIN_NEIGHBORS) {
        const p = idx * c;
        cD[p] = 0;
        cD[p + 1] = 0;
        cD[p + 2] = 0;
        cD[p + 3] = 255;
      }
    }
  }

  return sharp(cD, {
    raw: { width: colRaw.info.width, height: colRaw.info.height, channels: c },
  })
    .png()
    .toBuffer();
}

// Pick the best API size for a given aspect ratio.
// API only supports: 1024x1024, 1024x1536 (portrait), 1536x1024 (landscape)
function pickApiSize(w, h) {
  const ratio = w / h;
  if (ratio > 1.2) return { aw: 1536, ah: 1024 };   // landscape
  if (ratio < 0.8) return { aw: 1024, ah: 1536 };   // portrait
  return { aw: 1024, ah: 1024 };                      // square-ish
}

async function colorizeSegment(segBuf, index, total, prompt) {
  // Skip segments that are almost entirely black (nothing to colorize)
  if (await isBlankSegment(segBuf)) {
    console.log(`  Segment ${index + 1}/${total}: blank/black — skipping API call`);
    return segBuf;
  }

  const meta = await sharp(segBuf).metadata();
  const origW = meta.width;
  const origH = meta.height;

  // Choose the API output size that best matches this segment's aspect ratio
  const { aw, ah } = pickApiSize(origW, origH);
  const apiSize = `${aw}x${ah}`;

  // Resize segment to fit within the API dimensions while preserving aspect ratio,
  // then pad with black to exactly match the API size.
  const scale = Math.min(aw / origW, ah / origH);
  const fitW = Math.round(origW * scale);
  const fitH = Math.round(origH * scale);

  const sendBuf = await sharp(segBuf)
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

  console.log(`    Prepared ${origW}x${origH} → ${fitW}x${fitH} padded to ${aw}x${ah}`);

  const orchModel = CREDIT_SAVER ? "gpt-4.1" : "gpt-5.2";
  const fidelity = "high";
  const visionDetail = "high";

  const content = [
    {
      type: "input_image",
      image_url: toDataUrl(sendBuf),
      detail: visionDetail,
    },
    {
      type: "input_text",
      text: prompt,
    },
  ];

  const res = await client.responses.create({
    model: orchModel,
    input: [{ role: "user", content }],
    tools: [
      {
        type: "image_generation",
        action: "edit",
        input_fidelity: fidelity,
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
    // Log the full response for debugging
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

  // Crop out the padding, then resize back to exact original dimensions
  const cropped = await sharp(apiOut)
    .extract({ left: 0, top: 0, width: fitW, height: fitH })
    .resize(origW, origH, { fit: "fill" })
    .png()
    .toBuffer();

  // Force any pixel that was black in the original back to pure black
  const restored = await restoreBlacks(segBuf, cropped);

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

  console.log(`Mode: ${CREDIT_SAVER ? "CREDIT SAVER (gpt-4.1)" : "QUALITY (gpt-5.2)"}`);

  const PROMPT = await loadPalette();

  await ensureDir(OUTPUT_DIR);

  // 1. Load input slices
  const files = await listPngs(INPUT_DIR);
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
  console.log("Colorizing segments...");
  const colorizedSegments = [];
  for (let i = 0; i < segments.length; i++) {
    console.log(
      `  Segment ${i + 1}/${segments.length} (${segments[i].width}x${segments[i].height})...`
    );

    let colorized = null;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        colorized = await colorizeSegment(
          segments[i].buffer,
          i,
          segments.length,
          PROMPT
        );
        break;
      } catch (err) {
        const isSafety = /safety|content_policy|moderation/i.test(err.message);
        if (isSafety && attempt < MAX_RETRIES) {
          console.log(`    Safety filter triggered — retrying (${attempt + 1}/${MAX_RETRIES})...`);
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

    colorizedSegments.push({
      buffer: colorized,
      height: segments[i].height,
    });
    await debugSave(`03_segment_${pad(i + 1)}_colorized.png`, colorized);
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

  for (let i = 0; i < outputSlices.length; i++) {
    const idx = startIdx + i;
    const outName = `${namePrefix}${pad(idx, Math.max(idxDigits, 3))}${ext}`;
    const outPath = path.join(OUTPUT_DIR, outName);
    await fsp.writeFile(outPath, outputSlices[i]);
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
