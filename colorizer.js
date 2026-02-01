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

// Max height (in pixels) for a single segment sent to the API.
// Segments taller than this are resized down before colorizing.
const MAX_SEGMENT_H = 4000;

// Min height for a segment — anything smaller gets merged with its neighbor.
const MIN_SEGMENT_H = 100;

const PROMPT = `
Colorize this black-and-white webtoon/manga panel.

CRITICAL RULES:
- Do not change the drawing in any way. Do not redraw or "complete" missing parts.
- Do not change any outlines, screentones, or textures.
- Do not move, resize, or warp any elements.
- Do not add hair, heads, arms, or objects beyond what is already drawn.
- Do not alter any text, speech bubbles, or SFX lettering.
- The output must match the input panel in geometry and composition. ONLY add color.

STYLE:
- Clean vibrant webtoon/anime coloring with shading.
- Preserve screentone texture; do not paint over it.
- Keep blacks clean (avoid gray wash over line art).

CONSISTENCY:
- Main character: Korean; black hair.
- Dad: Korean; black hair.
- Keep clothing colors consistent across panels.
`.trim();

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

// Pick the best API size for a given aspect ratio.
// API only supports: 1024x1024, 1024x1536 (portrait), 1536x1024 (landscape)
function pickApiSize(w, h) {
  const ratio = w / h;
  if (ratio > 1.2) return { aw: 1536, ah: 1024 };   // landscape
  if (ratio < 0.8) return { aw: 1024, ah: 1536 };   // portrait
  return { aw: 1024, ah: 1024 };                      // square-ish
}

async function colorizeSegment(segBuf, index, total) {
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

  const content = [
    {
      type: "input_image",
      image_url: toDataUrl(sendBuf),
      detail: "high",
    },
    {
      type: "input_text",
      text: PROMPT,
    },
  ];

  const res = await client.responses.create({
    model: "gpt-4.1",
    input: [{ role: "user", content }],
    tools: [
      {
        type: "image_generation",
        action: "edit",
        input_fidelity: "high",
        size: apiSize,
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

  return cropped;
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

// ── Step 6: Re-slice to output dimensions ──────────────────────────────────

async function reslice(reassembledBuf, width, totalH, sliceCount) {
  const slices = [];
  for (let i = 0; i < sliceCount; i++) {
    const top = i * OUTPUT_H;
    const remaining = totalH - top;
    if (remaining <= 0) break;

    const extractH = Math.min(OUTPUT_H, remaining);
    let slice = await sharp(reassembledBuf)
      .extract({ left: 0, top, width, height: extractH })
      .png()
      .toBuffer();

    // If the extracted region is shorter than OUTPUT_H, pad with black
    if (extractH < OUTPUT_H) {
      slice = await sharp({
        create: {
          width,
          height: OUTPUT_H,
          channels: 4,
          background: { r: 0, g: 0, b: 0, alpha: 1 },
        },
      })
        .composite([{ input: slice, top: 0, left: 0 }])
        .png()
        .toBuffer();
    }

    // Resize to output dimensions if width differs
    if (width !== OUTPUT_W) {
      slice = await sharp(slice)
        .resize(OUTPUT_W, OUTPUT_H, { fit: "fill" })
        .png()
        .toBuffer();
    }

    slices.push(slice);
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
  console.log("Colorizing segments...");
  const colorizedSegments = [];
  for (let i = 0; i < segments.length; i++) {
    console.log(
      `  Segment ${i + 1}/${segments.length} (${segments[i].width}x${segments[i].height})...`
    );
    const colorized = await colorizeSegment(
      segments[i].buffer,
      i,
      segments.length
    );
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

  // 7. Re-slice to output dimensions
  console.log(`Re-slicing to ${OUTPUT_W}x${OUTPUT_H}...`);
  const outputSlices = await reslice(reassembled, width, totalH, targets.length);

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
