import "dotenv/config";
import fsp from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const INPUT_DIR = "./input";
const OUTPUT_DIR = "./output";

// Context strip height (in ORIGINAL panel coords).
const STRIP_H = 256;

// If true, writes debug images to ./debug to inspect framing decisions.
const DEBUG = false;

function buildPrompt(hasStripAbove, hasStripBelow) {
  let context = "";
  if (hasStripAbove && hasStripBelow) {
    context =
      "I'm giving you 3 images. The FIRST image is the panel to colorize. " +
      "The SECOND image is a thin strip from the panel ABOVE (for color context only). " +
      "The THIRD image is a thin strip from the panel BELOW (for color context only). " +
      "Only colorize the FIRST image and return it.";
  } else if (hasStripAbove) {
    context =
      "I'm giving you 2 images. The FIRST image is the panel to colorize. " +
      "The SECOND image is a thin strip from the panel ABOVE (for color context only). " +
      "Only colorize the FIRST image and return it.";
  } else if (hasStripBelow) {
    context =
      "I'm giving you 2 images. The FIRST image is the panel to colorize. " +
      "The SECOND image is a thin strip from the panel BELOW (for color context only). " +
      "Only colorize the FIRST image and return it.";
  } else {
    context = "Colorize this image.";
  }

  return `
${context}

These are vertical webtoon slices. Characters may be cut off by the panel edges.

CRITICAL RULES:
- Do not change the drawing in any way. Do not redraw or "complete" missing parts.
- Do not change any outlines, screentones, or textures.
- Do not move/resize/warp any elements.
- Do not add hair/heads/arms/objects beyond what is already drawn in the target panel.
- Do not alter any text, speech bubbles, or SFX lettering.
- The output must match the target panel in geometry and composition; ONLY add color.

STYLE:
- Clean vibrant webtoon/anime coloring with restrained shading.
- Preserve screentone texture; do not paint over it.
- Keep blacks clean.

CONSISTENCY:
- Main character: Korean; black hair.
- Dad: Korean; black hair.
- Keep clothing colors consistent across panels.
`.trim();
}

function parseName(filename) {
  const base = path.basename(filename);
  const m = base.match(/^(.*?)(\d+)(\.[^.]+)$/);
  if (!m) return { key: base, idx: 0 };
  return { key: m[1], idx: Number(m[2]) };
}

async function ensureDir(dir) {
  await fsp.mkdir(dir, { recursive: true });
}

async function listPngs(dir) {
  const files = await fsp.readdir(dir);
  return files.filter((f) => /\.png$/i.test(f)).map((f) => path.join(dir, f));
}

async function getNonBlackBBox(buf, threshold = 16) {
  const { data, info } = await sharp(buf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const w = info.width;
  const h = info.height;
  const c = info.channels;

  let minX = w, minY = h, maxX = -1, maxY = -1;

  for (let y = 0; y < h; y++) {
    const row = y * w * c;
    for (let x = 0; x < w; x++) {
      const i = row + x * c;
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      if (a < 10) continue;
      if (r > threshold || g > threshold || b > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) return { left: 0, top: 0, width: w, height: h };
  return { left: minX, top: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

async function matchFramingToInput(inputBuf, outputBuf) {
  const inMeta = await sharp(inputBuf).metadata();
  const W = inMeta.width;
  const H = inMeta.height;

  const inBox = await getNonBlackBBox(inputBuf);
  const outBox = await getNonBlackBBox(outputBuf);

  const outCropped = await sharp(outputBuf).extract(outBox).png().toBuffer();

  const outResized = await sharp(outCropped)
    .resize(inBox.width, inBox.height, { fit: "fill" })
    .png()
    .toBuffer();

  return sharp({
    create: {
      width: W,
      height: H,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 1 },
    },
  })
    .composite([{ input: outResized, left: inBox.left, top: inBox.top }])
    .png()
    .toBuffer();
}

/** Convert a buffer to a base64 data URL for the Responses API. */
function toDataUrl(buf) {
  return `data:image/png;base64,${buf.toString("base64")}`;
}

async function makeContextStrip(neighborPath, fromAbove) {
  const buf = await fsp.readFile(neighborPath);
  const meta = await sharp(buf).metadata();
  const W = meta.width;
  const H = meta.height;

  const top = fromAbove ? Math.max(0, H - STRIP_H) : 0;
  const height = Math.min(STRIP_H, H);

  const strip = await sharp(buf)
    .extract({ left: 0, top, width: W, height })
    .png()
    .toBuffer();

  return strip;
}

async function colorizePanel(panelPath, abovePath, belowPath) {
  const inputBuf = await fsp.readFile(panelPath);
  const inMeta = await sharp(inputBuf).metadata();

  // Build the content array for the Responses API
  const content = [];

  // Target panel image (always first)
  content.push({
    type: "input_image",
    image_url: toDataUrl(inputBuf),
    detail: "high",
  });

  let hasAbove = false;
  let hasBelow = false;

  if (abovePath) {
    const strip = await makeContextStrip(abovePath, true);
    content.push({
      type: "input_image",
      image_url: toDataUrl(strip),
      detail: "low",
    });
    hasAbove = true;
  }

  if (belowPath) {
    const strip = await makeContextStrip(belowPath, false);
    content.push({
      type: "input_image",
      image_url: toDataUrl(strip),
      detail: "low",
    });
    hasBelow = true;
  }

  // Add the text prompt
  content.push({
    type: "input_text",
    text: buildPrompt(hasAbove, hasBelow),
  });

  // Call the Responses API with image generation tool
  const res = await client.responses.create({
    model: "gpt-4o",
    input: [{ role: "user", content }],
    tools: [{ type: "image_generation" }],
  });

  // Find the image output in the response
  let b64 = null;
  for (const item of res.output) {
    if (item.type === "image_generation_call") {
      b64 = item.result;
      break;
    }
  }

  if (!b64) {
    // Log what we got back for debugging
    const types = res.output.map((o) => o.type).join(", ");
    throw new Error(`No image in response. Output types: ${types}`);
  }

  const apiOut = Buffer.from(b64, "base64");

  // Restore to original dimensions
  const restored = await sharp(apiOut)
    .resize(inMeta.width, inMeta.height, { fit: "fill" })
    .png()
    .toBuffer();

  // Fix framing if model shifted content
  const framed = await matchFramingToInput(inputBuf, restored);

  if (DEBUG) {
    await ensureDir("./debug");
    const base = path.basename(panelPath, path.extname(panelPath));
    await fsp.writeFile(`./debug/${base}_input.png`, inputBuf);
    await fsp.writeFile(`./debug/${base}_api.png`, apiOut);
    await fsp.writeFile(`./debug/${base}_restored.png`, restored);
    await fsp.writeFile(`./debug/${base}_framed.png`, framed);
  }

  return framed;
}

async function main() {
  if (!process.env.OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY in .env");

  await ensureDir(OUTPUT_DIR);

  const files = await listPngs(INPUT_DIR);

  const targets = files
    .map((f) => ({ file: f, ...parseName(f) }))
    .filter((x) => x.key.includes("Damashi Game Chapter - 001A_Output_"))
    .sort((a, b) => a.idx - b.idx);

  if (targets.length === 0) throw new Error("No matching files found in ./input");

  const inFiles = targets.map((t) => t.file);

  for (let i = 0; i < inFiles.length; i++) {
    const name = path.basename(inFiles[i]);
    const above = i > 0 ? inFiles[i - 1] : null;
    const below = i < inFiles.length - 1 ? inFiles[i + 1] : null;

    console.log(`Colorizing: ${name}...`);
    const outBuf = await colorizePanel(inFiles[i], above, below);
    await fsp.writeFile(path.join(OUTPUT_DIR, name), outBuf);
    console.log(`  ✅ Saved ${name}`);
  }

  console.log("Done. Check ./output");
}

main().catch((err) => {
  console.error("❌", err.message);
  process.exit(1);
});
