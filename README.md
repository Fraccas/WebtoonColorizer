# WebtoonColorizer

Automatic colorization of black-and-white webtoon panels using OpenAI's GPT-5.2 + gpt-image-1.5 with smart panel detection.

## The Problem

Webtoons are continuous vertical strips arbitrarily sliced at fixed dimensions (e.g., 800x1280). Characters and scenes bleed across slice boundaries — a character's head might be at the bottom of one slice and their body at the top of the next.

Generative AI models treat each slice as a standalone image and attempt to "complete" what they perceive as missing — adding heads, limbs, or objects that don't exist in the original art.

## The Solution

Instead of fighting the AI, WebtoonColorizer restructures the input so every chunk sent for colorization is self-contained:

1. **Stitch** — All input slices are joined into one continuous vertical strip.
2. **Smart Split** — The strip is scanned for natural panel boundaries (horizontal bands of pure black pixels). These are safe cut points where no content bleeds across.
3. **Colorize** — Each self-contained segment is sent to GPT-5.2 with the `image_generation` tool in **edit mode** with **high input fidelity**, preserving original line art and composition.
4. **Reassemble** — Colorized segments are stitched back together.
5. **Re-slice** — The reassembled strip is cut back to the original output dimensions.

### How Smart Splitting Works

The algorithm scans every horizontal row of the stitched image. A row is considered "safe" if nearly all pixels are dark (RGB values below a configurable threshold). It then looks for **runs** of 30+ consecutive safe rows — these correspond to the black divider bands between panels.

White text on black backgrounds (e.g., "WEEKS EARLIER...", "IT CAN'T END THIS WAY...") is automatically preserved because those text pixels are bright and fail the darkness check. The text stays inside its segment and is never split through.

Splits happen at the midpoint of each safe band, ensuring clean cuts with no content on either side.

### Aspect Ratio Handling

The image generation API only supports three output sizes: 1024x1024, 1024x1536, and 1536x1024. Since detected segments can have arbitrary dimensions, the pipeline:

1. Picks the closest API size based on the segment's aspect ratio
2. Scales the segment to fit within those dimensions (preserving aspect ratio)
3. Pads with black to fill the exact API size
4. After colorization, crops out the padding and resizes back to original dimensions

This ensures no content is cropped or distorted by the API.

### Blank Segment Detection

Segments that are 98%+ black pixels (divider bands, empty space) are automatically skipped — returned as-is without an API call. This saves API credits and prevents the AI from inventing content to fill empty space.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   ```

3. Place your black-and-white webtoon slices (PNG) in `./input/`.

## Usage

```bash
node colorizer.js
```

Colorized output is saved to `./output/` with the same filenames as the input.

## Configuration

All settings are optional. Add any of these to your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `INPUT_DIR` | `./input` | Directory containing input PNG slices |
| `OUTPUT_DIR` | `./output` | Directory for colorized output |
| `OUTPUT_WIDTH` | `800` | Output slice width in pixels |
| `OUTPUT_HEIGHT` | `1280` | Output slice height in pixels |
| `DARK_THRESHOLD` | `20` | Max RGB value (0-255) to consider a pixel "black" for split detection |
| `MIN_GAP_HEIGHT` | `30` | Minimum consecutive dark rows required for a valid split point |
| `EDGE_TOLERANCE` | `0.02` | Fraction of pixels per row allowed to be non-dark (handles compression artifacts) |
| `DEBUG` | `false` | Set to `true` to save intermediate images to `./debug/` |

### Character Colors

Character descriptions are defined in the `PROMPT` constant in `colorizer.js`. Edit these to match your webtoon's characters for consistent colorization across panels.

### Output Dimensions

Webtoon hosting platforms require specific slice dimensions. Common sizes:

| Platform | Width | Height | .env Setting |
|---|---|---|---|
| Standard | 800 | 1280 | `OUTPUT_WIDTH=800` `OUTPUT_HEIGHT=1280` |
| HD | 1600 | 2560 | `OUTPUT_WIDTH=1600` `OUTPUT_HEIGHT=2560` |

If input slices don't match the configured output dimensions, the final re-slice step resizes them automatically.

## Debug Mode

Set `DEBUG=true` to save intermediate images to `./debug/`:

**Bash / macOS / Linux:**
```bash
DEBUG=true node colorizer.js
```

**PowerShell (Windows):**
```powershell
$env:DEBUG="true"; node colorizer.js
```

**Or** set `DEBUG=true` in your `.env` file to always enable it.

Intermediate files saved:

- `01_stitched.png` — Full continuous strip
- `02_segment_NNN_input.png` — Each segment before colorization
- `03_segment_NNN_colorized.png` — Each segment after colorization
- `04_reassembled.png` — Full colorized strip before re-slicing

This is useful for tuning `DARK_THRESHOLD` and `MIN_GAP_HEIGHT` for your specific webtoon's art style.

## How It Works (Technical)

```
Input Slices (800x1280 each)
        |
        v
    Stitch vertically (800 x N*1280)
        |
        v
    Scan rows for dark bands (split detection)
        |
        v
    Split at midpoints of dark bands
        |
        v
    Skip blank segments (98%+ black)
        |
        v
    Pad each segment to API-compatible size (1024x1024, 1024x1536, or 1536x1024)
        |
        v
    Colorize via GPT-5.2 + image_generation tool (action: edit, input_fidelity: high)
        |
        v
    Crop padding, restore original dimensions
        |
        v
    Reassemble into full strip
        |
        v
    Re-slice to 800x1280 output
```

The colorization uses the OpenAI Responses API with GPT-5.2 as the orchestrating model and gpt-image-1.5 (via the `image_generation` tool) for image editing. The `action: "edit"` parameter ensures the original art is preserved — only color is added. The `input_fidelity: "high"` parameter preserves fine details like faces, line art, and composition.

## Cost Estimates

Cost is driven primarily by the image generation model (gpt-image-1.5), not the orchestrating text model. Blank segments (pure black dividers) are skipped and cost nothing.

| Scale | Slices | Est. API Calls | Est. Cost |
|---|---|---|---|
| Test (3 slices) | 3 | ~3 | ~$0.65 |
| Full chapter (100 slices) | 100 | ~60–80 | ~$13–$17 |

**How the estimate works:** 3 input slices produced 5 segments, 2 were blank (skipped), so 3 API calls cost $0.65 (~$0.22/call). A 100-slice chapter will have more panels but also more black dividers. Assuming ~60–80 non-blank segments at ~$0.22 each gives the range above.

Costs may vary based on segment size, API pricing changes, and how many blank segments your webtoon has.

## Limitations

- Requires clear black panel dividers for optimal splitting. Continuous scenes without any dark bands will be treated as a single segment.
- Segments are padded to fit API-supported aspect ratios (1:1, 2:3, 3:2). Very unusual aspect ratios will have more padding, but content is always preserved.
- Colorization quality depends on the AI model. Some artistic interpretation is inherent.
- API costs scale with the number of non-blank segments detected (blank segments are skipped).

## Project Structure

```
WebtoonColorizer/
    colorizer.js        Main script
    package.json        Dependencies (openai, sharp, dotenv)
    .env                API key and configuration
    input/              Input B&W slices (PNG)
    output/             Colorized output slices
    debug/              Intermediate images (when DEBUG=true)
```
