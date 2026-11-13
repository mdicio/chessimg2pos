// Batch runner for fenshot (MIT), mirroring its own tools/benchmark/run.ts:
// board detection + tile model + orientation resolution.
// Reads JSON image paths (one per line) on stdin, writes one JSON FEN (or null) per line.
import { readFileSync } from "fs";
import { createInterface } from "readline";
import sharp from "sharp";
import * as ort from "onnxruntime-web";
import {
  recognizeGray, rgbaToGray, extractTiles, probsToPlacement, flipPlacement,
  resolveOrientation, type BoardCorners,
} from "@scoriiu/fenshot";

const MODEL = "packages/fenshot/model/chess-tiles-v2.onnx";
const MAX_DETECT_DIM = 1600;

async function loadGray(path: string) {
  const meta = await sharp(path, { animated: false }).metadata();
  const scale = Math.min(1, MAX_DETECT_DIM / Math.max(meta.width || 1, meta.height || 1));
  const w = Math.round((meta.width || 1) * scale);
  const h = Math.round((meta.height || 1) * scale);
  const rgba = await sharp(path, { animated: false }).resize(w, h).ensureAlpha().raw().toBuffer();
  return rgbaToGray(new Uint8ClampedArray(rgba), w, h);
}

async function main() {
  const session = await ort.InferenceSession.create(readFileSync(MODEL).buffer as ArrayBuffer, { executionProviders: ["wasm"] });
  const rl = createInterface({ input: process.stdin });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const path = JSON.parse(line);
    let fen: string | null = null;
    try {
      const gray = await loadGray(path);
      const classify = async (corners: BoardCorners) => {
        const out = await session.run({ tiles: new ort.Tensor("float32", extractTiles(gray, corners), [64, 1024]) });
        return probsToPlacement(out.probs.data as Float32Array);
      };
      const result = await recognizeGray(gray, classify);
      if (result) {
        fen = resolveOrientation(result.placement).orientation === "black"
          ? flipPlacement(result.placement) : result.placement;
      }
    } catch (e) {
      process.stderr.write(`ERR ${path} ${e}\n`);
    }
    process.stdout.write("@@FEN " + JSON.stringify(fen) + "\n");
  }
}
main();
