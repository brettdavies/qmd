// test/vram-padding-calibration.ts
import { getLlama, resolveModelFile, LlamaLogLevel } from "node-llama-cpp";

// --- CLI arg validation ---
const rawPercentage = process.argv[2] || "0.06";
const rawMaxReservedGB = process.argv[3] || "1";

const percentage = parseFloat(rawPercentage);
const maxReservedGB = parseFloat(rawMaxReservedGB);

if (!Number.isFinite(percentage) || percentage < 0 || percentage > 1) {
    console.error(`Invalid percentage: ${rawPercentage} (must be 0-1)`);
    process.exit(2);
}
if (!Number.isFinite(maxReservedGB) || maxReservedGB < 0 || maxReservedGB > 24) {
    console.error(`Invalid maxReservedGB: ${rawMaxReservedGB} (must be 0-24)`);
    process.exit(2);
}

const maxReserved = maxReservedGB * Math.pow(1024, 3);

// --- Environment metadata ---
console.log(`--- Environment ---`);
console.log(`node-llama-cpp: ${(await import("node-llama-cpp/package.json", { with: { type: "json" } })).default.version}`);
console.log(`bun: ${Bun.version}`);
console.log(`---`);
console.log(`Testing vramPadding: percentage=${percentage}, maxReserved=${maxReservedGB} GB`);

// --- 60-second timeout ---
const timeout = setTimeout(() => {
    console.error("TIMEOUT: Test exceeded 60 seconds");
    process.exit(3);
}, 60_000);

let llama: Awaited<ReturnType<typeof getLlama>> | undefined;

try {
    llama = await getLlama({
        logLevel: LlamaLogLevel.error,
        vramPadding(totalVram) {
            const padding = Math.floor(Math.min(totalVram * percentage, maxReserved));
            console.log(`  totalVram: ${(totalVram / Math.pow(1024, 3)).toFixed(2)} GB`);
            console.log(`  padding:   ${(padding / Math.pow(1024, 3)).toFixed(2)} GB`);
            return padding;
        }
    });

    // Use the SAME model URI as production (src/llm.ts:176)
    const modelPath = await resolveModelFile(
        "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
        `${process.env.HOME}/.cache/qmd/models`
    );

    console.log(`Loading model: ${modelPath}`);
    const model = await llama.loadModel({ modelPath });

    // Log GPU layer decision — critical for the maintainer
    console.log(`  gpuLayers: ${model.gpuLayers}`);
    console.log(`  modelSize: ${(model.size / Math.pow(1024, 2)).toFixed(1)} MiB`);

    const ctx = await model.createEmbeddingContext();
    const result = await ctx.getEmbeddingFor("test embedding");
    console.log(`Success! Embedding dim: ${result.vector.length}`);

    await ctx.dispose();
    await model.dispose();
} catch (err) {
    console.error("Error:", err);
    process.exit(1);
} finally {
    clearTimeout(timeout);
    if (llama) await llama.dispose();
}

process.exit(0);
