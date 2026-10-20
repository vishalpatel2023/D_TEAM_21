/**
 * PlateCard
 * ---------
 * Displays detection details for a single license plate:
 *   - Extracted plate text (monospace, large)
 *   - Detection confidence bar (blue)
 *   - OCR confidence bar (purple)
 *   - Bounding box coordinates
 */
export default function PlateCard({ detection, index }) {
  const { plate_text, confidence, ocr_confidence, bbox, raw_ocr_text } = detection;
  const [x1, y1, x2, y2] = bbox;

  const hasText = Boolean(plate_text);

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-5 space-y-3">

      {/* ── Header row ───────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          Plate #{index + 1}
        </span>
        {hasText ? (
          <span className="px-3 py-0.5 bg-green-900/40 text-green-400
            border border-green-700/40 text-xs font-semibold rounded-full">
            ✓ Text Extracted
          </span>
        ) : (
          <span className="px-3 py-0.5 bg-yellow-900/40 text-yellow-400
            border border-yellow-700/40 text-xs font-semibold rounded-full">
            ⚠ No Text
          </span>
        )}
      </div>

      {/* ── Plate text display ───────────────────────────────────────────── */}
      <div className="bg-gray-950 rounded-lg px-4 py-4 text-center border border-gray-700/50">
        <p className={[
          "font-mono font-bold tracking-widest",
          hasText ? "text-2xl text-white" : "text-lg text-gray-600",
        ].join(" ")}>
          {hasText ? plate_text : "— — —"}
        </p>
        {raw_ocr_text && raw_ocr_text.trim() !== plate_text && (
          <p className="text-xs text-gray-600 mt-1.5">
            raw: <span className="font-mono">{raw_ocr_text}</span>
          </p>
        )}
      </div>

      {/* ── Confidence bars ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">

        {/* Detection confidence */}
        <div className="bg-gray-900/60 rounded-lg p-3 space-y-1.5">
          <p className="text-xs text-gray-500">Detection Confidence</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-700 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all"
                style={{ width: `${(confidence * 100).toFixed(0)}%` }}
              />
            </div>
            <span className="text-xs font-bold text-blue-400 w-9 text-right">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* OCR confidence */}
        <div className="bg-gray-900/60 rounded-lg p-3 space-y-1.5">
          <p className="text-xs text-gray-500">OCR Confidence</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-700 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-purple-500 h-1.5 rounded-full transition-all"
                style={{ width: `${(ocr_confidence * 100).toFixed(0)}%` }}
              />
            </div>
            <span className="text-xs font-bold text-purple-400 w-9 text-right">
              {(ocr_confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

      </div>

      {/* ── Bounding box coords ──────────────────────────────────────────── */}
      <p className="text-xs text-gray-600 font-mono">
        bbox: [{x1}, {y1}, {x2}, {y2}]
      </p>

    </div>
  );
}
