import PlateCard from "./PlateCard";

/**
 * ResultsPanel
 * ------------
 * Shown after a successful detection run. Displays:
 *   - Summary bar (plate count + download button)
 *   - Annotated output image (base64 JPEG from Python)
 *   - Per-plate PlateCard list
 */
export default function ResultsPanel({ result }) {
  const { detections = [], annotated_image_b64, total_plates } = result;

  const withText = detections.filter((d) => d.plate_text).length;

  // ── Download annotated image ──────────────────────────────────────────────
  const handleDownload = () => {
    const link    = document.createElement("a");
    link.href     = `data:image/jpeg;base64,${annotated_image_b64}`;
    link.download = "anpr_result.jpg";
    link.click();
  };

  return (
    <section className="space-y-6">

      {/* ── Summary bar ─────────────────────────────────────────────────── */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 px-5 py-4
        flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <span className="text-3xl select-none">🔖</span>
          <div>
            <p className="text-lg font-bold text-white">
              {total_plates} Plate{total_plates !== 1 ? "s" : ""} Detected
            </p>
            <p className="text-sm text-gray-400">
              {withText} of {total_plates} with text extracted
            </p>
          </div>
        </div>
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700
            text-gray-300 text-sm font-medium rounded-xl border border-gray-700
            transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          ⬇&ensp;Download Annotated Image
        </button>
      </div>

      {/* ── Main content grid ────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">

        {/* Annotated image */}
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-4 space-y-2">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
            Annotated Output
          </h3>
          <img
            src={`data:image/jpeg;base64,${annotated_image_b64}`}
            alt="Annotated detection result"
            className="w-full rounded-xl object-contain"
          />
        </div>

        {/* Plate cards */}
        <div className="space-y-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
            Detection Results
          </h3>

          {total_plates === 0 ? (
            <div className="bg-gray-900 rounded-2xl border border-gray-800 py-16 px-6 text-center">
              <p className="text-4xl mb-3">🔍</p>
              <p className="text-gray-400 font-medium">No license plates detected.</p>
              <p className="text-gray-600 text-sm mt-1">
                Try lowering the confidence threshold or uploading a clearer image.
              </p>
            </div>
          ) : (
            detections.map((det, i) => (
              <PlateCard key={i} detection={det} index={i} />
            ))
          )}
        </div>

      </div>
    </section>
  );
}
