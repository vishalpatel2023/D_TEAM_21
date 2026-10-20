import { useRef, useState } from "react";
import axios from "axios";

export default function UploadSection({ onResult, onLoading, onError }) {
  const fileInputRef          = useRef(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [conf, setConf]       = useState(0.4);

  // ── File handling ─────────────────────────────────────────────────────────
  const handleFile = (file) => {
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    onError(null);
    onResult(null);
  };

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      // Sync the hidden input so submit picks it up
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInputRef.current.files = dt.files;
      handleFile(file);
    }
  };

  // ── Submit ────────────────────────────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    const file = fileInputRef.current?.files[0];
    if (!file) {
      onError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("conf", conf.toString());

    onLoading(true);
    onError(null);
    onResult(null);

    try {
      const res = await axios.post("/api/anpr/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60_000,
      });
      onResult(res.data);
    } catch (err) {
      const msg =
        err.response?.data?.error ??
        err.message ??
        "Failed to process image.";
      onError(msg);
    } finally {
      onLoading(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <section className="bg-gray-900 rounded-2xl border border-gray-800 p-6 space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-white">Upload Vehicle Image</h2>
        <p className="text-sm text-gray-400 mt-1">
          Supports JPEG, PNG, WebP and BMP. Max 16 MB.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">

        {/* ── Drop zone ──────────────────────────────────────────────────── */}
        <div
          onClick={() => fileInputRef.current.click()}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          className={[
            "relative border-2 border-dashed rounded-xl cursor-pointer transition-all duration-150",
            dragOver
              ? "border-blue-400 bg-blue-950/30"
              : "border-gray-700 hover:border-gray-500 bg-gray-800/30",
          ].join(" ")}
        >
          {preview ? (
            <img
              src={preview}
              alt="Selected preview"
              className="w-full max-h-72 object-contain rounded-xl p-2"
            />
          ) : (
            <div className="flex flex-col items-center justify-center py-16 gap-3 select-none">
              <span className="text-5xl">📷</span>
              <p className="text-gray-400 text-sm font-medium">
                Drag &amp; drop an image here, or click to browse
              </p>
              <p className="text-gray-600 text-xs">JPEG · PNG · WebP · BMP</p>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/bmp"
            className="hidden"
            onChange={handleInputChange}
          />
        </div>

        {/* ── Confidence slider ───────────────────────────────────────────── */}
        <div className="flex items-center gap-4">
          <label className="text-sm text-gray-400 shrink-0 w-52">
            Detection Confidence:&nbsp;
            <span className="text-blue-400 font-bold">
              {(conf * 100).toFixed(0)}%
            </span>
          </label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            className="flex-1 accent-blue-500"
          />
        </div>

        {/* ── Submit ──────────────────────────────────────────────────────── */}
        <button
          type="submit"
          className="w-full py-3 bg-blue-600 hover:bg-blue-700 active:scale-[0.98]
            text-white font-semibold rounded-xl transition-all duration-150 text-sm
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            focus:ring-offset-gray-900"
        >
          🔍&ensp;Detect Number Plates
        </button>
      </form>
    </section>
  );
}
