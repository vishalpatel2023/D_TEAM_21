import { useState } from "react";
import Header from "./components/Header";
import UploadSection from "./components/UploadSection";
import ResultsPanel from "./components/ResultsPanel";

export default function App() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  return (
    <div className="min-h-screen bg-gray-950">
      <Header />

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">

        <UploadSection
          onResult={setResult}
          onLoading={setLoading}
          onError={setError}
        />

        {/* Loading spinner */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <div className="w-14 h-14 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-400 text-sm">Detecting license plates…</p>
          </div>
        )}

        {/* Error banner */}
        {error && !loading && (
          <div className="bg-red-950/50 border border-red-500/40 rounded-xl px-5 py-4 text-red-300 text-sm">
            <span className="font-semibold">Error: </span>{error}
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <ResultsPanel result={result} />
        )}

      </main>
    </div>
  );
}
