export default function Header() {
  return (
    <header className="bg-gray-900 border-b border-gray-800 shadow-lg sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-4 py-4 flex items-center gap-3">
        <span className="text-3xl select-none">🚗</span>
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight leading-tight">
            Road Safety Monitoring System
          </h1>
          <p className="text-xs text-blue-400 font-semibold tracking-widest uppercase">
            Automatic Number Plate Recognition · ANPR
          </p>
        </div>
      </div>
    </header>
  );
}
