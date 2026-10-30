import React, { useState, useEffect, useRef, useCallback } from 'react';

// FastAPI Backend URL
const BACKEND_URL = 'http://127.0.0.1:8000';

export default function App() {
  // Navigation tabs: 'helmet', 'pothole', 'anpr', 'triple_riding', or 'all'
  const [activeTab, setActiveTab] = useState('helmet');

  // Backend connection status
  const [isBackendOnline, setIsBackendOnline] = useState(false);

  // File and preview states
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  // Detection and results states
  const [isScanning, setIsScanning] = useState(false);
  const [results, setResults] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [confidence, setConfidence] = useState(0.35);
  const [showBoxes, setShowBoxes] = useState(true);

  // Project statistics counter
  const [stats, setStats] = useState({
    scans: 0,
    helmetViolations: 0,
    potholes: 0,
    plates: 0,
    tripleRidingViolations: 0,
  });

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const originalImageObj = useRef(null);

  // Check if FastAPI backend is reachable on port 8000
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`);
        setIsBackendOnline(response.ok);
      } catch {
        setIsBackendOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Helper to style each bounding box
  const getBoxStyle = useCallback((detection) => {
    const className = (detection.class || '').toLowerCase();

    // 1. Helmet module
    if (className === 'without helmet' || (activeTab === 'helmet' && detection.violation)) {
      return {
        color: '#dc2626', // Red
        label: `Without Helmet (${(detection.confidence * 100).toFixed(0)}%)`,
      };
    }
    if (className === 'with helmet') {
      return {
        color: '#16a34a', // Green
        label: `With Helmet (${(detection.confidence * 100).toFixed(0)}%)`,
      };
    }

    // 2. Pothole module
    if (className === 'pothole' || detection.hazard) {
      return {
        color: '#d97706', // Orange / Amber
        label: `Pothole (${(detection.confidence * 100).toFixed(0)}%)`,
      };
    }

    // 3. ANPR module
    if (className === 'license_plate' || className === 'plate') {
      const plateText = detection.plate_text ? `Plate: ${detection.plate_text}` : 'License Plate';
      return {
        color: '#0284c7', // Sky Blue
        label: `${plateText} (${(detection.confidence * 100).toFixed(0)}%)`,
      };
    }

    // 4. Triple Riding module
    if (className === 'triple riding' || (detection.violation && className.includes('riding'))) {
      const riders = detection.rider_count ? `${detection.rider_count} Riders` : 'Violation';
      return {
        color: '#dc2626', // Red Alert
        label: `⚠️ Triple Riding: ${riders} (${(detection.confidence * 100).toFixed(0)}%)`,
      };
    }
    if (className === 'motorcycle') {
      return {
        color: '#16a34a', // Green
        label: `Motorcycle (${(detection.confidence * 100).toFixed(0)}%)`,
      };
    }

    return {
      color: '#4f46e5',
      label: `${detection.class} (${(detection.confidence * 100).toFixed(0)}%)`,
    };
  }, [activeTab]);

  // Extract flat list of detections
  const getDetectionsList = useCallback(() => {
    if (!results) return [];
    if (activeTab === 'all') {
      return [
        ...(results.helmet?.detections || []),
        ...(results.pothole?.detections || []),
        ...(results.anpr?.detections || []),
        ...(results.triple_riding?.detections || []),
      ];
    }
    return results.detections || [];
  }, [results, activeTab]);

  // Draw image and bounding boxes onto HTML5 canvas
  const drawOnCanvas = useCallback((img, detections = []) => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;

    // 1. Draw base image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // 2. Draw boxes if enabled
    if (!showBoxes || !detections || detections.length === 0) return;

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const width = x2 - x1;
      const height = y2 - y1;
      const style = getBoxStyle(det);

      // Box outline
      ctx.lineWidth = Math.max(3, Math.round(canvas.width / 300));
      ctx.strokeStyle = style.color;
      ctx.strokeRect(x1, y1, width, height);

      // Label background
      const fontSize = Math.max(14, Math.round(canvas.width / 45));
      ctx.font = `bold ${fontSize}px sans-serif`;

      const textWidth = ctx.measureText(style.label).width;
      const padding = 6;
      const labelW = textWidth + padding * 2;
      const labelH = fontSize + padding * 1.5;
      const labelY = Math.max(0, y1 - labelH);

      ctx.fillStyle = style.color;
      ctx.fillRect(x1, labelY, labelW, labelH);

      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(style.label, x1 + padding, labelY + labelH - padding);
    });
  }, [showBoxes, getBoxStyle]);

  // Redraw when toggle box checkbox or detections change
  useEffect(() => {
    if (originalImageObj.current) {
      drawOnCanvas(originalImageObj.current, getDetectionsList());
    }
  }, [drawOnCanvas, getDetectionsList]);

  // When a user selects or drops an image
  const handleFileSelect = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResults(null);
    setErrorMessage(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target.result;
      setImagePreview(url);

      const img = new Image();
      img.onload = () => {
        originalImageObj.current = img;
        drawOnCanvas(img, []);
      };
      img.src = url;
    };
    reader.readAsDataURL(file);
  };

  // Send request to FastAPI backend
  const runDetection = async () => {
    if (!selectedFile) {
      setErrorMessage('Please select or upload an image first.');
      return;
    }

    setIsScanning(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Determine target endpoint
    let endpoint = `${BACKEND_URL}/detect/helmet?conf=${confidence}`;
    if (activeTab === 'pothole') {
      endpoint = `${BACKEND_URL}/detect/pothole?conf=${confidence}`;
    } else if (activeTab === 'anpr') {
      endpoint = `${BACKEND_URL}/detect/anpr?conf=${confidence}`;
    } else if (activeTab === 'triple_riding') {
      endpoint = `${BACKEND_URL}/detect/triple-riding?conf=${confidence}`;
    } else if (activeTab === 'all') {
      endpoint = `${BACKEND_URL}/detect/all`;
    }

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${res.status}`);
      }

      const data = await res.json();
      setResults(data);

      // Extract detections for canvas drawing
      let currentList = [];
      if (activeTab === 'all') {
        currentList = [
          ...(data.helmet?.detections || []),
          ...(data.pothole?.detections || []),
          ...(data.anpr?.detections || []),
          ...(data.triple_riding?.detections || []),
        ];
      } else {
        currentList = data.detections || [];
      }

      // Update counters
      setStats((prev) => ({
        scans: prev.scans + 1,
        helmetViolations: prev.helmetViolations + (data.violations_count && activeTab === 'helmet' ? data.violations_count : data.helmet?.violations_count || 0),
        potholes: prev.potholes + (data.hazards_count || data.pothole?.hazards_count || 0),
        plates: prev.plates + (data.plates_count || data.anpr?.plates_count || 0),
        tripleRidingViolations: prev.tripleRidingViolations + (data.violations_count && activeTab === 'triple_riding' ? data.violations_count : data.triple_riding?.violations_count || 0),
      }));

      // Render boxes on canvas
      if (originalImageObj.current) {
        drawOnCanvas(originalImageObj.current, currentList);
      }
    } catch (err) {
      setErrorMessage(
        `Failed to run detection: ${err.message}. Ensure FastAPI server is running on port 8000.`
      );
    } finally {
      setIsScanning(false);
    }
  };

  // Download annotated canvas image
  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `detected_${activeTab}_result.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  // Reset form
  const handleReset = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setResults(null);
    setErrorMessage(null);
  };

  const detections = getDetectionsList();

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="header-title">
          <h1>Road Safety Monitoring System</h1>
          <p>YOLOv8 Computer Vision & Deep Learning Final Project</p>
        </div>

        <div className={`backend-status ${isBackendOnline ? 'online' : 'offline'}`}>
          <span className="status-dot"></span>
          {isBackendOnline ? 'FastAPI Backend Online (:8000)' : 'Backend Offline (:8000)'}
        </div>
      </header>

      {/* Module Tabs */}
      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'helmet' ? 'active' : ''}`}
          onClick={() => { setActiveTab('helmet'); setResults(null); }}
        >
          <span>🛡️ Helmet Detection</span>
          <span className="tab-tag">Vishal</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'pothole' ? 'active' : ''}`}
          onClick={() => { setActiveTab('pothole'); setResults(null); }}
        >
          <span>🕳️ Pothole Detection</span>
          <span className="tab-tag">Virajsingh</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'anpr' ? 'active' : ''}`}
          onClick={() => { setActiveTab('anpr'); setResults(null); }}
        >
          <span>🔢 License Plate (ANPR)</span>
          <span className="tab-tag">Shibchandan</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'triple_riding' ? 'active' : ''}`}
          onClick={() => { setActiveTab('triple_riding'); setResults(null); }}
        >
          <span>👥 Triple Riding</span>
          <span className="tab-tag">Sojwal</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => { setActiveTab('all'); setResults(null); }}
        >
          <span>🔍 All Models (Combined)</span>
          <span className="tab-tag">Combined</span>
        </button>
      </div>

      {/* Project Statistics Row */}
      <div className="stats-row">
        <div className="stat-box">
          <div className="stat-icon">📸</div>
          <div className="stat-info">
            <div className="stat-label">Total Scans</div>
            <div className="stat-value">{stats.scans}</div>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon">⚠️</div>
          <div className="stat-info">
            <div className="stat-label">Helmet Violations</div>
            <div className="stat-value">{stats.helmetViolations}</div>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon">🕳️</div>
          <div className="stat-info">
            <div className="stat-label">Potholes Detected</div>
            <div className="stat-value">{stats.potholes}</div>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon">🔢</div>
          <div className="stat-info">
            <div className="stat-label">Plates Recognized</div>
            <div className="stat-value">{stats.plates}</div>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon">👥</div>
          <div className="stat-info">
            <div className="stat-label">Triple Riding</div>
            <div className="stat-value">{stats.tripleRidingViolations}</div>
          </div>
        </div>
      </div>

      {/* Main Grid: Left Controls, Right Output */}
      <div className="main-grid">
        {/* Left Card: Upload & Detection Settings */}
        <div className="card">
          <div className="card-title">
            <span>Upload Image</span>
            {selectedFile && (
              <button className="btn btn-secondary" onClick={handleReset}>
                Clear
              </button>
            )}
          </div>

          {/* Upload Drop Area */}
          <div
            className="upload-box"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const dropped = e.dataTransfer.files?.[0];
              if (dropped) handleFileSelect(dropped);
            }}
          >
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              {selectedFile ? selectedFile.name : 'Choose an image or drag here'}
            </div>
            <div className="upload-subtext">Supports JPG, JPEG, PNG</div>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept="image/*"
              onChange={(e) => handleFileSelect(e.target.files?.[0])}
            />
          </div>

          {/* Confidence Slider */}
          <div className="form-group">
            <label>
              <span>Confidence Threshold</span>
              <span>{(confidence * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0.15"
              max="0.95"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
            />
          </div>

          {/* Action Button */}
          <button
            className="btn btn-primary"
            onClick={runDetection}
            disabled={!selectedFile || isScanning}
          >
            {isScanning ? 'Running YOLO Model...' : `Run ${activeTab.toUpperCase().replace('_', ' ')} Detection`}
          </button>

          {/* Offline Warning Notice */}
          {!isBackendOnline && (
            <div className="alert alert-warning">
              <strong>Backend Offline:</strong> Start FastAPI by running:
              <br />
              <code>uvicorn backend.api:app --reload --port 8000</code>
            </div>
          )}

          {/* Error Message */}
          {errorMessage && (
            <div className="alert alert-danger">
              {errorMessage}
            </div>
          )}
        </div>

        {/* Right Card: Canvas Preview & Detections Table */}
        <div className="card">
          <div className="card-title">
            <span>Detection Output</span>
            {imagePreview && (
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setShowBoxes(!showBoxes)}
                >
                  {showBoxes ? 'Hide Bounding Boxes' : 'Show Bounding Boxes'}
                </button>
                <button className="btn btn-secondary" onClick={handleDownload}>
                  Save Image
                </button>
              </div>
            )}
          </div>

          {/* Canvas or Empty State */}
          <div className="canvas-wrapper">
            {imagePreview ? (
              <canvas ref={canvasRef} className="detection-canvas" />
            ) : (
              <div className="canvas-placeholder">
                <span className="canvas-placeholder-icon">🚦</span>
                <p>Select a traffic photo on the left to view detections.</p>
              </div>
            )}
          </div>

          {/* Result Banner */}
          {results && (
            <div
              className={`result-banner ${
                results.has_violation || results.has_hazard || results.summary?.has_safety_alert
                  ? 'violation'
                  : 'safe'
              }`}
            >
              <div className="banner-icon">
                {results.has_violation || results.has_hazard || results.summary?.has_safety_alert
                  ? '⚠️'
                  : '✅'}
              </div>
              <div>
                <div className="banner-text">
                  {/* Helmet tab */}
                  {activeTab === 'helmet' && (
                    results.has_violation
                      ? `Violation: ${results.violations_count} rider(s) without helmet detected!`
                      : 'Safe: All detected riders are wearing helmets.'
                  )}

                  {/* Pothole tab */}
                  {activeTab === 'pothole' && (
                    results.has_hazard
                      ? `Road Hazard: ${results.hazards_count} pothole(s) detected!`
                      : 'Safe: No road potholes detected in this frame.'
                  )}

                  {/* ANPR tab */}
                  {activeTab === 'anpr' && (
                    results.total_detections > 0
                      ? `Plates Detected: ${results.total_detections} license plate(s) found.`
                      : 'No license plates detected.'
                  )}

                  {/* Triple Riding tab */}
                  {activeTab === 'triple_riding' && (
                    results.has_violation
                      ? `Violation: ${results.violations_count} overloaded motorcycle(s) (triple riding) detected!`
                      : 'Safe: No triple riding detected. Two-wheelers have compliant passenger counts.'
                  )}

                  {/* Combined tab */}
                  {activeTab === 'all' && (
                    `Unified Scan: ${results.summary?.helmet_violations || 0} Helmet | ${results.summary?.potholes_found || 0} Potholes | ${results.summary?.plates_found || 0} Plates | ${results.summary?.triple_riding_violations || 0} Triple Riding`
                  )}
                </div>
                <div className="banner-meta">
                  Processing Time: {results.time_taken_ms || 0} ms | Total Detections: {detections.length}
                </div>
              </div>
            </div>
          )}

          {/* Results Table */}
          {results && detections.length > 0 && (
            <div className="table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Class</th>
                    <th>Status / Info</th>
                    <th>Confidence</th>
                    <th>Bounding Box</th>
                  </tr>
                </thead>
                <tbody>
                  {detections.map((det, index) => (
                    <tr
                      key={index}
                      className={det.violation ? 'row-violation' : ''}
                    >
                      <td>{index + 1}</td>
                      <td>
                        <strong>{det.class}</strong>
                      </td>
                      <td>
                        {det.class === 'Triple Riding' && (
                          <span className="pill pill-danger">
                            ⚠️ Overloaded ({det.rider_count || 3} Riders)
                          </span>
                        )}
                        {det.class === 'Motorcycle' && (
                          <span className="pill pill-safe">
                            ✓ Compliant ({det.rider_count || 1} Rider{det.rider_count === 1 ? '' : 's'})
                          </span>
                        )}
                        {det.violation && det.class !== 'Triple Riding' && (
                          <span className="pill pill-danger">No Helmet</span>
                        )}
                        {det.class === 'With Helmet' && (
                          <span className="pill pill-safe">Helmet Worn</span>
                        )}
                        {det.hazard && (
                          <span className="pill pill-warning">Road Pothole</span>
                        )}
                        {det.plate_text && (
                          <span className="pill pill-info">
                            Plate: <strong>{det.plate_text}</strong>
                          </span>
                        )}
                        {!det.violation &&
                          !det.hazard &&
                          !det.plate_text &&
                          det.class !== 'With Helmet' &&
                          det.class !== 'Motorcycle' &&
                          det.class !== 'Triple Riding' && (
                            <span className="pill pill-safe">Detected</span>
                          )}
                      </td>
                      <td>{(det.confidence * 100).toFixed(1)}%</td>
                      <td className="mono-bbox">[{det.bbox.join(', ')}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>Road Safety Monitoring System • Group Academic Project</p>
        <div className="team-list">
          <div className="team-item">Vishal Patel (20233314) - Helmet</div>
          <div className="team-item">Virajsingh Rajput (20233360) - Pothole</div>
          <div className="team-item">Shibchandan Mistry (20233263) - ANPR</div>
          <div className="team-item">Sojwal Vilhekar (20233359) - Triple Riding</div>
        </div>
      </footer>
    </div>
  );
}
