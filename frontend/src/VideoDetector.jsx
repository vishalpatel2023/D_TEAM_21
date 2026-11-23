import React, { useState, useEffect, useRef } from 'react';

// Video Backend URL (Port 8001)
// When on localhost, connects directly; when accessed via Cloudflare Tunnel/remote URL, uses Vite proxy
const VIDEO_BACKEND_URL = (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
  ? 'http://127.0.0.1:8001'
  : '';

export default function VideoDetector() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [selectedModule, setSelectedModule] = useState('all');
  const [confidence, setConfidence] = useState(0.35);
  const [frameSkip, setFrameSkip] = useState(1);

  // Connection & processing states
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [processingResult, setProcessingResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const fileInputRef = useRef(null);
  const outputVideoRef = useRef(null);
  const timerRef = useRef(null);

  // Poll video backend health on port 8001
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${VIDEO_BACKEND_URL}/video/health`);
        setIsBackendOnline(res.ok);
      } catch {
        setIsBackendOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Timer during active video processing
  useEffect(() => {
    if (isProcessing) {
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isProcessing]);

  // Handle video selection
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setVideoFile(file);
    setProcessingResult(null);
    setErrorMessage(null);

    // Create a local blob preview for the user to watch the uploaded video
    const preview = URL.createObjectURL(file);
    setVideoPreviewUrl(preview);
  };

  // Upload and process video
  const handleProcessVideo = async () => {
    if (!videoFile) {
      setErrorMessage('Please select a video file first.');
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setProcessingResult(null);

    const formData = new FormData();
    formData.append('file', videoFile);

    const queryParams = new URLSearchParams({
      module: selectedModule,
      conf: confidence.toString(),
      frame_skip: frameSkip.toString(),
    });

    try {
      const response = await fetch(`${VIDEO_BACKEND_URL}/video/process?${queryParams}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error status ${response.status}`);
      }

      const data = await response.json();
      setProcessingResult(data);
    } catch (err) {
      setErrorMessage(
        `Video analysis failed: ${err.message}. Make sure the video backend is running on port 8001.`
      );
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset form
  const handleReset = () => {
    setVideoFile(null);
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(null);
    setProcessingResult(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Seek video player to a specific timestamp
  const handleSeekTo = (seconds) => {
    if (outputVideoRef.current) {
      outputVideoRef.current.currentTime = seconds;
      outputVideoRef.current.play();
    }
  };

  const outputVideoUrl = processingResult?.download_url
    ? `${VIDEO_BACKEND_URL}${processingResult.download_url}`
    : null;

  return (
    <div style={{ marginTop: '10px' }}>
      {/* Top Video Backend Status Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: '#ffffff',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          padding: '12px 18px',
          marginBottom: '20px',
        }}
      >
        <div>
          <strong style={{ fontSize: '1rem', color: '#0f172a' }}>
            🎥 Video Clip Hazard & Violation Monitoring
          </strong>
          <p style={{ fontSize: '0.82rem', color: '#64748b', margin: '2px 0 0 0' }}>
            Upload an 8–12 second clip to run frame-by-frame deep learning models and view annotated output.
          </p>
        </div>

        <div className={`backend-status ${isBackendOnline ? 'online' : 'offline'}`}>
          <span className="status-dot"></span>
          {isBackendOnline ? 'Video Backend Online (:8001)' : 'Video Backend Offline (:8001)'}
        </div>
      </div>

      {/* Main Grid: Controls on Left, Video Display on Right */}
      <div className="main-grid">
        {/* Left Card: Upload & Video Settings */}
        <div className="card">
          <div className="card-title">
            <span>Video Settings</span>
            {videoFile && (
              <button className="btn btn-secondary" onClick={handleReset} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
                Clear
              </button>
            )}
          </div>

          {/* Upload Area */}
          <div
            className="drop-zone"
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '24px 16px',
              textAlign: 'center',
              cursor: 'pointer',
              border: '2px dashed #cbd5e1',
              borderRadius: '8px',
              background: '#f8fafc',
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
              style={{ display: 'none' }}
            />
            <div style={{ fontSize: '2rem', marginBottom: '6px' }}>🎬</div>
            <div style={{ fontWeight: 600, color: '#1e293b', fontSize: '0.95rem' }}>
              {videoFile ? videoFile.name : 'Choose or drag a video clip here'}
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '4px' }}>
              Supports MP4, MOV, AVI, WEBM (Recommended length: 8–12s)
            </p>
          </div>

          {/* Video Preview If Uploaded */}
          {videoPreviewUrl && !processingResult && (
            <div style={{ marginTop: '16px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '6px' }}>
                Original Video Preview:
              </label>
              <video
                src={videoPreviewUrl}
                controls
                style={{ width: '100%', maxHeight: '200px', borderRadius: '6px', background: '#000000' }}
              />
            </div>
          )}

          {/* Module Selector */}
          <div style={{ marginTop: '18px' }}>
            <label style={{ fontSize: '0.88rem', fontWeight: 600, color: '#334155', display: 'block', marginBottom: '6px' }}>
              Select Detection Target:
            </label>
            <select
              value={selectedModule}
              onChange={(e) => setSelectedModule(e.target.value)}
              disabled={isProcessing}
              style={{
                width: '100%',
                padding: '9px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                background: '#ffffff',
                fontSize: '0.9rem',
                color: '#1e293b',
              }}
            >
              <option value="all">🔍 All Models Combined (Helmet, Pothole, Plate, Triple Riding)</option>
              <option value="helmet">🛡️ Helmet Detection Only</option>
              <option value="pothole">🕳️ Pothole Detection Only</option>
              <option value="anpr">🔢 License Plate (ANPR) Only</option>
              <option value="triple_riding">👥 Triple Riding Only</option>
            </select>
          </div>

          {/* Frame Skip / Speed Optimization */}
          <div style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <label style={{ fontSize: '0.88rem', fontWeight: 600, color: '#334155' }}>
                Processing Speed:
              </label>
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: frameSkip === 2 ? '#15803d' : '#475569' }}>
                {frameSkip === 2 ? '⚡ Fast (2x Speedup)' : '🎯 Standard (All Frames)'}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <button
                type="button"
                onClick={() => setFrameSkip(1)}
                className={`btn ${frameSkip === 1 ? 'btn-primary' : 'btn-secondary'}`}
                style={{ fontSize: '0.82rem', padding: '6px' }}
              >
                Every Frame (1x)
              </button>
              <button
                type="button"
                onClick={() => setFrameSkip(2)}
                className={`btn ${frameSkip === 2 ? 'btn-primary' : 'btn-secondary'}`}
                style={{ fontSize: '0.82rem', padding: '6px' }}
              >
                Skip 2x (Recommended)
              </button>
            </div>
          </div>

          {/* Confidence Slider */}
          <div style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <label style={{ fontSize: '0.88rem', fontWeight: 600, color: '#334155' }}>
                Confidence Threshold:
              </label>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#0284c7' }}>
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.15"
              max="0.85"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              disabled={isProcessing}
              style={{ width: '100%', accentColor: '#0284c7' }}
            />
          </div>

          {/* Action Button */}
          <button
            className="btn btn-primary"
            onClick={handleProcessVideo}
            disabled={!videoFile || isProcessing || !isBackendOnline}
            style={{ width: '100%', marginTop: '20px', padding: '12px', fontSize: '0.95rem' }}
          >
            {isProcessing ? `Processing Video... (${elapsedSeconds}s)` : '🚀 Run Video Analysis'}
          </button>

          {/* Backend offline warning */}
          {!isBackendOnline && (
            <div style={{ marginTop: '12px', padding: '8px 12px', background: '#fee2e2', border: '1px solid #fecaca', borderRadius: '6px', fontSize: '0.8rem', color: '#991b1b' }}>
              Start video backend: <code>uvicorn video_extension.api:app --reload --port 8001</code>
            </div>
          )}

          {/* Error Message */}
          {errorMessage && (
            <div style={{ marginTop: '12px', padding: '10px', background: '#fef2f2', border: '1px solid #f87171', borderRadius: '6px', color: '#dc2626', fontSize: '0.85rem' }}>
              {errorMessage}
            </div>
          )}
        </div>

        {/* Right Card: Output Video Player & Analytics */}
        <div className="card">
          <div className="card-title">
            <span>Video Output & Detection HUD</span>
            {outputVideoUrl && (
              <a
                href={outputVideoUrl}
                download
                className="btn btn-secondary"
                style={{ textDecoration: 'none', padding: '4px 10px', fontSize: '0.8rem' }}
              >
                💾 Download Video
              </a>
            )}
          </div>

          {/* Loading Indicator */}
          {isProcessing && (
            <div style={{ textAlign: 'center', padding: '40px 20px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: '2.4rem', animation: 'spin 2s linear infinite' }}>⏳</div>
              <h3 style={{ marginTop: '12px', fontSize: '1.1rem', color: '#0f172a' }}>
                Analyzing Video Frames with YOLOv8...
              </h3>
              <p style={{ color: '#64748b', fontSize: '0.88rem', marginTop: '6px' }}>
                Elapsed Time: <strong>{elapsedSeconds} seconds</strong>
              </p>
              <p style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: '4px' }}>
                Each frame is processed in memory and annotated with live alerts.
              </p>
            </div>
          )}

          {/* Output Video Player */}
          {!isProcessing && outputVideoUrl && (
            <div>
              <video
                key={outputVideoUrl}
                ref={outputVideoRef}
                src={outputVideoUrl}
                controls
                playsInline
                preload="auto"
                style={{
                  width: '100%',
                  borderRadius: '8px',
                  background: '#000000',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                }}
              />

              {/* Status Banner */}
              <div
                style={{
                  marginTop: '14px',
                  padding: '10px 14px',
                  borderRadius: '6px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: processingResult?.analytics?.has_violations ? '#fef2f2' : '#f0fdf4',
                  border: `1px solid ${processingResult?.analytics?.has_violations ? '#fecaca' : '#bbf7d0'}`,
                }}
              >
                <div>
                  <strong style={{ color: processingResult?.analytics?.has_violations ? '#991b1b' : '#166534', fontSize: '0.9rem' }}>
                    {processingResult?.analytics?.has_violations
                      ? '⚠️ Safety Violations Detected in Video!'
                      : '✅ No Severe Violations Detected'}
                  </strong>
                  <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '2px' }}>
                    Processed {processingResult?.video_metadata?.processed_frames} frames in{' '}
                    {processingResult?.performance?.total_processing_time_sec}s ({processingResult?.performance?.processing_fps} FPS)
                  </div>
                </div>
              </div>

              {/* Analytics Metric Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginTop: '14px' }}>
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Helmet Violations</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#dc2626' }}>
                    {processingResult?.analytics?.helmet_violations_detected || 0}
                  </div>
                </div>

                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Potholes</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#d97706' }}>
                    {processingResult?.analytics?.potholes_detected || 0}
                  </div>
                </div>

                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Plates Located</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0284c7' }}>
                    {processingResult?.analytics?.license_plates_detected || 0}
                  </div>
                </div>

                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '10px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Triple Riding</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#dc2626' }}>
                    {processingResult?.analytics?.triple_riding_violations_detected || 0}
                  </div>
                </div>
              </div>

              {/* Timeline of Incidents */}
              {processingResult?.timeline_events && processingResult.timeline_events.length > 0 && (
                <div style={{ marginTop: '18px' }}>
                  <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#334155', marginBottom: '8px' }}>
                    Incident Timeline (Click timestamp to jump to second):
                  </div>
                  <div style={{ maxHeight: '180px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '6px' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                      <thead>
                        <tr style={{ background: '#f1f5f9', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>
                          <th style={{ padding: '6px 10px' }}>Timestamp</th>
                          <th style={{ padding: '6px 10px' }}>Frame</th>
                          <th style={{ padding: '6px 10px' }}>Detected Hazard / Violation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {processingResult.timeline_events.map((event, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '6px 10px' }}>
                              <button
                                type="button"
                                onClick={() => handleSeekTo(event.timestamp_sec)}
                                style={{
                                  background: '#e0f2fe',
                                  color: '#0369a1',
                                  border: 'none',
                                  borderRadius: '4px',
                                  padding: '2px 8px',
                                  fontSize: '0.78rem',
                                  cursor: 'pointer',
                                  fontWeight: 600,
                                }}
                              >
                                ⏱️ {event.timestamp_sec}s
                              </button>
                            </td>
                            <td style={{ padding: '6px 10px', color: '#64748b' }}>#{event.frame}</td>
                            <td style={{ padding: '6px 10px' }}>
                              {event.violations.map((v, i) => (
                                <span
                                  key={`v-${i}`}
                                  style={{
                                    display: 'inline-block',
                                    background: '#fee2e2',
                                    color: '#b91c1c',
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    fontSize: '0.75rem',
                                    fontWeight: 600,
                                    marginRight: '4px',
                                  }}
                                >
                                  {v}
                                </span>
                              ))}
                              {event.hazards.map((h, i) => (
                                <span
                                  key={`h-${i}`}
                                  style={{
                                    display: 'inline-block',
                                    background: '#fef3c7',
                                    color: '#b45309',
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    fontSize: '0.75rem',
                                    fontWeight: 600,
                                    marginRight: '4px',
                                  }}
                                >
                                  {h}
                                </span>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Empty State before upload */}
          {!isProcessing && !outputVideoUrl && (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#94a3b8' }}>
              <div style={{ fontSize: '3rem', marginBottom: '8px' }}>📹</div>
              <div style={{ fontWeight: 600, fontSize: '1rem', color: '#64748b' }}>
                No Video Analyzed Yet
              </div>
              <p style={{ fontSize: '0.82rem', marginTop: '4px' }}>
                Upload an 8–12s road traffic video on the left and click "Run Video Analysis".
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
