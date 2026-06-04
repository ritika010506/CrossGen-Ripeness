import { useState, useCallback } from "react";

const API_URL = "http://127.0.0.1:8000";  // replace with your Render URL

const LABELS = {
  ripe:     { emoji: "✅", color: "#1D9E75", message: "This fruit is ripe and ready to eat." },
  unripe:   { emoji: "🟡", color: "#F59E0B", message: "This fruit needs more time to ripen." },
  overripe: { emoji: "🔴", color: "#E24B4A", message: "This fruit is overripe. Use it soon." },
};

export default function App() {
  const [image, setImage]       = useState(null);
  const [preview, setPreview]   = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [waking, setWaking]     = useState(false);

  const handleFile = (file) => {
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const predict = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    setWaking(false);

    // Show "waking up" message after 3 seconds if still loading (Render cold start)
    const wakeTimer = setTimeout(() => setWaking(true), 3000);

    try {
      const form = new FormData();
      form.append("file", image);

      const res  = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Could not reach the prediction server. Please try again.");
    } finally {
      clearTimeout(wakeTimer);
      setLoading(false);
      setWaking(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.title}>🍎 Fruit Ripeness Detector</h1>
        <p style={styles.subtitle}>Upload a photo of a fruit to predict its ripeness</p>

        {/* Drop zone */}
        <div
          style={styles.dropzone}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById("fileInput").click()}
        >
          {preview
            ? <img src={preview} alt="preview" style={styles.preview} />
            : <p style={styles.dropText}>Drag & drop an image here, or click to select</p>
          }
        </div>
        <input id="fileInput" type="file" accept="image/*" style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])} />

        {/* Predict button */}
        <button style={styles.button} onClick={predict} disabled={!image || loading}>
          {loading ? (waking ? "Waking server up... (~30s)" : "Analysing...") : "Predict Ripeness"}
        </button>

        {error && <p style={styles.error}>{error}</p>}

        {/* Result */}
        {result && (() => {
          const label = LABELS[result.prediction] || {};
          return (
            <div style={{ ...styles.result, borderColor: label.color }}>
              <p style={{ fontSize: 40, margin: 0 }}>{label.emoji}</p>
              <p style={{ ...styles.predLabel, color: label.color }}>
                {result.prediction.toUpperCase()}
              </p>
              <p style={styles.predMessage}>{label.message}</p>
              <p style={styles.confidence}>Confidence: {result.confidence}%</p>

              {/* Confidence bars */}
              <div style={styles.bars}>
                {Object.entries(result.all_scores)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, score]) => (
                    <div key={cls} style={styles.barRow}>
                      <span style={styles.barLabel}>{cls}</span>
                      <div style={styles.barTrack}>
                        <div style={{
                          ...styles.barFill,
                          width: `${score}%`,
                          background: LABELS[cls]?.color || "#888"
                        }} />
                      </div>
                      <span style={styles.barScore}>{score}%</span>
                    </div>
                  ))}
              </div>
            </div>
          );
        })()}

        <p style={styles.footer}>
          Ensemble of MobileNetV2 · EfficientNetB0 · ResNet50 · ~74% accuracy
        </p>
      </div>
    </div>
  );
}

const styles = {
  page:       { minHeight: "100vh", background: "#f4f4f0", display: "flex", alignItems: "center", justifyContent: "center", padding: 20, fontFamily: "system-ui, sans-serif" },
  card:       { background: "#fff", borderRadius: 16, padding: 32, maxWidth: 520, width: "100%", boxShadow: "0 4px 24px rgba(0,0,0,0.08)" },
  title:      { fontSize: 24, fontWeight: 700, margin: "0 0 4px" },
  subtitle:   { color: "#666", margin: "0 0 24px", fontSize: 14 },
  dropzone:   { border: "2px dashed #ddd", borderRadius: 12, padding: 24, cursor: "pointer", textAlign: "center", marginBottom: 16, minHeight: 160, display: "flex", alignItems: "center", justifyContent: "center" },
  dropText:   { color: "#aaa", fontSize: 14 },
  preview:    { maxHeight: 200, maxWidth: "100%", borderRadius: 8 },
  button:     { width: "100%", padding: "12px 0", background: "#3266ad", color: "#fff", border: "none", borderRadius: 10, fontSize: 16, fontWeight: 600, cursor: "pointer", marginBottom: 16 },
  error:      { color: "#E24B4A", fontSize: 14, textAlign: "center" },
  result:     { border: "2px solid", borderRadius: 12, padding: 20, textAlign: "center", marginTop: 8 },
  predLabel:  { fontSize: 28, fontWeight: 700, margin: "8px 0 4px" },
  predMessage:{ color: "#555", margin: "0 0 8px", fontSize: 14 },
  confidence: { fontWeight: 600, margin: "0 0 16px" },
  bars:       { textAlign: "left" },
  barRow:     { display: "flex", alignItems: "center", gap: 8, marginBottom: 6 },
  barLabel:   { width: 72, fontSize: 13, textTransform: "capitalize" },
  barTrack:   { flex: 1, height: 10, background: "#f0f0f0", borderRadius: 5, overflow: "hidden" },
  barFill:    { height: "100%", borderRadius: 5, transition: "width 0.4s ease" },
  barScore:   { width: 40, fontSize: 12, textAlign: "right" },
  footer:     { color: "#bbb", fontSize: 11, textAlign: "center", marginTop: 20 },
};