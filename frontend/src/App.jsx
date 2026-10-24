import { useEffect, useState } from "react";

// The browser calls model-api's NodePort directly (http://localhost:8800 by
// default, since that's how it's exposed from kind on this laptop). Override
// at build time with --build-arg VITE_API_BASE=... if that port ever changes.
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8800";

export default function App() {
  const [info, setInfo] = useState(null);
  const [schema, setSchema] = useState(null);
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/info`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setLoadError(`Could not reach the model API at ${API_BASE}`));

    fetch(`${API_BASE}/schema`)
      .then((r) => r.json())
      .then((s) => {
        setSchema(s);
        const defaults = {};
        for (const [name, prop] of Object.entries(s.properties)) {
          defaults[name] = prop.enum ? prop.enum[0] : (prop.examples?.[0] ?? 0);
        }
        setValues(defaults);
      })
      .catch(() => setLoadError(`Could not reach the model API at ${API_BASE}`));
  }, []);

  function update(name, value) {
    setValues((v) => ({ ...v, [name]: value }));
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applications: [values] }),
    });
    const body = await res.json();
    if (!res.ok) {
      setError(JSON.stringify(body, null, 2));
      return;
    }
    setResult(body.predictions[0]);
  }

  if (loadError) {
    return <p style={{ fontFamily: "system-ui", margin: "2rem", color: "#ea4335" }}>{loadError}</p>;
  }
  if (!schema) {
    return <p style={{ fontFamily: "system-ui", margin: "2rem" }}>Loading...</p>;
  }

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 640, margin: "2rem auto", padding: "0 1rem" }}>
      <h1 style={{ fontSize: "1.3rem" }}>Loan Default Risk</h1>
      {info && (
        <div style={{ color: "#666", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
          model: {info.model} v{info.version}
        </div>
      )}
      <form onSubmit={submit} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem 1rem" }}>
        {Object.entries(schema.properties).map(([name, prop]) => (
          <label key={name} style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem", gap: "0.25rem" }}>
            {name}
            {prop.enum ? (
              <select value={values[name] ?? ""} onChange={(e) => update(name, e.target.value)}>
                {prop.enum.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                min={prop.minimum}
                max={prop.maximum}
                value={values[name] ?? ""}
                onChange={(e) => update(name, Number(e.target.value))}
                required
              />
            )}
          </label>
        ))}
        <button type="submit" style={{ gridColumn: "1 / -1", padding: "0.6rem", fontSize: "1rem", cursor: "pointer", marginTop: "0.5rem" }}>
          Predict
        </button>
      </form>
      {result && (
        <div
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            borderRadius: 6,
            background: result.decision === "repay" ? "#e6f4ea" : "#fce8e6",
            border: `1px solid ${result.decision === "repay" ? "#34a853" : "#ea4335"}`,
          }}
        >
          {result.decision.toUpperCase()} — default probability: {(result.default_probability * 100).toFixed(1)}%
        </div>
      )}
      {error && <pre style={{ color: "#ea4335", marginTop: "1rem", fontSize: "0.85rem", whiteSpace: "pre-wrap" }}>{error}</pre>}
    </div>
  );
}
