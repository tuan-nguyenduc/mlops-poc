import { useEffect, useState } from "react";

// The browser calls model-api's NodePort directly (http://localhost:8800 by
// default, since that's how it's exposed from kind on this laptop). Override
// at build time with --build-arg VITE_API_BASE=... if that port ever changes.
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8800";

function emptyItem() {
  return { food: "", grams: 100 };
}

export default function App() {
  const [info, setInfo] = useState(null);
  const [items, setItems] = useState([emptyItem()]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/info`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setLoadError(`Could not reach the model API at ${API_BASE}`));
  }, []);

  function update(index, field, value) {
    setItems((rows) => rows.map((row, i) => (i === index ? { ...row, [field]: value } : row)));
  }

  function addRow() {
    setItems((rows) => [...rows, emptyItem()]);
  }

  function removeRow(index) {
    setItems((rows) => rows.filter((_, i) => i !== index));
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    const body = await res.json();
    if (!res.ok) {
      setError(JSON.stringify(body, null, 2));
      return;
    }
    setResult(body);
  }

  if (loadError) {
    return <p style={{ fontFamily: "system-ui", margin: "2rem", color: "#ea4335" }}>{loadError}</p>;
  }

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 640, margin: "2rem auto", padding: "0 1rem" }}>
      <h1 style={{ fontSize: "1.3rem" }}>Meal Nutrition Estimator</h1>
      {info && (
        <div style={{ color: "#666", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
          model: {info.model} v{info.version}
        </div>
      )}
      <form onSubmit={submit}>
        {items.map((row, i) => (
          <div key={i} style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem", alignItems: "center" }}>
            <input
              type="text"
              placeholder="e.g. grilled chicken breast"
              value={row.food}
              onChange={(e) => update(i, "food", e.target.value)}
              required
              style={{ flex: 3, padding: "0.4rem" }}
            />
            <input
              type="number"
              min={1}
              max={5000}
              value={row.grams}
              onChange={(e) => update(i, "grams", Number(e.target.value))}
              required
              style={{ flex: 1, padding: "0.4rem" }}
            />
            <span style={{ fontSize: "0.8rem", color: "#666" }}>g</span>
            {items.length > 1 && (
              <button type="button" onClick={() => removeRow(i)} style={{ padding: "0.3rem 0.6rem" }}>
                ×
              </button>
            )}
          </div>
        ))}
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
          <button type="button" onClick={addRow} style={{ padding: "0.5rem 1rem" }}>
            + Add food
          </button>
          <button type="submit" style={{ padding: "0.5rem 1rem", flex: 1, cursor: "pointer" }}>
            Estimate meal
          </button>
        </div>
      </form>

      {result && (
        <div style={{ marginTop: "1.5rem" }}>
          <div style={{ padding: "1rem", borderRadius: 6, background: "#e6f4ea", border: "1px solid #34a853" }}>
            <strong>Total:</strong> {result.total.calories} kcal &middot; {result.total.protein_g}g protein &middot;{" "}
            {result.total.carbs_g}g carbs &middot; {result.total.fat_g}g fat
          </div>
          <table style={{ width: "100%", marginTop: "1rem", fontSize: "0.85rem", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>
                <th>Food</th>
                <th>Grams</th>
                <th>Calories</th>
                <th>Protein</th>
                <th>Carbs</th>
                <th>Fat</th>
              </tr>
            </thead>
            <tbody>
              {result.items.map((item, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #eee" }}>
                  <td>{item.food}</td>
                  <td>{item.grams}g</td>
                  <td>{item.calories}</td>
                  <td>{item.protein_g}g</td>
                  <td>{item.carbs_g}g</td>
                  <td>{item.fat_g}g</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {error && <pre style={{ color: "#ea4335", marginTop: "1rem", fontSize: "0.85rem", whiteSpace: "pre-wrap" }}>{error}</pre>}
    </div>
  );
}
