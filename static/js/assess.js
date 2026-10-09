const form = document.getElementById('f');
const info = document.getElementById('sampleInfo');
const counters = {};
document.querySelectorAll('[data-label]').forEach(btn => {
  btn.addEventListener('click', async () => {
    const label = btn.dataset.label;
    const n = counters[label] = (counters[label] ?? -1) + 1;
    const res = await fetch(`/api/sample?label=${label}&n=${n}`);
    const d = await res.json();
    if (d.error) { info.textContent = d.error; return; }
    for (const [k, v] of Object.entries(d.features)) {
      const el = form.elements[k];
      if (el) el.value = v;
    }
    info.textContent = `Loaded dataset row #${d.index} (actual outcome: ${d.actual})`;
  });
});