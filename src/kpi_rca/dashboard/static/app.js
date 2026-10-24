/* KPI RCA dashboard: renders window.KPI_RCA_DATA (written by `python -m kpi_rca dashboard`).
   Charts are Apache ECharts and icons are Lucide, both bundled in vendor/ so the page opens from disk. */
(function () {
  "use strict";

  const D = window.KPI_RCA_DATA;
  const main = document.getElementById("main");
  const FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const THEME_KEY = "kpi-rca-theme";
  const state = { view: "overview", incident: null, answerKey: true };
  const mounted = [];
  let pending = [];

  // ------------------------------------------------------------------ DOM helpers

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : String(v));
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : String(kid));
    }
    return el;
  }
  const icon = (name) => h("i", { "data-lucide": name, "aria-hidden": "true" });
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function badge(tone, iconName, text) {
    return h("span", { class: `badge ${tone || ""}` }, iconName ? icon(iconName) : null, text);
  }
  function sevBadge(sev) {
    return sev === "SEV1" ? badge("critical", "circle-alert", sev) : badge("warning", "triangle-alert", sev);
  }
  function section(title, desc, content, aside) {
    return h("section", { class: "section" },
      h("div", { class: "section-head" },
        h("div", {}, h("h2", {}, title), desc ? h("p", { class: "section-desc" }, desc) : null),
        aside || null),
      content);
  }
  function pageHead(title, sub, aside) {
    return h("div", { class: "page-head" },
      h("div", {}, h("h1", {}, title), sub ? h("p", { class: "page-sub" }, sub) : null), aside || null);
  }
  const panel = (...kids) => h("div", { class: "panel" }, kids);
  const body = (...kids) => h("div", { class: "panel-body" }, kids);

  /** A metric group: label + badge, value + qualifier, optional detail, footnote. */
  function metric({ label, tag, value, unit, sub, extra, foot }) {
    return h("div", { class: "metric" },
      h("div", { class: "metric-top" }, h("dt", {}, label), tag || null),
      h("dd", { class: "metric-value" }, value, unit ? h("span", { class: "unit" }, unit) : null),
      sub ? h("dd", { class: "metric-sub" }, sub) : null,
      extra ? h("dd", { class: "metric-extra" }, extra) : null,
      foot ? h("dd", { class: "metric-foot" }, foot) : null);
  }
  const metrics = (items, wide) => panel(h("dl", { class: `metrics ${wide ? "wide" : ""}` }, items));

  function progress(rows) {
    return h("ul", { class: "progress" }, rows.filter(Boolean).map(({ label, value, total, text, color }) =>
      h("li", {},
        h("div", { class: "progress-label" }, h("span", {}, label), h("span", { class: "progress-val" }, text || `${num(value)}/${num(total)}`)),
        h("div", { class: "track" }, h("i", { style: `width:${total ? Math.max(0, Math.min(100, (value / total) * 100)) : 0}%${color ? `;background:${color}` : ""}` })))));
  }
  function barList(items, valueFmt, head) {
    const top = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
    return h("div", {},
      head ? h("div", { class: "barlist-head" }, h("span", {}, head[0]), h("span", {}, head[1])) : null,
      h("ul", { class: "barlist" }, items.map((i) => h("li", {},
        h("div", { class: "barlist-bar" }, h("i", { style: `width:${Math.max(1.5, (Math.abs(i.value) / top) * 100)}%` }),
          i.href ? h("a", { class: "barlist-name", href: i.href, title: i.name }, i.name) : h("span", { class: "barlist-name", title: i.name }, i.name)),
        h("span", { class: "barlist-val" }, valueFmt(i.value))))));
  }
  function empty(title, text) {
    return h("div", { class: "empty" }, h("b", {}, title), h("div", {}, text));
  }
  const segText = (segment) => Object.values(segment || {}).join(" / ");
  function segLine(segment) {
    return Object.entries(segment || {}).map(([k, v], i) => [i ? " · " : null, `${k.replace(/_/g, " ")} `, h("b", {}, v)]);
  }

  // ------------------------------------------------------------------ formatting

  const fmt = {};
  const nf = (d) => (fmt[d] = fmt[d] || new Intl.NumberFormat("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
  const num = (x, d = 0) => (x == null ? "-" : nf(d).format(x));
  const pct = (x, d = 1) => (x == null ? "-" : nf(d).format(x * 100) + "%");
  function money(x, cur) {
    if (x == null) return "-";
    const c = cur || "USD", a = Math.abs(x);
    if (a >= 1e6) return `${c} ${nf(2).format(x / 1e6)}M`;
    if (a >= 1e4) return `${c} ${nf(1).format(x / 1e3)}k`;
    return `${c} ${nf(0).format(x)}`;
  }
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const pad = (n) => String(n).padStart(2, "0");
  function ms(iso) {
    const m = /^(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d)/.exec(iso || "");
    return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) : NaN;
  }
  function day(t) {
    const d = new Date(typeof t === "number" ? t : ms(t));
    return `${MON[d.getUTCMonth()]} ${d.getUTCDate()}`;
  }
  function clock(t) {
    const d = new Date(typeof t === "number" ? t : ms(t));
    return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  }
  const when = (t) => `${day(t)}, ${clock(t)}`;
  const signed = (x, d = 1) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${num(Math.abs(x), d)}`;
  const shortNarrator = (n) => String(n || "").replace(/^[^/]+\//, "").replace(/:free$/, "");
  const sentence = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
  function seconds(s) {
    if (s == null) return "-";
    if (s < 1) return `${num(s * 1000, 0)} ms`;
    return `${num(s, 1)} s`;
  }
  function keyDelta(est, key) {
    if (!state.answerKey || !key) return null;
    const d = (est - key) / key, ok = Math.abs(d) <= 0.15;
    return badge(ok ? "good" : "warning", ok ? "check" : "triangle-alert", `${signed(d * 100, 1)}% vs key`);
  }

  // ------------------------------------------------------------------ chart plumbing

  function colors() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    return {
      s1: v("--series-1"), s1soft: v("--series-1-soft"), ref: v("--reference"), grid: v("--grid"),
      ink1: v("--ink-1"), ink2: v("--ink-2"), ink3: v("--ink-3"), surface: v("--surface"), border: v("--border"), tick: v("--tick"),
      critical: v("--critical"), warning: v("--warning"), key: v("--key"),
      critBand: v("--critical-band"), warnBand: v("--warning-band"), accentSoft: v("--accent-soft"),
    };
  }
  function tooltipBase(c) {
    return {
      backgroundColor: c.surface, borderColor: c.border, borderWidth: 1, padding: [8, 10],
      textStyle: { color: c.ink1, fontSize: 12, fontFamily: FONT },
      extraCssText: "border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,.10);",
    };
  }
  function tipHtml(c, title, rows) {
    const mark = (col, shape) => col
      ? `<span style="display:inline-block;width:${shape === "line" ? 10 : 8}px;height:${shape === "line" ? 2 : 8}px;border-radius:2px;background:${col}"></span>` : "";
    return `<div style="font-weight:600;margin-bottom:4px">${esc(title)}</div>` + rows.filter(Boolean).map(([col, label, value, shape]) =>
      `<div style="display:flex;justify-content:space-between;gap:20px;align-items:center;line-height:20px">` +
      `<span style="display:inline-flex;gap:6px;align-items:center;color:${c.ink3}">${mark(col, shape)}${esc(label)}</span>` +
      `<span style="font-weight:500;font-variant-numeric:tabular-nums">${esc(value)}</span></div>`).join("");
  }
  function axisCommon(c) {
    return {
      axisLine: { lineStyle: { color: c.border } }, axisTick: { show: false },
      axisLabel: { color: c.ink3, fontSize: 12, fontFamily: FONT },
      splitLine: { lineStyle: { color: c.grid } },
    };
  }
  const TIME_LABELS = { year: "{yyyy}", month: "{MMM} {d}", day: "{MMM} {d}", hour: "{HH}:{mm}", minute: "{HH}:{mm}" };

  /** A chart container; the chart is created after the view is in the DOM and has a size. */
  function chartBox(cls, build) {
    const el = h("div", { class: `chart ${cls || ""}`, role: "img" });
    pending.push(() => {
      if (!window.echarts) { el.replaceWith(empty("Charts unavailable", "vendor/echarts.min.js did not load.")); return; }
      const c = colors();
      const chart = echarts.init(el, null, { renderer: "canvas" });
      const spec = build(c, chart) || {};
      el.setAttribute("aria-label", spec.label || "chart");
      chart.setOption(Object.assign({
        useUTC: true, animationDuration: REDUCED ? 0 : 250, animationDurationUpdate: REDUCED ? 0 : 200,
        textStyle: { fontFamily: FONT },
      }, spec.option));
      if (spec.on) for (const [ev, fn] of Object.entries(spec.on)) chart.on(ev, fn);
      const ro = new ResizeObserver(() => chart.resize());
      ro.observe(el);
      mounted.push({ chart, ro });
    });
    return el;
  }
  function disposeCharts() {
    while (mounted.length) {
      const { chart, ro } = mounted.pop();
      ro.disconnect();
      chart.dispose();
    }
  }

  // ------------------------------------------------------------------ shared pieces

  const ROWS = D ? D.kpi_hourly : [];
  const BY_TIME = new Map(ROWS.map((r) => [ms(r.t), r]));
  const windowAt = (t) => (D ? D.windows.find((w) => t >= ms(w.start) && t < ms(w.end)) : null);
  const showKey = () => state.answerKey && D.run.has_truth;

  /** The hour inside a window where conversion sits furthest from expected; a drop picks its lowest gap. */
  function windowPeak(w) {
    const a = ms(w.start), b = ms(w.end), drop = w.kpi_type === "kpi_drop";
    let best = null;
    for (const r of ROWS) {
      const t = ms(r.t);
      if (t < a || t >= b) continue;
      const g = (r.conversion - r.expected) * 100;
      if (!best || (drop ? g < best.g : Math.abs(g) > Math.abs(best.g))) best = { t, g };
    }
    return best;
  }

  /** Hourly conversion against the expected baseline (top) and the gap between them in pp (bottom),
      each detected window shaded, coloured and labelled where it peaks. */
  function timeline({ from, to, focus, slider }) {
    return chartBox(slider ? "duo" : "duo-sm", (c, chart) => {
      const t0 = from != null ? from : ms(ROWS[0].t), t1 = to != null ? to : ms(ROWS[ROWS.length - 1].t);
      const inRange = (w) => ms(w.end) > t0 && ms(w.start) < t1;
      const bands = D.windows.map((w) => {
        const drop = w.kpi_type === "kpi_drop";
        const focused = focus && w.incident_id === focus;
        return [{
          name: w.window_id, xAxis: ms(w.start),
          itemStyle: { color: drop ? c.critBand : c.warnBand, borderColor: drop ? c.critical : c.warning, borderWidth: focused ? 1 : 0 },
        }, { xAxis: ms(w.end) }];
      });
      const keys = showKey() ? D.truth_incidents.map((t) => [{
        name: t.incident_id, xAxis: ms(t.start),
        itemStyle: { color: "rgba(0,0,0,0)", borderColor: c.key, borderWidth: 1, borderType: "dashed" },
      }, { xAxis: ms(t.end) }]) : [];
      const narrow = chart.getWidth() < 640;
      const peaks = D.windows.filter(inRange).map((w) => {
        const p = windowPeak(w);
        if (!p) return null;
        const inc = D.incidents.find((i) => i.incident_id === w.incident_id);
        const what = w.kpi_type !== "kpi_drop" ? "Surge, skipped" : inc ? segText(inc.segment) : (w.top_segment || w.window_id);
        return {
          name: w.window_id, coord: [p.t, p.g], value: p.g,
          label: { position: p.g < 0 ? "bottom" : "top", formatter: narrow ? `{v|${signed(p.g, 1)} pp}` : `{v|${signed(p.g, 1)} pp}\n{s|${what}}` },
        };
      }).filter(Boolean);

      const H = chart.getHeight(), S = slider ? 44 : 0;
      const room = H - 32 - 30 - 26 - S;
      const topH = Math.round(room * 0.6), gapTop = 32 + topH + 30, gapH = room - topH;
      const ax = axisCommon(c);
      const xBase = Object.assign({}, ax, { type: "time", min: t0, max: t1, splitLine: { show: false } });
      const title = (text, top) => ({ text, left: 0, top, textStyle: { color: c.ink2, fontSize: 12, fontWeight: 500, fontFamily: FONT } });
      const pctLabel = (v) => `${num(v * 100, Math.abs(v * 100 - Math.round(v * 100)) > 1e-9 ? 1 : 0)}%`;
      return {
        label: "Hourly checkout conversion against the expected baseline, and the gap between them, with detected windows shaded and labelled",
        option: {
          title: [title("Conversion", 0), title("Gap vs expected, pp", gapTop - 24)],
          grid: [
            { left: 48, right: 12, top: 32, height: topH },
            { left: 48, right: 12, top: gapTop, height: gapH },
          ],
          axisPointer: { link: [{ xAxisIndex: "all" }] },
          tooltip: Object.assign(tooltipBase(c), {
            trigger: "axis", axisPointer: { type: "line", lineStyle: { color: c.border } },
            formatter: (ps) => {
              const t = ps[0].value[0], r = BY_TIME.get(t), w = windowAt(t);
              if (!r) return "";
              return tipHtml(c, when(t), [
                [c.s1, "Observed", pct(r.conversion, 2), "line"],
                [c.ref, "Expected", pct(r.expected, 2), "line"],
                [null, "Gap", `${signed((r.conversion - r.expected) * 100, 2)} pp`],
                [null, "Attempts", num(r.attempts)],
                w ? [w.kpi_type === "kpi_drop" ? c.critical : c.warning, w.kpi_type === "kpi_drop" ? "KPI drop" : "Volume surge", w.incident_id || "skipped"] : null,
              ]);
            },
          }),
          xAxis: [
            Object.assign({}, xBase, { gridIndex: 0, axisLabel: { show: false } }),
            Object.assign({}, xBase, { gridIndex: 1, axisLabel: Object.assign({}, ax.axisLabel, { formatter: TIME_LABELS, hideOverlap: true }) }),
          ],
          yAxis: [
            Object.assign({}, ax, { gridIndex: 0, type: "value", scale: true, splitNumber: 4, axisLine: { show: false },
              axisLabel: Object.assign({}, ax.axisLabel, { formatter: pctLabel }) }),
            Object.assign({}, ax, { gridIndex: 1, type: "value", splitNumber: 2, axisLine: { show: false },
              min: (v) => Math.min(-1, v.min * 1.9), max: (v) => Math.max(1, v.max * 1.9),
              axisLabel: Object.assign({}, ax.axisLabel, { showMinLabel: false, showMaxLabel: false, formatter: (v) => signed(v, 0) }) }),
          ],
          dataZoom: [
            { type: "inside", xAxisIndex: [0, 1], filterMode: "none" },
            slider ? {
              type: "slider", xAxisIndex: [0, 1], filterMode: "none", height: 24, bottom: 8, left: 48, right: 12,
              borderColor: c.border, backgroundColor: "rgba(0,0,0,0)", fillerColor: c.accentSoft,
              dataBackground: { lineStyle: { color: c.ref, width: 1 }, areaStyle: { color: "rgba(0,0,0,0)" } },
              selectedDataBackground: { lineStyle: { color: c.s1, width: 1 }, areaStyle: { color: "rgba(0,0,0,0)" } },
              handleStyle: { color: c.surface, borderColor: c.ref }, moveHandleStyle: { color: c.border }, moveHandleSize: 4,
              textStyle: { color: c.ink3, fontSize: 11 }, labelFormatter: (v) => when(v),
            } : null,
          ].filter(Boolean),
          series: [
            {
              name: "Observed", type: "line", z: 3, xAxisIndex: 0, yAxisIndex: 0,
              data: ROWS.map((r) => [ms(r.t), r.conversion]), showSymbol: false, symbolSize: 6,
              lineStyle: { width: 2, color: c.s1 }, itemStyle: { color: c.s1, borderColor: c.surface, borderWidth: 2 },
              markArea: { data: bands, label: { show: false } },
            },
            {
              name: "Expected", type: "line", z: 2, xAxisIndex: 0, yAxisIndex: 0,
              data: ROWS.map((r) => [ms(r.t), r.expected]), showSymbol: false, symbolSize: 5,
              lineStyle: { width: 1.5, type: "dashed", color: c.ref }, itemStyle: { color: c.ref },
              markArea: { silent: true, data: keys, label: { show: false } },
            },
            {
              name: "Gap", type: "bar", xAxisIndex: 1, yAxisIndex: 1, barWidth: "70%",
              data: ROWS.map((r) => {
                const t = ms(r.t), w = windowAt(t);
                return { value: [t, (r.conversion - r.expected) * 100],
                  itemStyle: { color: !w ? c.tick : w.kpi_type === "kpi_drop" ? c.critical : c.warning } };
              }),
              markPoint: {
                symbol: "circle", symbolSize: 1, itemStyle: { color: "rgba(0,0,0,0)" },
                label: { show: true, distance: 6, lineHeight: 15, rich: {
                  v: { color: c.ink1, fontSize: 12, fontWeight: 600, fontFamily: FONT },
                  s: { color: c.ink3, fontSize: 11, fontFamily: FONT } } },
                data: peaks,
              },
            },
          ],
        },
        on: {
          click: (p) => {
            let w = null;
            if (p.componentType === "markArea" || p.componentType === "markPoint") w = D.windows.find((x) => x.window_id === p.name);
            else if (p.seriesName === "Gap") w = windowAt(p.value[0]);
            if (w && w.incident_id) location.hash = `#incident/${w.incident_id}`;
          },
        },
      };
    });
  }
  function timelineLegend(from, to) {
    const inRange = (w) => (from == null || ms(w.end) > from) && (to == null || ms(w.start) < to);
    const surge = D.windows.some((w) => w.kpi_type !== "kpi_drop" && inRange(w));
    return h("div", { class: "legend" },
      h("span", {}, h("i", { class: "key" }), "Observed"),
      h("span", {}, h("i", { class: "key dash" }), "Expected"),
      h("span", {}, h("i", { class: "key sq drop" }), "KPI drop"),
      surge ? h("span", {}, h("i", { class: "key sq surge" }), "Volume surge") : null,
      showKey() && D.truth_incidents.length ? h("span", {}, h("i", { class: "key answer" }), "Answer key") : null);
  }

  // ------------------------------------------------------------------ overview

  const narratorName = () => (D.run.model ? shortNarrator(D.run.model) : D.run.narrator);

  function runMetrics() {
    const incs = D.incidents;
    const cur = incs.length ? incs[0].currency : "USD";
    const lost = incs.reduce((s, i) => s + (i.lost_revenue || 0), 0);
    const orders = incs.reduce((s, i) => s + (i.lost_orders || 0), 0);
    const keyLost = incs.reduce((s, i) => s + ((i.truth && i.truth.lost_revenue) || 0), 0);
    const matched = incs.filter((i) => i.truth && i.truth.segment_match).length;
    const found = new Set(incs.filter((i) => i.truth).map((i) => i.truth.incident_id)).size;
    const surges = D.windows.filter((w) => w.kpi_type !== "kpi_drop").length;
    const passed = incs.filter((i) => i.guardrails.passed).length;
    const key = showKey();
    const windows = `${D.windows.length} windows · ${surges} surge${surges === 1 ? "" : "s"} skipped`;

    return metrics([
      key ? metric({ label: "Incidents found", value: String(found), unit: `of ${D.truth_incidents.length}`, sub: windows })
        : metric({ label: "Incidents", value: String(incs.length), sub: windows }),
      key ? metric({ label: "Root cause correct", value: String(matched), unit: `of ${incs.length}`,
        tag: matched === incs.length ? badge("good", "check", "exact") : badge("warning", "triangle-alert", "partial"), sub: "segment matches the key" })
        : metric({ label: "Memos passing guardrails", value: String(passed), unit: `of ${incs.length}` }),
      metric({ label: "Revenue lost", tag: keyDelta(lost, keyLost), value: money(lost, cur),
        sub: key ? `key ${money(keyLost, cur)}` : `${num(orders)} orders` }),
      metric({ label: "Runtime", value: num(D.run.timings_s.total_s, 1), unit: "s", sub: `${narratorName()} narrator` }),
    ]);
  }

  function incidentTable() {
    if (!D.incidents.length) return empty("No incidents", "No KPI drop windows in this run.");
    const key = showKey();
    const head = ["Incident", "Root-cause segment", "Started", "Conversion", "Lost revenue"].concat(key ? ["Answer key"] : [], [""]);
    const rows = D.incidents.map((inc) => {
      const go = () => { location.hash = `#incident/${inc.incident_id}`; };
      return h("tr", { class: "link", onclick: go },
        h("td", { class: "nowrap" }, h("a", { class: "id", href: `#incident/${inc.incident_id}` }, inc.incident_id), " ", sevBadge(inc.severity)),
        h("td", {}, h("span", { class: "strong" }, segText(inc.segment))),
        h("td", { class: "nowrap" }, `${when(inc.start)} · ${num(inc.duration_hours, 0)} h`),
        h("td", { class: "num" }, h("span", { class: "strong" }, pct(inc.conversion_observed, 1)), h("span", { class: "muted" }, ` / ${pct(inc.conversion_expected, 1)}`)),
        h("td", { class: "num strong" }, money(inc.lost_revenue, inc.currency)),
        key ? h("td", { class: "nowrap" }, !inc.truth ? h("span", { class: "muted" }, "none")
          : inc.truth.segment_match ? badge("good", "check", inc.truth.incident_id) : badge("critical", "x", "differs")) : null,
        h("td", { class: "go" }, icon("chevron-right")));
    });
    return h("div", { class: "panel flush" }, h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, head.map((x, i) => h("th", { class: [3, 4].includes(i) ? "num" : null }, x)))),
      h("tbody", {}, rows))));
  }

  function renderOverview() {
    const first = ROWS[0].t, last = ROWS[ROWS.length - 1].t;
    return [
      pageHead("Checkout health", `${day(first)} to ${day(last)} · ${num(ROWS.length / 24, 0)} days hourly · ${num(ROWS.reduce((s, r) => s + r.attempts, 0))} attempts`),
      section("This run", null, runMetrics()),
      section("Checkout conversion", null, panel(body(timeline({ slider: true }))), timelineLegend()),
      section("Incidents", null, incidentTable()),
    ];
  }

  // ------------------------------------------------------------------ incident case

  const CHECKS = ["Schema and length", "Numbers grounded", "Segment matches RCA", "Quotes match tickets"];
  const more = (label, ...kids) => h("details", { class: "more" }, h("summary", {}, icon("chevron-right"), label), kids);

  function renderIncident() {
    if (!D.incidents.length) return [empty("No incidents in this run", "The detector found no KPI drop windows.")];
    const inc = D.incidents.find((i) => i.incident_id === state.incident) || D.incidents[0];
    state.incident = inc.incident_id;
    const truth = state.answerKey ? inc.truth : null;

    const switcher = D.incidents.length > 1 ? h("div", { class: "switcher", role: "group", "aria-label": "Choose incident" },
      D.incidents.map((i) => h("button", {
        type: "button", "aria-pressed": i.incident_id === inc.incident_id ? "true" : "false",
        onclick: () => { location.hash = `#incident/${i.incident_id}`; },
      }, i.incident_id))) : null;

    const head = h("div", { class: "page-head" },
      h("div", { style: "min-width:0;flex:1 1 520px" },
        h("div", { class: "page-meta" }, sevBadge(inc.severity), h("span", { class: "id" }, inc.incident_id),
          h("span", {}, `${when(inc.start)} to ${clock(inc.end)} UTC`), h("span", {}, `${inc.confidence} confidence`)),
        h("h1", {}, sentence(inc.headline.replace(/^SEV\d+:\s*/, ""))),
        h("div", { class: "seg-line" }, h("span", {}, segLine(inc.segment)),
          truth ? (truth.segment_match ? badge("good", "check", `matches key ${truth.incident_id}`)
            : badge("critical", "x", `key: ${segText(truth.segment)}`)) : null)),
      switcher);

    const gap = (inc.conversion_observed - inc.conversion_expected) * 100;
    const stats = metrics([
      metric({ label: "Conversion", tag: badge("critical", "trending-down", `${signed(gap, 1)} pp`),
        value: pct(inc.conversion_observed, 1), sub: `expected ${pct(inc.conversion_expected, 1)}` }),
      metric({ label: "Revenue lost", tag: truth ? keyDelta(inc.lost_revenue, truth.lost_revenue) : null,
        value: money(inc.lost_revenue, inc.currency), sub: truth ? `key ${money(truth.lost_revenue, inc.currency)}` : null }),
      metric({ label: "Orders lost", tag: truth ? keyDelta(inc.lost_orders, truth.lost_orders) : null,
        value: num(inc.lost_orders), sub: truth ? `key ${num(truth.lost_orders)}` : null }),
      metric({ label: "Duration", value: num(inc.duration_hours, 0), unit: "h",
        sub: `${money(inc.revenue_at_risk_per_hour, inc.currency)} at risk per hour` }),
    ]);

    const from = ms(inc.start) - 18 * 36e5, to = ms(inc.end) + 18 * 36e5;
    const zoom = section("Conversion, 18 h either side", null,
      panel(body(timeline({ from, to, focus: inc.incident_id }))), timelineLegend(from, to));

    const fw = inc.failure_rate_window || 0, fb = inc.failure_rate_baseline || 0;
    const why = section("Why this segment", null,
      panel(body(
        h("div", { class: "ratio" }, h("span", { class: "v" }, fb ? `${num(fw / fb, 1)}x` : "-"),
          h("span", { class: "l" }, `baseline failure rate · ${num(inc.segment_attempts)} attempts`)),
        progress([
          { label: "Failure rate, window", value: fw, total: 1, text: pct(fw, 1), color: "var(--critical)" },
          { label: "Failure rate, baseline", value: fb, total: 1, text: pct(fb, 1), color: "var(--reference)" },
          { label: "Share of excess failures", value: Math.min(1, inc.share_of_excess_failures || 0), total: 1, text: pct(inc.share_of_excess_failures, 0) },
        ]))));

    const shap = section("SHAP attribution", null,
      panel(body(barList((inc.shap || []).map((s) => ({ name: s.feature, value: s.contribution })), (v) => num(v, 2), ["Feature", "SHAP"]))));

    const codes = inc.error_codes || [];
    const keyCode = truth && truth.error_code;
    const errors = section("Error codes", null,
      panel(body(barList(codes.slice(0, 6).map((e) => ({ name: e.code, value: e.count })), (v) => num(v), ["Code", "Failures"]))),
      keyCode ? (codes[0] && codes[0].code === keyCode ? badge("good", "check", "top code matches key") : badge("warning", "triangle-alert", `key ${keyCode}`)) : null);

    const tk = inc.tickets || [];
    const ticketRow = (t) => h("li", {},
      h("div", { class: "tk" }, h("div", { class: "text", title: t.text }, t.text),
        h("div", { class: "meta" }, [t.ticket_id, t.channel, when(t.created_at)].filter(Boolean).join(" · "))),
      h("div", { class: "sim", title: "Similarity" }, h("span", { class: "track" }, h("i", { style: `width:${Math.round((t.similarity || 0) * 100)}%` })),
        h("span", { class: "small" }, num(t.similarity, 2))));
    const tickets = section("Similar tickets", null,
      h("div", { class: "panel flush" }, h("ul", { class: "tickets" }, tk.slice(0, 3).map(ticketRow)),
        tk.length > 3 ? more(`${tk.length - 3} more`, h("ul", { class: "tickets" }, tk.slice(3).map(ticketRow))) : null),
      h("span", { class: "section-aside" }, `${num(inc.similar_tickets)} found${inc.ticket_spike_ratio ? ` · ${num(inc.ticket_spike_ratio, 1)}x usual rate` : ""}`));

    const actions = section("Actions", null, h("div", { class: "panel flush" }, h("ul", { class: "actions" },
      (inc.actions || []).map((a) => h("li", { title: a.rationale || null },
        badge(a.priority === "P0" ? "critical" : "", null, a.priority),
        h("span", { class: "what" }, a.action),
        a.owner_team ? h("span", { class: "owner" }, a.owner_team) : null)))));

    const g = inc.guardrails;
    const narr = g.narrator_used || D.run.narrator;
    const violations = (g.attempt_violations || []).flat();
    const memo = section("Memo", null,
      h("div", { class: "panel flush" },
        h("div", { class: "panel-body" },
          h("ul", { class: "chips" }, CHECKS.map((name) => h("li", {}, badge(g.passed ? "good" : "critical", g.passed ? "check" : "x", name)))),
          h("div", { class: "memo-meta" }, [`${shortNarrator(narr)} narrator`, `attempt ${g.attempts || 1} of ${(D.run.max_retries || 0) + 1}`,
            g.fallback_used ? "template fallback" : null, violations.length ? `rejected earlier: ${violations.join("; ")}` : null].filter(Boolean).join(" · "))),
        more("Read the memo", h("div", { class: "prose" },
          h("div", {}, h("h3", {}, "Summary"), h("p", {}, inc.summary)),
          h("div", {}, h("h3", {}, "Root cause"), h("p", {}, inc.root_cause))))),
      g.passed ? badge("good", "check", "Guardrails passed") : badge("critical", "x", "Guardrails failed"));

    return [
      head,
      section("Impact", null, stats),
      zoom,
      h("div", { class: "grid-3" }, why, shap, errors),
      h("div", { class: "cols" },
        h("div", { class: "col" }, tickets),
        h("div", { class: "col" }, memo, actions)),
    ];
  }

  // ------------------------------------------------------------------ benchmark

  function ciBar(row) {
    const box = h("div", { class: "ci" }, h("span", { class: "track" }));
    if (row.value == null) return box;
    const tgt = /(-?\d+(?:\.\d+)?)/.exec(row.target || "");
    const vals = [row.value].concat(row.ci95 || [], tgt ? [+tgt[1]] : []);
    let lo = Math.min(0, ...vals), hi = Math.max(...vals);
    if (lo >= 0 && hi <= 1) { lo = 0; hi = 1; } else hi = hi * 1.1 || 1;
    const x = (v) => `${((v - lo) / (hi - lo || 1)) * 100}%`;
    if (row.ci95) box.append(h("span", { class: "band", style: `left:${x(row.ci95[0])};width:calc(${x(row.ci95[1])} - ${x(row.ci95[0])})` }));
    if (tgt && /[<>]=?/.test(row.target)) box.append(h("span", { class: "mark", title: `target ${row.target}`, style: `left:${x(+tgt[1])}` }));
    box.append(h("span", { class: "pt", style: `left:${x(row.value)}` }));
    return box;
  }
  const STATUS = { met: ["good", "check", "Met"], missed: ["critical", "x", "Missed"], reported: ["", "minus", "Reported"] };

  function narratorChart() {
    const rows = D.narrators;
    return chartBox("", (c) => ({
      label: "Memo pass rate per narrator, first attempt and after retries, with median latency",
      option: {
        grid: { left: 40, right: 72, top: 20, bottom: 48 },
        tooltip: Object.assign(tooltipBase(c), {
          trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: c.accentSoft } },
          formatter: (ps) => {
            const r = rows[ps[0].dataIndex];
            return tipHtml(c, shortNarrator(r.narrator), [
              [c.s1soft, "First attempt", num(r.first_attempt, 2)], [c.s1, "After retries", num(r.after_retries, 2)],
              [null, "Template fallback", num(r.fallback, 2)], [null, "Memos", num(r.memos)],
              r.median_latency_s != null ? [null, "Median latency", seconds(r.median_latency_s)] : null]);
          },
        }),
        xAxis: Object.assign(axisCommon(c), { type: "category", splitLine: { show: false },
          data: rows.map((r) => `{n|${shortNarrator(r.narrator)}}\n{l|${r.median_latency_s == null ? "deterministic" : `${seconds(r.median_latency_s)} median`}}`),
          axisLabel: { interval: 0, lineHeight: 16, rich: {
            n: { color: c.ink1, fontSize: 12, fontWeight: 500, fontFamily: FONT },
            l: { color: c.ink3, fontSize: 11, fontFamily: FONT } } } }),
        yAxis: Object.assign(axisCommon(c), { type: "value", min: 0, max: 1, interval: 0.25, axisLine: { show: false } }),
        series: [
          { name: "First attempt", type: "bar", barGap: "8%", barMaxWidth: 28, data: rows.map((r) => r.first_attempt),
            itemStyle: { color: c.s1soft, borderRadius: [4, 4, 0, 0] },
            label: { show: true, position: "top", color: c.ink2, fontSize: 11, formatter: (p) => num(p.value, 2) },
            markLine: { symbol: "none", silent: true, lineStyle: { color: c.ink3, type: "dashed", width: 1 },
              label: { position: "end", color: c.ink3, fontSize: 11, formatter: (p) => `target ${num(p.value, 2)}` },
              data: [{ yAxis: 0.8 }, { yAxis: 0.95 }] } },
          { name: "After retries", type: "bar", barMaxWidth: 28, data: rows.map((r) => r.after_retries),
            itemStyle: { color: c.s1, borderRadius: [4, 4, 0, 0] },
            label: { show: true, position: "top", color: c.ink2, fontSize: 11, formatter: (p) => num(p.value, 2) } },
        ],
      },
    }));
  }

  function scoreRows(rows) {
    const groups = [];
    for (const r of rows) {
      let g = groups.find((x) => x.name === r.experiment);
      if (!g) groups.push((g = { name: r.experiment, rows: [] }));
      g.rows.push(r);
    }
    return groups.map((g) => {
      const judged = g.rows.filter((r) => r.status !== "reported").length;
      return [
        h("div", { class: "exp-title" }, g.name,
          judged ? h("span", { class: "muted" }, `${g.rows.filter((r) => r.status === "met").length} of ${judged} met`) : null),
        g.rows.map((r, i) => {
          const [tone, ic, text] = STATUS[r.status] || STATUS.reported;
          const d = r.value != null && Math.abs(r.value) < 0.1 && r.value !== 0 ? 3 : 2;
          return h("div", { class: `metric-row ${i === 0 ? "first" : ""}` },
            h("span", {}, r.metric),
            h("span", { class: "val", title: r.ci95 ? `95% CI ${num(r.ci95[0], d)} to ${num(r.ci95[1], d)}` : null }, r.value == null ? r.text : num(r.value, d)),
            ciBar(r),
            h("span", { class: "tgt" }, r.target),
            h("span", {}, badge(tone, ic, text)));
        }),
      ];
    });
  }

  function renderBenchmark() {
    const B = D.benchmark;
    const head = pageHead("Benchmark", B ? [
      `${num(B.settings.n_scenarios)} random scenarios × ${num(B.settings.days)} days`,
      `seed ${B.settings.seed}`,
      B.settings.runtime_s ? `${num(B.settings.runtime_s / 60, 0)} min to run` : null].filter(Boolean).join(" · ") : "Randomised scenarios with a known answer key.");
    if (!B && !D.narrators.length) {
      return [head, section("No results yet", null, empty("No benchmark results", "Run `python -m kpi_rca evaluate` and rebuild the dashboard."))];
    }
    const out = [head];
    if (B) {
      const met = B.rows.filter((r) => r.status === "met").length;
      const missed = B.rows.filter((r) => r.status === "missed");
      const judged = met + missed.length;
      out.push(section("Targets", null, metrics([
        metric({ label: "Targets met", value: String(met), unit: `of ${judged}`,
          extra: progress([{ label: "Met", value: met, total: judged, text: pct(judged ? met / judged : 0, 0) }]) }),
        metric({ label: "Targets missed", tag: missed.length ? badge("critical", "x", "shown below") : badge("good", "check", "none"),
          value: String(missed.length), sub: missed.map((r) => r.metric).join(", ") || null }),
        metric({ label: "Scenarios", value: num(B.settings.n_scenarios), sub: (B.settings.backends || []).join(" + ") || null }),
      ])));
    }
    if (D.narrators.length) {
      out.push(section("Memo pass rate by narrator", null, panel(body(narratorChart())),
        h("div", { class: "legend" }, h("span", {}, h("i", { class: "key sq s1-soft" }), "First attempt"), h("span", {}, h("i", { class: "key sq s1" }), "After retries"),
          h("span", {}, h("i", { class: "key tgt" }), "Targets"))));
    }
    if (B) {
      const judged = B.rows.filter((r) => r.status !== "reported"), reported = B.rows.filter((r) => r.status === "reported");
      const card = h("div", { class: "panel flush" },
        h("div", { class: "metric-head" }, h("span", {}, "Metric"), h("span", { class: "r" }, "Value"), h("span", { class: "c-ci" }, "95% CI and target"), h("span", { class: "c-tgt" }, "Target"), h("span", {}, "Status")),
        scoreRows(judged),
        reported.length ? more(`${reported.length} reported without a target`, scoreRows(reported)) : null);
      out.push(section("Scorecard", null, card,
        h("div", { class: "legend" }, h("span", {}, h("i", { class: "key dot" }), "Mean"), h("span", {}, h("i", { class: "key sq band" }), "95% CI"),
          h("span", {}, h("i", { class: "key mark" }), "Target"))));
    }
    return out;
  }

  // ------------------------------------------------------------------ real data

  function renderRealData() {
    const R = D.rees46;
    const head = pageHead(R ? `Real e-commerce data, REES46 ${R.month}` : "Real e-commerce data",
      R ? `${num(R.attempts)} checkout attempts · ${R.days} days · ${R.injected} incidents injected into a copy` : null);
    if (!R) return [head, section("No results yet", null, empty("No REES46 results", "Run scripts/validate_rees46.py and rebuild the dashboard."))];
    const detected = R.incidents.filter((i) => i.detected).length;
    const exact = R.incidents.filter((i) => i.exact).length;

    const stats = metrics([
      metric({ label: "Incidents detected", value: `${detected}`, unit: `of ${R.injected}`,
        extra: progress([{ label: "Recall", value: R.detection_recall, total: 1, text: pct(R.detection_recall, 0) }]) }),
      metric({ label: "Exact root cause, guided", value: pct(R.rca_exact_guided, 0), sub: `${exact} of ${detected} detected` }),
      metric({ label: "Exact root cause, brute force", value: pct(R.rca_exact_brute, 0) }),
      metric({ label: "False drop windows", value: num(R.false_drop_windows_per_day, 2), unit: "per day", sub: "untouched month" }),
    ]);

    const scatter = chartBox("tall", (c) => {
      const pts = (flag) => R.incidents.filter((i) => i.detected === flag).map((i) => ({ value: [i.visibility_z, i.failure_rate], inc: i }));
      return {
        label: "Injected incidents by visibility in total conversion and failure rate, detected versus missed",
        option: {
          grid: { left: 60, right: 16, top: 36, bottom: 52 },
          legend: { top: 0, right: 0, textStyle: { color: c.ink2, fontSize: 12 }, itemWidth: 10, itemHeight: 10 },
          tooltip: Object.assign(tooltipBase(c), {
            trigger: "item",
            formatter: (p) => {
              const i = p.data.inc;
              return tipHtml(c, `${i.incident} · ${i.segment}`, [
                [null, "Starts", i.start], [null, "Duration", `${num(i.duration_h, 0)} h`],
                [null, "Failure rate", pct(i.failure_rate, 1)], [null, "Visibility", `${num(i.visibility_z, 1)} σ`],
                [i.detected ? c.s1 : c.ref, "Detected", i.detected ? "yes" : "no"],
                i.detected ? [null, "Found segment", i.found_segment || "none named"] : null]);
            },
          }),
          xAxis: Object.assign(axisCommon(c), { type: "value", name: "← more visible   dip in total conversion, σ", nameLocation: "middle", nameGap: 32,
            nameTextStyle: { color: c.ink3, fontSize: 12 } }),
          yAxis: Object.assign(axisCommon(c), { type: "value", name: "segment failure rate", nameLocation: "middle", nameGap: 42, nameTextStyle: { color: c.ink3, fontSize: 12 },
            axisLine: { show: false }, axisLabel: { color: c.ink3, fontSize: 12, formatter: (v) => pct(v, 0) } }),
          series: [
            { name: `Detected (${detected})`, type: "scatter", data: pts(true), symbol: "circle", symbolSize: 12,
              itemStyle: { color: c.s1, borderColor: c.surface, borderWidth: 2 } },
            { name: `Missed (${R.incidents.length - detected})`, type: "scatter", data: pts(false), symbol: "diamond", symbolSize: 13,
              itemStyle: { color: "rgba(0,0,0,0)", borderColor: c.ref, borderWidth: 2 } },
          ],
        },
      };
    });

    const table = h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Incident", "Injected segment", "Start", "Length", "Failure rate", "Visibility", "Detected", "Found segment"].map((x, i) =>
        h("th", { class: [3, 4, 5].includes(i) ? "num" : null }, x)))),
      h("tbody", {}, R.incidents.map((i) => h("tr", {},
        h("td", { class: "nowrap" }, h("span", { class: "id" }, i.incident)), h("td", {}, i.segment), h("td", { class: "nowrap" }, i.start),
        h("td", { class: "num" }, `${num(i.duration_h, 0)} h`), h("td", { class: "num" }, pct(i.failure_rate, 1)), h("td", { class: "num" }, `${num(i.visibility_z, 1)} σ`),
        h("td", {}, i.detected ? badge("good", "check", "Detected") : badge("", "eye-off", "Missed")),
        h("td", {}, !i.detected ? h("span", { class: "muted" }, "-") : i.exact ? badge("good", "check", i.found_segment)
          : (i.found_segment || h("span", { class: "muted" }, "no segment named"))))))));

    return [head,
      section("Results", null, stats),
      section("Detected vs missed, by visibility", null, panel(body(scatter))),
      section("Injected incidents", null, h("div", { class: "panel flush" }, more(`Show all ${R.incidents.length}`, table)))];
  }

  // ------------------------------------------------------------------ shell

  const VIEWS = {
    overview: ["Overview", renderOverview],
    incident: ["Incidents", renderIncident],
    benchmark: ["Benchmark", renderBenchmark],
    realdata: ["Real data", renderRealData],
  };

  function parseHash() {
    const [view, id] = location.hash.replace(/^#/, "").split("/");
    state.view = VIEWS[view] ? view : "overview";
    if (state.view === "incident" && id) state.incident = decodeURIComponent(id);
  }

  function render() {
    disposeCharts();
    pending = [];
    const [title, fn] = VIEWS[state.view];
    main.replaceChildren(...fn());
    const crumbs = document.getElementById("crumbs");
    crumbs.replaceChildren(h("span", {}, "KPI RCA"), icon("chevron-right"), h("b", {}, title),
      ...(state.view === "incident" && state.incident ? [icon("chevron-right"), h("b", {}, state.incident)] : []));
    for (const a of document.querySelectorAll(".nav-item")) {
      if (a.dataset.view === state.view) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
    document.title = `${title} · KPI RCA`;
    if (window.lucide) lucide.createIcons();
    for (const fnc of pending) fnc();
    pending = [];
  }

  function shell() {
    const r = D.run;
    document.getElementById("nav-incidents").textContent = D.incidents.length ? String(D.incidents.length) : "";
    const row = (k, v) => h("div", { class: "row" }, h("span", {}, k), h("b", {}, v));
    document.getElementById("run-card").replaceChildren(
      h("div", { class: "title" }, "Run details"),
      row("Narrator", narratorName()),
      row("Data", r.data_dir), row("Run", r.dir.split("/").pop()),
      row("Built", D.generated_at.replace("T", " ").slice(0, 16)));
    document.getElementById("status-chip").replaceChildren(icon("circle-check"), "Run complete", h("span", { class: "muted" }, `· ${seconds(r.timings_s.total_s)}`));
    const keyBox = document.getElementById("answer-key");
    if (!r.has_truth) { keyBox.checked = false; keyBox.disabled = true; keyBox.closest(".switch").title = "This data directory has no answer key"; }
    state.answerKey = keyBox.checked;
    keyBox.addEventListener("change", () => { state.answerKey = keyBox.checked; render(); });
    document.getElementById("footer").textContent =
      `Built ${D.generated_at.replace("T", " ").slice(0, 16)} from ${r.dir} · answer key from the data generator, never read by the pipeline · ECharts, Lucide, Inter`;
  }

  function initTheme() {
    const btn = document.getElementById("theme-toggle");
    const order = ["auto", "light", "dark"];
    let mode = "auto";
    try { mode = localStorage.getItem(THEME_KEY) || "auto"; } catch (e) { /* storage blocked */ }
    const apply = () => {
      if (mode === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", mode);
      btn.querySelector("span").textContent = `Theme: ${mode}`;
    };
    apply();
    btn.addEventListener("click", () => {
      mode = order[(order.indexOf(mode) + 1) % order.length];
      try { localStorage.setItem(THEME_KEY, mode); } catch (e) { /* storage blocked */ }
      apply();
      if (D) render();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (mode === "auto" && D) render(); });
  }

  initTheme();
  if (!D) {
    main.replaceChildren(empty("No run data loaded",
      "This is the page template. Build the dashboard with `python -m kpi_rca dashboard`, then open outputs/dashboard/index.html."));
    document.getElementById("crumbs").textContent = "No data";
    if (window.lucide) lucide.createIcons();
    return;
  }
  shell();
  parseHash();
  render();
  window.addEventListener("hashchange", () => {
    parseHash();
    render();
    window.scrollTo({ top: 0 });
    main.focus({ preventScroll: true });
  });
})();
