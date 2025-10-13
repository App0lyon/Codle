/* Minimal front-end glue to fetch today's problem from the backend and hydrate the UI.
   - Uses Europe/Paris timezone for "today"
   - GET /problems/{date}; if 404, attempts POST /problems with difficulty from localStorage or default 'medium'
   - Populates: title, difficulty badge, description, CodeMirror starter code, language select
   - Optionally renders hints if returned by POST endpoint
*/

(() => {
  const $ = (sel) => document.querySelector(sel);

  const API_BASE = localStorage.getItem("CODLE_API") || (window.CODLE_API_BASE || "http://localhost:8000");

  // --- Time helpers (Europe/Paris) ---
  function parisTodayISO() {
    // Get YYYY-MM-DD for Europe/Paris
    const nowParis = new Date(
      new Date().toLocaleString("en-CA", { timeZone: "Europe/Paris" })
    );
    // toLocaleString trick yields a string, then Date() will parse local time; better to format manually:
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: "Europe/Paris",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(new Date());
    const y = parts.find(p => p.type === "year").value;
    const m = parts.find(p => p.type === "month").value;
    const d = parts.find(p => p.type === "day").value;
    return `${y}-${m}-${d}`;
  }

  // --- Theme toggle (kept minimal) ---
  function initThemeToggle() {
    const pref = localStorage.getItem("codle-theme") || "light";
    document.documentElement.setAttribute("data-theme", pref);
    const btn = $("#themeToggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("codle-theme", next);
      if (window.editor) {
        window.editor.setOption("theme", next === "dark" ? "material-darker" : "default");
      }
    });
  }

  // --- CodeMirror init ---
  function initEditor() {
    const textarea = $("#codeEditor");
    if (!textarea) return null;
    const isDark = (document.documentElement.getAttribute("data-theme") || "light") === "dark";
    const editor = CodeMirror.fromTextArea(textarea, {
      lineNumbers: true,
      mode: "python",
      theme: isDark ? "material-darker" : "default",
      matchBrackets: true,
      autoCloseBrackets: true,
      placeholder: "// Write your solution here…",
      indentUnit: 2,
      tabSize: 2,
    });
    window.editor = editor;
    return editor;
  }

  function setStatus(msg) {
    const s = $("#status");
    if (s) s.textContent = msg;
  }

  function difficultyClass(diff) {
    switch ((diff || "").toLowerCase()) {
      case "medium": return "difficulty-medium";
      case "hard": return "difficulty-hard";
      case "extreme": return "difficulty-extreme";
      default: return "difficulty-easy";
    }
  }

  function renderMarkdownBasic(text) {
    // Very small, safe-ish markdown: paragraphs + code blocks + inline code + lists
    if (!text) return "";
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    // code fences ```
    html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
    // inline code `code`
    html = html.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
    // headings ###
    html = html.replace(/^###\s*(.*)$/gm, "<h3>$1</h3>");
    html = html.replace(/^##\s*(.*)$/gm, "<h2>$1</h2>");
    html = html.replace(/^#\s*(.*)$/gm, "<h1>$1</h1>");
    // lists
    html = html.replace(/^\s*-\s+(.*)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)(\s*(?!<li>))/gs, "<ul>$1</ul>");
    // paragraphs
    html = html.replace(/^(?!<h\d>|<ul>|<pre>|<li>)(.+)$/gm, "<p>$1</p>");
    return html;
  }

  function populateProblem(problem) {
    const title = $("#problem-title");
    const badge = $("#difficultyBadge");
    const desc = $("#problemDescription");
    const helper = $("#helperText");
    if (title) title.textContent = problem.title || "Untitled";
    if (badge) {
      badge.textContent = (problem.difficulty || "").replace(/^\w/, c => c.toUpperCase());
      badge.className = `badge ${difficultyClass(problem.difficulty)}`;
      badge.setAttribute("aria-label", `Difficulty: ${problem.difficulty}`);
    }
    if (desc) {
      desc.innerHTML = renderMarkdownBasic(problem.description || "");
    }
    if (helper) {
      helper.innerHTML = `Starter for <code>${(problem.language || "python").toUpperCase()}</code>`;
    }

    // Language select
    const langSel = $("#language");
    const lang = (problem.language || "python").toLowerCase();
    if (langSel && Array.from(langSel.options).some(o => o.value === lang)) {
      langSel.value = lang;
    }

    // Editor code
    if (window.editor) {
      window.editor.setOption("mode", lang === "python" ? "python" :
        (lang === "javascript" ? "javascript" : (lang === "java" ? "text/x-java" : "text/x-c++src")));
      window.editor.setValue(problem.starter_code || "");
    } else {
      const ta = $("#codeEditor");
      if (ta) ta.value = problem.starter_code || "";
    }
  }

  function renderHints(hints) {
    const btn = $("#hintNextBtn");
    const container = $("#hintsContent");
    if (!btn || !container) return;

    container.innerHTML = '<p class="hint-shortcuts">Hints reveal one at a time.</p>';
    if (!hints || !Array.isArray(hints) || hints.length === 0) {
      btn.disabled = true;
      return;
    }
    btn.disabled = false;
    hints.forEach((h, i) => {
      const div = document.createElement("div");
      div.className = "hint-item";
      div.dataset.hintIndex = String(i);
      div.hidden = i !== 0;
      div.innerHTML = `<strong>Hint ${i+1}:</strong> ${h}`;
      container.appendChild(div);
    });
    let idx = 0;
    function showNext() {
      const items = container.querySelectorAll(".hint-item");
      if (idx < items.length - 1) {
        items[++idx].hidden = false;
      }
      if (idx >= items.length - 1) btn.disabled = true;
    }
    btn.onclick = showNext;
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        if (!btn.disabled) showNext();
      }
    });
  }

  // --- Buttons, tests, and runner ---
  function initUIButtons() {
    // Run
    const runBtn = $("#runBtn");
    if (runBtn) runBtn.addEventListener("click", runAllTests);

    // Reset
    const resetBtn = $("#resetBtn");
    if (resetBtn) resetBtn.addEventListener("click", resetToStarter);

    // Tests controls
    const addBtn = $("#addTestBtn");
    if (addBtn) addBtn.addEventListener("click", () => addTestCase());

    const prevBtn = $("#prevTestBtn");
    const nextBtn = $("#nextTestBtn");
    if (prevBtn) prevBtn.addEventListener("click", () => goToTest(currentTestIndex - 1));
    if (nextBtn) nextBtn.addEventListener("click", () => goToTest(currentTestIndex + 1));

    // Keyboard: Ctrl/Cmd + Enter runs all
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        runAllTests();
      }
      // Theme: Ctrl/Cmd + B
      if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase?.() === "b")) {
        const btn = $("#themeToggle");
        if (btn) btn.click();
      }
    });

    // Start with a single empty test if none
    if (!document.querySelector(".test-case")) addTestCase();
    updateTestCounter();
  }

  let currentTestIndex = 0;
  function addTestCase(initial = {input:"", expected:""}) {
    const container = $("#testCases");
    if (!container) return;
    const idx = container.querySelectorAll(".test-case").length;
    const wrapper = document.createElement("div");
    wrapper.className = "test-case";
    wrapper.dataset.index = String(idx);
    wrapper.innerHTML = `
      <div class="test-case-header">
        <strong>Test #${idx+1}</strong>
        <div>
          <button class="btn-icon" title="Delete" aria-label="Delete test">✕</button>
        </div>
      </div>
      <div class="test-inputs">
        <div class="input-group">
          <label>Input (JSON)</label>
          <textarea class="test-input" rows="3" placeholder='e.g. [2,3] or {"a":1,"b":2}'>${initial.input ?? ""}</textarea>
        </div>
        <div class="input-group">
          <label>Expected (JSON)</label>
          <textarea class="test-output" rows="3" placeholder="e.g. 5">${initial.expected ?? ""}</textarea>
        </div>
      </div>
    `;
    const delBtn = wrapper.querySelector(".btn-icon");
    delBtn?.addEventListener("click", () => {
      wrapper.remove();
      const items = [...container.querySelectorAll(".test-case")];
      items.forEach((el,i) => {
        el.dataset.index = String(i);
        el.querySelector(".test-case-header strong").textContent = `Test #${i+1}`;
      });
      if (currentTestIndex >= items.length) currentTestIndex = Math.max(0, items.length-1);
      updateTestCounter();
      goToTest(currentTestIndex);
    });
    container.appendChild(wrapper);
    goToTest(idx);
    updateTestCounter();
  }

  function goToTest(index) {
    const tests = document.querySelectorAll(".test-case");
    if (tests.length === 0) return;
    currentTestIndex = Math.max(0, Math.min(index, tests.length-1));
    tests.forEach((el,i) => {
      el.style.outline = i === currentTestIndex ? "2px solid var(--primary)" : "none";
    });
    updateTestCounter();
    tests[currentTestIndex].scrollIntoView({behavior:"smooth", block:"nearest"});
  }

  function updateTestCounter() {
    const counter = $("#testCounter");
    const total = document.querySelectorAll(".test-case").length || 1;
    if (counter) counter.textContent = `Test ${total ? currentTestIndex+1 : 0} of ${total}`;
  }

  function collectTests() {
    const nodes = document.querySelectorAll(".test-case");
    return [...nodes].map(node => {
      const [inputEl, expectedEl] = node.querySelectorAll("textarea");
      return {
        inputRaw: inputEl?.value ?? "",
        expectedRaw: expectedEl?.value ?? ""
      };
    });
  }

  // Store starter code/language for Reset
  let STARTER = { code:"", language:"python" };

  // Extend populateProblem to remember starter values
  const _populateProblem = populateProblem;
  populateProblem = function(problem) {
    STARTER.code = problem.starter_code || "";
    STARTER.language = (problem.language || "python").toLowerCase();
    _populateProblem(problem);
  };

  function resetToStarter() {
    if (window.editor) {
      window.editor.setValue(STARTER.code);
      window.editor.focus();
    } else {
      const ta = $("#codeEditor");
      if (ta) ta.value = STARTER.code;
    }
    setStatus("Reset to starter template.");
  }

  function renderResultRow(label, ok, details = "") {
    const div = document.createElement("div");
    div.className = "result-item";
    div.innerHTML = `
      <div class="result-label">${label}</div>
      <div class="pill ${ok ? "ok" : "bad"}">${ok ? "PASS" : "FAIL"}</div>
      ${details ? `<div class="stacktrace">${details}</div>` : ""}
    `;
    return div;
  }

  function safeJSON(str) {
    if (!str?.trim()) return undefined;
    try { return JSON.parse(str); } catch { return {__PARSE_ERROR__: true}; }
  }

  function deepEqual(a,b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  function runAllTests() {
    const results = $("#results");
    if (!results) return;
    results.innerHTML = "";

    const languageSel = $("#language");
    const language = (languageSel?.value || STARTER.language || "python").toLowerCase();
    const code = window.editor ? window.editor.getValue() : ($("#codeEditor")?.value ?? "");

    const tests = collectTests();
    if (tests.length === 0) {
      results.appendChild(renderResultRow("No tests to run", false, "Add a test first."));
      return;
    }

    if (language !== "javascript") {
      results.appendChild(renderResultRow("Unsupported language", false, "Only JavaScript execution is supported in-browser."));
      setStatus("Run finished (JS only).");
      return;
    }

    let solutionFn = null, buildErr = null;
    try {
      const factory = new Function(`${code}; return (typeof solution === "function") ? solution : null;`);
      solutionFn = factory();
      if (!solutionFn) buildErr = "Define a function named `solution` to be tested.";
    } catch (e) {
      buildErr = String(e && e.stack || e);
    }
    if (buildErr) {
      results.appendChild(renderResultRow("Build error", false, buildErr));
      setStatus("Run finished with errors.");
      return;
    }

    let passCount = 0;
    tests.forEach((t, i) => {
      const input = safeJSON(t.inputRaw);
      const expected = safeJSON(t.expectedRaw);
      if (input && input.__PARSE_ERROR__) {
        results.appendChild(renderResultRow(`Test #${i+1}`, false, "Could not parse Input JSON."));
        return;
      }
      if (expected && expected.__PARSE_ERROR__) {
        results.appendChild(renderResultRow(`Test #${i+1}`, false, "Could not parse Expected JSON."));
        return;
      }
      let actual, err;
      try {
        const args = Array.isArray(input) ? input : [input];
        actual = solutionFn.apply(null, args);
      } catch (e) {
        err = String(e && e.stack || e);
      }
      if (err) {
        results.appendChild(renderResultRow(`Test #${i+1}`, false, err));
      } else {
        const ok = deepEqual(actual, expected);
        if (ok) passCount++;
        const detail = ok ? "" : `Expected: ${JSON.stringify(expected)}\nActual: ${JSON.stringify(actual)}`;
        results.appendChild(renderResultRow(`Test #${i+1}`, ok, detail));
      }
    });

    const summary = document.createElement("div");
    summary.className = "result-item";
    summary.innerHTML = `<div class="result-label"><strong>Summary</strong></div><div class="pill ${passCount===tests.length?"ok":"bad"}">${passCount}/${tests.length}</div>`;
    results.prepend(summary);
    setStatus("Run complete.");
  }

  async function fetchJSON(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      err.body = await res.text().catch(() => "");
      throw err;
    }
    return res.json();
  }

  async function loadTodaysProblem() {
    setStatus("Loading today’s problem…");
    const date = parisTodayISO();
    try {
      // Try to GET from DB
      const problem = await fetchJSON(`${API_BASE}/problems/${date}`);
      populateProblem(problem);
      renderHints(null); // No hints on GET route
      setStatus(`Loaded problem for ${date}.`);
      return;
    } catch (e) {
      if (e.status !== 404) {
        console.error(e);
        showError(`Could not load today’s problem (GET): ${e.status || ""} ${e.body || ""}`);
        return;
      }
    }

    // If not found, generate one (POST /problems)
    try {
      setStatus("No problem stored for today — generating…");
      const difficulty = localStorage.getItem("codle-difficulty") || "medium";
      const payload = { difficulty };
      const data = await fetchJSON(`${API_BASE}/problems`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      populateProblem(data.problem);
      renderHints(data.hints);
      setStatus("Generated and saved today’s problem.");
    } catch (e) {
      console.error(e);
      showError(`Could not generate problem (POST): ${e.status || ""} ${e.body || ""}`);
    }
  }

  function showError(msg) {
    const desc = $("#problemDescription");
    if (desc) {
      desc.innerHTML = `<div class="error-box"><strong>Oops.</strong> ${msg}</div>`;
    }
    setStatus("Error.");
  }

  // Boot
  document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initEditor();
    initUIButtons();
    loadTodaysProblem();
  });
})();
