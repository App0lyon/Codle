/* Minimal front-end glue to fetch today's problem from the backend and hydrate the UI.
   - Uses Europe/Paris timezone for "today"
   - GET /problems/{date}; if 404, attempts POST /problems with difficulty from localStorage or default 'medium'
   - Populates: title, difficulty badge, description, CodeMirror starter code, language select
   - Optionally renders hints if returned by POST endpoint
*/

(() => {
  const $ = (sel) => document.querySelector(sel);

  // === Language Runtimes ===

  // --- Pyodide (Python) ---
  let __pyodidePromise = null;
  async function ensurePyodide() {
    if (!__pyodidePromise) {
      if (typeof loadPyodide !== "function") {
        throw new Error("Pyodide script not loaded.");
      }
      __pyodidePromise = loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/" });
    }
    return await __pyodidePromise;
  }

  // --- Judge0 client (Java/C++) ---
  const JUDGE0_BASE = "https://ce.judge0.com";
  let __judgeLangIds = null;

  async function judge0GetLanguages() {
    if (__judgeLangIds) return __judgeLangIds;
    const res = await fetch(`${JUDGE0_BASE}/languages/`);
    if (!res.ok) throw new Error("Failed to fetch Judge0 languages.");
    const list = await res.json();
    const byName = Object.fromEntries(list.map(x => [x.name, x.id]));
    function pickId(prefix) {
      const candidates = Object.entries(byName).filter(([name]) => name.startsWith(prefix));
      if (candidates.length === 0) return null;
      candidates.sort((a, b) => a[0] < b[0] ? 1 : -1);
      return candidates[0][1];
    }
    __judgeLangIds = {
      java: pickId("Java ("),
      cpp: pickId("C++ ("),
      javascript: pickId("JavaScript ("),
      python: pickId("Python (3")
    };
    return __judgeLangIds;
  }

  function b64(s) { return btoa(unescape(encodeURIComponent(s))); }

  async function judge0Run({ languageKey, source, stdin, expected }) {
    const langs = await judge0GetLanguages();
    const language_id = langs[languageKey];
    if (!language_id) throw new Error(`Judge0 language not found for ${languageKey}`);
    const url = `${JUDGE0_BASE}/submissions?base64_encoded=true&wait=true`;
    const payload = {
      language_id,
      source_code: b64(source),
      stdin: stdin != null ? b64(stdin) : undefined,
      expected_output: expected != null ? b64(expected) : undefined
    };
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const t = await res.text().catch(()=>"");
      throw new Error(`Judge0 POST failed: ${res.status} ${t}`);
    }
    const data = await res.json();
    return data;
  }

  const API_BASE = "http://localhost:8000";

  // --- Time helpers (Europe/Paris) ---
  function parisTodayISO() {
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

  // ==============================================================
  // === 🌙 THEME SYSTEM (UPDATED FOR ALL PAGES) ==================
  // ==============================================================

  function getCurrentTheme() {
    const html = document.documentElement;
    return html.getAttribute("data-theme") || localStorage.getItem("codle-theme") || "light";
  }

  function applyTheme(theme) {
    const html = document.documentElement;
    html.setAttribute("data-theme", theme);
    localStorage.setItem("codle-theme", theme);

    const btn = document.getElementById("themeToggle");
    if (btn) {
      const isDark = theme === "dark";
      btn.textContent = isDark ? "☀️" : "🌙";
      btn.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");
      btn.title = isDark ? "Light mode" : "Dark mode";
    }

    if (window.editor) {
      window.editor.setOption("theme", theme === "dark" ? "material-darker" : "default");
    }
  }

  function initThemeToggle() {
    const saved = localStorage.getItem("codle-theme");
    const initial = saved || (document.documentElement.getAttribute("data-theme") || "light");
    applyTheme(initial);

    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = getCurrentTheme() === "dark" ? "light" : "dark";
        applyTheme(next);
      });
    }

    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase?.() === "b")) {
        const next = getCurrentTheme() === "dark" ? "light" : "dark";
        applyTheme(next);
      }
    });
  }

  // ==============================================================

  function initEditor() {
    const textarea = $("#codeEditor");
    if (!textarea) return null;
    const isDark = getCurrentTheme() === "dark";
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

  function renderMarkdown(md = "") {
    const raw = marked.parse(md, { gfm: true, breaks: true, headerIds: false, mangle: false });
    return DOMPurify.sanitize(raw);
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
    }
    if (desc) desc.innerHTML = renderMarkdown(problem.description || "");
    if (helper) helper.innerHTML = `Starter for <code>${(problem.language || "python").toUpperCase()}</code>`;

    const langSel = $("#language");
    const lang = (problem.language || "python").toLowerCase();
    if (langSel && Array.from(langSel.options).some(o => o.value === lang)) {
      langSel.value = lang;
    }

    if (window.editor) {
      window.editor.setOption("mode", lang === "python" ? "python" :
        (lang === "javascript" ? "javascript" : (lang === "java" ? "text/x-java" : "text/x-c++src")));
      window.editor.setValue((problem.starter_code || "").replace(/\\n/g, "\n"));
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
      div.hidden = true;
      div.innerHTML = `<strong>Hint ${i + 1}:</strong> ${renderMarkdown(h)}`;
      container.appendChild(div);
    });
    let idx = -1;
    function showNext() {
      const items = container.querySelectorAll(".hint-item");
      if (idx < items.length - 1) items[++idx].hidden = false;
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

  // === Boot ===
  document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initEditor();
    if (typeof initUIButtons === "function") initUIButtons();
    if (typeof loadTodaysProblem === "function") loadTodaysProblem();
  });
})();

// ===== User Menu =====
(() => {
  const btn = document.getElementById('userMenuBtn');
  const menu = document.getElementById('userMenu');
  if (!btn || !menu) return;

  const elProfile = document.getElementById('menuProfile');
  const elSettings = document.getElementById('menuSettings');
  const elAuth = document.getElementById('menuAuth');

  const AUTH_KEY = 'codle_isLoggedIn';
  const isLoggedIn = () => localStorage.getItem(AUTH_KEY) === '1';
  const setAuthLabel = () => { elAuth.textContent = isLoggedIn() ? 'Logout' : 'Login'; };
  setAuthLabel();

  const items = [elProfile, elSettings, elAuth].filter(Boolean);

  const openMenu = () => { menu.classList.add('open'); btn.setAttribute('aria-expanded', 'true'); items[0]?.focus(); };
  const closeMenu = () => { menu.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); btn.focus(); };
  const isOpen = () => menu.classList.contains('open');

  btn.addEventListener('click', (e) => { e.stopPropagation(); isOpen() ? closeMenu() : openMenu(); });
  document.addEventListener('click', (e) => { if (isOpen() && !menu.contains(e.target) && !btn.contains(e.target)) closeMenu(); });
  document.addEventListener('keydown', (e) => {
    if (!isOpen()) return;
    if (e.key === 'Escape') { e.preventDefault(); closeMenu(); return; }
  });

  elProfile?.addEventListener('click', () => { alert('Profile clicked'); closeMenu(); });
  elSettings?.addEventListener('click', () => { alert('Settings clicked'); closeMenu(); });
  elAuth?.addEventListener('click', () => {
    if (isLoggedIn()) localStorage.removeItem(AUTH_KEY);
    else localStorage.setItem(AUTH_KEY, '1');
    setAuthLabel();
    closeMenu();
  });
})();
