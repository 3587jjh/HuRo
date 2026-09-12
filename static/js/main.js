// ---- theme toggle (persisted) ----
(function () {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("huro-theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) {}
  window.toggleTheme = function () {
    const cur = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = cur === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("huro-theme", next); } catch (e) {}
  };
})();

// ---- additional experiments: expand a card into a full-width panel below its grid row ----
(function () {
  function closeAll() {
    document.querySelectorAll(".exp-card.open").forEach((c) => {
      c.classList.remove("open"); c.querySelector(".exp-toggle").setAttribute("aria-expanded", "false");
      const slot = document.querySelector(".exp-slot"); if (slot) { const panel = slot.firstElementChild; panel.hidden = true; c.appendChild(panel); slot.remove(); }
    });
  }
  window.expToggle = function (btn) {
    const card = btn.closest(".exp-card"); const wasOpen = card.classList.contains("open");
    closeAll(); if (wasOpen) return;
    const grid = card.parentElement; const cards = [...grid.querySelectorAll(".exp-card")];
    const cols = getComputedStyle(grid).gridTemplateColumns.split(" ").length;
    const idx = cards.indexOf(card); const rowEnd = cards[Math.min(cards.length - 1, Math.floor(idx / cols) * cols + cols - 1)];
    const slot = document.createElement("div"); slot.className = "exp-slot";
    const panel = card.querySelector(".exp-panel"); panel.hidden = false; slot.appendChild(panel); rowEnd.after(slot);
    card.classList.add("open"); btn.setAttribute("aria-expanded", "true");
    requestAnimationFrame(() => { const top = card.getBoundingClientRect().top; if (top < 70 || top > window.innerHeight * 0.45) card.scrollIntoView({ behavior: "smooth", block: "start" }); });
  };
  document.addEventListener("DOMContentLoaded", () => {
    const m = (location.hash || "").match(/^#exp(\d)$/); if (!m) return;
    const btn = document.querySelector("#exp" + m[1] + " .exp-toggle"); if (btn) window.expToggle(btn);
  });
  let lastCols = null;
  window.addEventListener("resize", () => {
    const grid = document.querySelector(".exp-grid"); if (!grid) return;
    const cols = getComputedStyle(grid).gridTemplateColumns.split(" ").length;
    if (lastCols !== null && cols !== lastCols) { const open = document.querySelector(".exp-card.open"); if (open) { const b = open.querySelector(".exp-toggle"); closeAll(); window.expToggle(b); } }
    lastCols = cols;
  });
})();

// ---- videos: play when visible, pause off-screen; controls only if autoplay is refused ----
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const vids = [...document.querySelectorAll("video")];
    const visible = new WeakSet();
    const tryPlay = (v) => {
      v.muted = true;
      const p = v.play();
      // AbortError (play interrupted by pause) is normal while scrolling; only a policy refusal needs controls
      if (p && p.catch) p.catch((err) => { if (err && err.name === "NotAllowedError") v.controls = true; });
    };
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((ents) => ents.forEach((en) => {
        if (en.isIntersecting) { visible.add(en.target); tryPlay(en.target); }
        else { visible.delete(en.target); if (!en.target.paused) en.target.pause(); }
      }), { threshold: 0.2 });
      vids.forEach((v) => io.observe(v));
    } else { vids.forEach(tryPlay); }
    // a tap counts as a user gesture: retry any visible video that is still paused
    const kick = () => vids.forEach((v) => { if (v.paused && visible.has(v)) tryPlay(v); });
    ["touchstart", "click"].forEach((ev) => document.addEventListener(ev, kick, { passive: true }));
  });
})();
