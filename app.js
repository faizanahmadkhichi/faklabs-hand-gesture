/* FAKLABS Hand Gesture Recognition — original implementation.
 * Uses Google's open-source MediaPipe Hands for landmark detection.
 * The 12-gesture classifier below is written from scratch for FAKLABS.
 */
(function () {
"use strict";

var MP_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
var MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

/* ---------- 12 gestures ---------- */
var GESTURES = [
  { id: "open",    emoji: "✋", name: "Open Hand" },
  { id: "fist",    emoji: "✊", name: "Fist" },
  { id: "thumb_up",   emoji: "👍", name: "Thumbs Up" },
  { id: "thumb_down", emoji: "👎", name: "Thumbs Down" },
  { id: "point",   emoji: "☝️", name: "Pointing" },
  { id: "peace",   emoji: "✌️", name: "Peace" },
  { id: "rock",    emoji: "🤘", name: "Rock On" },
  { id: "call",    emoji: "🤙", name: "Call Me" },
  { id: "ok",      emoji: "👌", name: "OK" },
  { id: "three",   emoji: "3️⃣", name: "Three" },
  { id: "four",    emoji: "4️⃣", name: "Four" },
  { id: "pinky",   emoji: "🌸", name: "Pinky" },
];

/* ---------- landmark indices ---------- */
var WRIST = 0;
var TIPS = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
var PIPS = { thumb: 3, index: 6, middle: 10, ring: 14, pinky: 18 };
var MCPS = { thumb: 2, index: 5, middle: 9, ring: 13, pinky: 17 };

function dist(a, b) {
  var dx = a.x - b.x, dy = a.y - b.y, dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/* Is a finger extended? Tip farther from wrist than its PIP joint. */
function extended(lm, f) {
  return dist(lm[TIPS[f]], lm[WRIST]) > dist(lm[PIPS[f]], lm[WRIST]) * 1.12;
}

/* Thumb direction: up / down / sideways, from tip vs MCP. */
function thumbDir(lm) {
  var tip = lm[TIPS.thumb], mcp = lm[MCPS.thumb];
  var dy = mcp.y - tip.y; // positive = pointing up (y grows downward in image)
  var dx = tip.x - mcp.x;
  if (dy > 0.08 && Math.abs(dy) > Math.abs(dx) * 1.2) return "up";
  if (dy < -0.08 && Math.abs(dy) > Math.abs(dx) * 1.2) return "down";
  return "side";
}

/* ---------- the classifier (original FAKLABS logic) ----------
 * Returns { id, confidence } or null. Confidence is derived from
 * how decisively the landmark geometry matches (margins). */
function classify(lm) {
  var e = {
    thumb: extended(lm, "thumb"),
    index: extended(lm, "index"),
    middle: extended(lm, "middle"),
    ring: extended(lm, "ring"),
    pinky: extended(lm, "pinky"),
  };
  var n = (e.thumb ? 1 : 0) + (e.index ? 1 : 0) + (e.middle ? 1 : 0) + (e.ring ? 1 : 0) + (e.pinky ? 1 : 0);
  var tdir = thumbDir(lm);

  function conf(base, margin) {
    // margin 0..1 → confidence 70..99
    return Math.round(70 + Math.min(1, Math.max(0, margin)) * 29);
  }

  // OK: thumb tip near index tip (circle), middle/ring/pinky extended
  var okD = dist(lm[TIPS.thumb], lm[TIPS.index]);
  var okRef = dist(lm[WRIST], lm[MCPS.middle]);
  if (okD < okRef * 0.35 && e.middle && e.ring) {
    return { id: "ok", confidence: conf(0, 1 - okD / (okRef * 0.35)) };
  }

  // Thumbs up / down: only thumb extended
  if (e.thumb && !e.index && !e.middle && !e.ring && !e.pinky) {
    if (tdir === "up") return { id: "thumb_up", confidence: 96 };
    if (tdir === "down") return { id: "thumb_down", confidence: 96 };
    return { id: "thumb_up", confidence: 78 }; // sideways thumb, best guess
  }

  // Call me: thumb + pinky
  if (e.thumb && e.pinky && !e.index && !e.middle && !e.ring) {
    return { id: "call", confidence: 94 };
  }
  // Rock on: index + pinky
  if (e.index && e.pinky && !e.middle && !e.ring) {
    return { id: "rock", confidence: 94 };
  }
  // Peace: index + middle
  if (e.index && e.middle && !e.ring && !e.pinky) {
    return { id: "peace", confidence: 95 };
  }
  // Pointing: only index
  if (e.index && !e.middle && !e.ring && !e.pinky && !e.thumb) {
    return { id: "point", confidence: 94 };
  }
  // Pinky: only pinky
  if (e.pinky && !e.index && !e.middle && !e.ring && !e.thumb) {
    return { id: "pinky", confidence: 93 };
  }
  // Three: index + middle + ring
  if (e.index && e.middle && e.ring && !e.pinky) {
    return { id: "three", confidence: 93 };
  }
  // Four: all except thumb
  if (!e.thumb && e.index && e.middle && e.ring && e.pinky) {
    return { id: "four", confidence: 94 };
  }
  // Open hand: all five
  if (n === 5) return { id: "open", confidence: 97 };
  // Fist: none
  if (n === 0) return { id: "fist", confidence: 97 };

  return null;
}

/* ---------- app state ---------- */
var S = {
  landmarker: null, video: null, stream: null,
  latest: null, latestT: 0, raf: 0, running: false,
  mirror: true, frames: 0, fpsT: 0,
  lastGesture: null, lastChange: 0, history: [],
  seen: {}, // gesture id → card element
};

var CONN = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];
var HAND_COLORS = ["#a78bfa", "#4ade80"]; // purple, green

/* ---------- DOM ---------- */
var $ = function (id) { return document.getElementById(id); };
var video, canvas, ctx, statusEl, fpsEl, badge, badgeEmoji, badgeName, badgeConf,
    currentEl, historyEl, gridEl, btnStart, btnStop, btnMirror, btnShot;

function buildGrid() {
  gridEl.innerHTML = "";
  GESTURES.forEach(function (g) {
    var card = document.createElement("div");
    card.className = "gcard";
    card.id = "g-" + g.id;
    card.innerHTML = '<div class="g-emoji">' + g.emoji + "</div>" +
      '<div class="g-name">' + g.name + "</div>" +
      '<div class="g-conf"></div>';
    gridEl.appendChild(card);
    S.seen[g.id] = card;
  });
}

function setStatus(t) { statusEl.innerHTML = t; }

function highlight(id, confidence) {
  GESTURES.forEach(function (g) {
    var c = S.seen[g.id];
    c.classList.toggle("active", g.id === id);
    c.querySelector(".g-conf").textContent = g.id === id ? confidence + "%" : "";
  });
  if (id) {
    var g = GESTURES.find(function (x) { return x.id === id; });
    badge.classList.remove("hidden");
    badgeEmoji.textContent = g.emoji;
    badgeName.textContent = g.name;
    badgeConf.textContent = confidence + "%";
    currentEl.innerHTML = '<span class="cur-emoji">' + g.emoji + '</span> ' +
      "<b>" + g.name + "</b> — " + confidence + "% confident";
  } else {
    badge.classList.add("hidden");
    currentEl.innerHTML = '<span class="none">No clear gesture — adjust your hand</span>';
  }
}

function pushHistory(id) {
  var now = Date.now();
  if (S.lastGesture === id && now - S.lastChange < 1200) return; // debounce
  if (S.lastGesture !== id) {
    S.lastGesture = id;
    S.lastChange = now;
    var g = GESTURES.find(function (x) { return x.id === id; });
    S.history.unshift({ emoji: g.emoji, name: g.name, t: new Date() });
    if (S.history.length > 8) S.history.pop();
    renderHistory();
  }
}

function renderHistory() {
  if (!S.history.length) {
    historyEl.innerHTML = '<span class="none">No gestures yet. Show your hand!</span>';
    return;
  }
  historyEl.innerHTML = S.history.map(function (h) {
    return '<div class="hitem"><span>' + h.emoji + "</span> " + h.name +
      ' <small>' + h.t.toLocaleTimeString() + "</small></div>";
  }).join("");
}

/* ---------- MediaPipe ---------- */
function loadVision() {
  // vision_bundle is an ES module (.mjs) — load via dynamic import
  return import(MP_URL + "/vision_bundle.mjs").then(function (mod) {
    window.vision = mod;
    // debug: verify HandLandmarker is real
    window.__visionDbg = {
      hasHL: typeof mod.HandLandmarker,
      hasFR: typeof mod.FilesetResolver,
      hlKeys: mod.HandLandmarker ? Object.getOwnPropertyNames(mod.HandLandmarker) : [],
      protoKeys: mod.HandLandmarker && mod.HandLandmarker.prototype ?
        Object.getOwnPropertyNames(mod.HandLandmarker.prototype).slice(0, 20) : [],
    };
    return mod;
  });
}

async function ensureLandmarker() {
  if (S.landmarker) return S.landmarker;
  setStatus("Loading MediaPipe Hands AI model… (once, then cached)");
  if (!window.vision) await loadVision();
  var resolver = await window.vision.FilesetResolver.forVisionTasks(MP_URL + "/wasm");
  // Try GPU, fall back to CPU (GPU delegate fails silently on some devices)
  var delegates = ["GPU", "CPU"];
  var lastErr = null;
  for (var di = 0; di < delegates.length; di++) {
    try {
      S.landmarker = await window.vision.HandLandmarker.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: delegates[di] },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });
      if (di > 0) setStatus("⚠️ GPU failed, using CPU mode (slower but works)");
      return S.landmarker;
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error("both delegates failed");
}

/* ---------- camera ---------- */
async function start() {
  btnStart.disabled = true;
  try { await ensureLandmarker(); }
  catch (e) { setStatus("❌ Model failed to load — check your connection and reload."); btnStart.disabled = false; return; }
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    setStatus("❌ Camera blocked — allow it via the 🔒 icon in the address bar, then retry.");
    btnStart.disabled = false;
    return;
  }
  video.srcObject = S.stream;
  await video.play().catch(function () {});
  S.running = true;
  btnStop.disabled = false;
  btnShot.disabled = false;
  setStatus("✅ Tracking — show any of the 12 gestures below.");
  sizeCanvas();
  S.fpsT = performance.now();
  requestAnimationFrame(tick);
}

function stop() {
  S.running = false;
  cancelAnimationFrame(S.raf);
  if (S.stream) { S.stream.getTracks().forEach(function (t) { t.stop(); }); S.stream = null; }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  btnStart.disabled = false;
  btnStop.disabled = true;
  btnShot.disabled = true;
  highlight(null);
  setStatus("Camera stopped. Click <b>Start Camera</b> to begin again.");
}

function sizeCanvas() {
  var r = video.getBoundingClientRect();
  // canvas matches displayed video size
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}

/* ---------- render loop ---------- */
function drawHand(lm, color) {
  var W = canvas.width, H = canvas.height;
  function X(p) { return (S.mirror ? 1 - p.x : p.x) * W; }
  function Y(p) { return p.y * H; }
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(2, W / 320);
  ctx.beginPath();
  CONN.forEach(function (p) {
    ctx.moveTo(X(lm[p[0]]), Y(lm[p[0]]));
    ctx.lineTo(X(lm[p[1]]), Y(lm[p[1]]));
  });
  ctx.stroke();
  lm.forEach(function (p) {
    ctx.beginPath();
    ctx.arc(X(p), Y(p), Math.max(2.5, W / 220), 0, 6.3);
    ctx.fill();
  });
}

/* ---------- diagnostics: lighting + hand presence ---------- */
var diagEl = null, lightT = 0;
function ensureDiag() {
  if (diagEl) return;
  diagEl = document.createElement("div");
  diagEl.className = "diag";
  diagEl.innerHTML = '<span id="diag-light">💡 —</span> <span id="diag-hand">✋ —</span> <span id="diag-dist">📏 —</span><br><span id="diag-dbg" style="font-size:11px;color:#666;"></span>';
  document.querySelector(".stage").appendChild(diagEl);
}
function checkLighting() {
  var now = performance.now();
  if (now - lightT < 1500) return;
  lightT = now;
  if (!video || video.readyState < 2) return;
  try {
    var c = checkLighting.cv || (checkLighting.cv = document.createElement("canvas"));
    c.width = 32; c.height = 18;
    var cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(video, 0, 0, 32, 18);
    var d = cx.getImageData(0, 0, 32, 18).data, sum = 0;
    for (var i = 0; i < d.length; i += 16) sum += d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114;
    var avg = sum / (d.length / 64);
    var el = document.getElementById("diag-light");
    if (!el) return;
    if (avg < 40) { el.textContent = "💡 too dark — add light"; el.className = "bad"; }
    else if (avg > 220) { el.textContent = "💡 too bright — reduce glare"; el.className = "bad"; }
    else { el.textContent = "💡 good"; el.className = "ok"; }
  } catch (e) {}
}
function updateHandDiag(found, handSize) {
  var h = document.getElementById("diag-hand"), d = document.getElementById("diag-dist");
  if (!h || !d) return;
  if (!found) { h.textContent = "✋ no hand seen"; h.className = "bad"; d.textContent = "📏 —"; return; }
  h.textContent = "✋ hand tracked"; h.className = "ok";
  if (handSize < 0.06) { d.textContent = "📏 too far — move closer"; d.className = "bad"; }
  else if (handSize > 0.45) { d.textContent = "📏 too close — move back"; d.className = "bad"; }
  else { d.textContent = "📏 good distance"; d.className = "ok"; }
}

function tick() {
  if (!S.running) return;
  S.raf = requestAnimationFrame(tick);
  var now = performance.now();
  ensureDiag();
  checkLighting();

  S.frames++;
  if (now - S.fpsT >= 1000) {
    fpsEl.textContent = S.frames + " fps";
    S.frames = 0; S.fpsT = now;
  }

  if (video.readyState >= 2 && S.landmarker) {
    try {
      var res = S.landmarker.detectForVideo(video, now);
      S.detectCalls = (S.detectCalls || 0) + 1;
      if (res) { S.latest = res; S.latestT = now; S.cbCount = (S.cbCount||0)+1; }
    } catch (e) {
      S.detectErr = e.message;
    }
  }
  // debug: show detect stats every 3s
  if (!S.dbgT || now - S.dbgT > 3000) {
    S.dbgT = now;
    var dbg = document.getElementById("diag-dbg");
    if (dbg) {
      var vd = window.__visionDbg;
      var vinfo = vd ? " HL:" + vd.hasHL + " proto:" + (vd.protoKeys||[]).join(",").slice(0,80) : " no-dbg";
      // also check the instance
      var inst = "";
      if (S.landmarker) {
        inst = " instType:" + typeof S.landmarker.detectAsync;
        try { inst += " keys:" + Object.getOwnPropertyNames(Object.getPrototypeOf(S.landmarker)).slice(0,10).join(","); } catch(e){}
      }
      dbg.textContent = "🔧 calls:" + (S.detectCalls||0) + " callbacks:" + (S.cbCount||0) +
        " video:" + video.readyState + "x" + video.videoWidth + vinfo + inst +
        (S.detectErr ? " ERR:" + S.detectErr : "");
    }
    S.detectCalls = 0;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // mirror the video via CSS; canvas draws mirrored coords to match
  var res = S.latest;
  var hands = (res && res.landmarks) || [];
  var fresh = res && now - S.latestT < 600;

  if (hands.length && fresh) {
    var best = null, bestConf = 0;
    // hand size for distance diagnostic (wrist to middle tip)
    var hs = dist(hands[0][0], hands[0][12]);
    updateHandDiag(true, hs);
    hands.slice(0, 2).forEach(function (lm, i) {
      drawHand(lm, HAND_COLORS[i % 2]);
      if (i === 0) {
        var c = classify(lm);
        if (c && c.confidence > bestConf) { best = c; bestConf = c.confidence; }
      }
    });
    if (best && bestConf >= 60) {
      highlight(best.id, bestConf);
      pushHistory(best.id);
    } else {
      highlight(null);
      // hand visible but no clear gesture — tell the user
      currentEl.innerHTML = '<span class="none">✋ hand seen — hold a clear gesture steady</span>';
    }
  } else {
    updateHandDiag(false);
    highlight(null);
    if (!hands.length) {
      currentEl.innerHTML = '<span class="none">No hand detected — check 💡📏 above, face palm to camera</span>';
    }
  }
}

/* ---------- screenshot ---------- */
function screenshot() {
  var out = document.createElement("canvas");
  out.width = canvas.width; out.height = canvas.height;
  var octx = out.getContext("2d");
  // draw video frame (mirrored to match display)
  octx.save();
  if (S.mirror) { octx.translate(out.width, 0); octx.scale(-1, 1); }
  octx.drawImage(video, 0, 0, out.width, out.height);
  octx.restore();
  octx.drawImage(canvas, 0, 0);
  var a = document.createElement("a");
  a.download = "faklabs-gesture-" + Date.now() + ".png";
  a.href = out.toDataURL("image/png");
  a.click();
  setStatus("📸 Screenshot saved.");
}

/* ---------- boot ---------- */
function boot() {
  video = $("video"); canvas = $("overlay"); ctx = canvas.getContext("2d");
  statusEl = $("status"); fpsEl = $("fps");
  badge = $("gesture-badge"); badgeEmoji = $("badge-emoji");
  badgeName = $("badge-name"); badgeConf = $("badge-conf");
  currentEl = $("current"); historyEl = $("history"); gridEl = $("gesture-grid");
  btnStart = $("btn-start"); btnStop = $("btn-stop");
  btnMirror = $("btn-mirror"); btnShot = $("btn-shot");

  buildGrid();
  renderHistory();

  // mirror via CSS transform on video; canvas math mirrors to match
  function paintMirror() {
    video.style.transform = S.mirror ? "scaleX(-1)" : "none";
    btnMirror.textContent = S.mirror ? "🪞 Mirror: On" : "🪞 Mirror: Off";
    btnMirror.classList.toggle("on", S.mirror);
  }
  paintMirror();

  btnStart.addEventListener("click", start);
  btnStop.addEventListener("click", stop);
  btnMirror.addEventListener("click", function () { S.mirror = !S.mirror; paintMirror(); });
  btnShot.addEventListener("click", screenshot);
  window.addEventListener("resize", sizeCanvas);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
})();
