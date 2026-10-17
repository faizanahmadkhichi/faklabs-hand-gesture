# ✋ FAKLABS Hand Gesture Recognition

Free real-time AI hand gesture detection in your browser. **12 gestures**, no signup, runs 100% locally — your camera feed never leaves your device.

**Live demo:** _(deploy to Cloudflare Pages / Netlify / Vercel)_

## Features

- 🎥 Real-time hand tracking via [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (Google's open-source ML, WebGL-accelerated)
- 🤟 **12 gestures**: Open Hand, Fist, Thumbs Up, Thumbs Down, Pointing, Peace, Rock On, Call Me, OK, Three, Four, Pinky
- 📊 Confidence score per detection
- 🕘 Gesture history (last 8)
- 👐 2-hand tracking with color-coded skeletons
- 🪞 Mirror mode toggle
- 📸 Screenshot capture
- ⚡ FPS meter

## How it works

MediaPipe's Hand Landmarker detects 21 hand landmarks per hand in real time. Our original classifier (`classify()` in `app.js`) analyzes finger extension, thumb direction, and fingertip distances to identify gestures — all written from scratch for FAKLABS.

## Run locally

```bash
# serve over HTTP (camera needs a secure context: localhost or HTTPS)
npx serve .
# or
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Deploy

Static site — drop the three files (`index.html`, `app.js`, `styles.css`) on any static host:
Cloudflare Pages, Netlify, Vercel, GitHub Pages.

## Privacy

No video is uploaded. The MediaPipe model downloads once from CDN, then everything runs offline in your browser.

## License

MIT — built with ⚡ by [FAKLABS](https://faizankhichi.me) (Faizan Ahmad Khichi).
MediaPipe Hands is © Google, Apache 2.0.
