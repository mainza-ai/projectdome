# Web UI

The browser frontend provides a control panel for interacting with the avatar engine in real time.

## HTML

`web/index.html` — single-page application layout.

**Sections:**
- **Speech Interface** — text input, "Speak" button, play/pause audio, speaking style dropdown (12 VOCASET styles)
- **Affective Overlay** — emotion selector (HAPPY, SURPRISE, SAD, etc.), intensity slider
- **Head Pose & Gaze** — neck yaw/pitch sliders, horizontal/vertical gaze sliders, auto eye-contact checkbox
- **Character Identity** — gender + ethnicity selectors, "Generate Identity" button

## CSS

`web/style.css` — Apple light-mode aesthetic with Google color accents.

- Color palette: light grey background (#f5f5f7), white cards, Google Blue/Red/Yellow/Green accents
- Font: Outfit (Google Fonts)
- Glass-morphism header with rounded pill design
- 2-column responsive control grid (collapses to single column on mobile)
- Custom range slider styling with Google Blue thumb
- Loading spinner with fade-out transition

## Audio Sync

`web/audio_sync.js` — `AudioSync` class wrapping the Web Audio API.

- `loadAudioFromBase64()` — decodes base64 WAV, creates AudioBuffer
- `play()` / `pause()` / `stop()` — playback control with elapsed-time tracking
- `getCurrentTime()` — returns playback position in seconds (used by render loop for viseme sync)
- `onEnded` callback — fires when audio plays to completion

## Animation Controller

`web/animation_controller.js` — `AnimationController` class mirroring the Python animation pipeline in the browser.

- `loadVisemeTable()` — fetches `data/viseme_table.json` from server, falls back to hardcoded defaults
- `getSpeechCoefficients(timeS, timeline)` — samples the viseme timeline at a given time, with 40ms ramp transitions and gap interpolation (mirrors `VisemeInterpolator`)
- `blend(speechCoeffs, emotionCoeffs)` — additive blend (mirrors `EmotionBlender.blend()`)

## File reference

| File | Role |
|---|---|
| `web/index.html` | Application entry point and UI layout |
| `web/style.css` | Apple-inspired premium stylesheet |
| `web/audio_sync.js` | Web Audio API playback with timing |
| `web/animation_controller.js` | Client-side viseme interpolation + emotion blending |
