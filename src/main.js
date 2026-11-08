import { FaceApp } from './app.js';
import { skinParams, eyeParams, IRIS_PRESETS } from './materials.js';
import { PRESETS } from './lighting.js';
import { EXPRESSIONS, EXPRESSION_LABELS } from './animator.js';

const $ = (s) => document.querySelector(s);
const canvas = $('#view');
const app = new FaceApp(canvas, './assets/');
window.faceApp = app; // handy for the console

// ---------------------------------------------------------------- builders
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

function group(title, hint, ...kids) {
  return el('section', { class: 'group' }, el('h2', {}, title, hint ? el('small', {}, hint) : null), ...kids);
}

function slider(id, label, { min, max, step, value, fmt = (v) => v.toFixed(2), onInput }) {
  const out = el('output', { for: id });
  const input = el('input', { type: 'range', id, min, max, step, value });
  const paint = () => {
    const v = parseFloat(input.value);
    input.style.setProperty('--p', `${((v - min) / (max - min)) * 100}%`);
    out.textContent = fmt(v);
  };
  input.addEventListener('input', () => { paint(); onInput(parseFloat(input.value)); });
  paint();
  const row = el('div', { class: 'row' }, el('label', { for: id }, label), out, input);
  row.set = (v) => { input.value = v; paint(); };
  return row;
}

function chips(name, items, current, onPick, swatch) {
  const wrap = el('div', { class: 'chips', role: 'group', 'aria-label': name });
  const buttons = items.map(([key, label]) => {
    const b = el('button', { class: 'chip', type: 'button', 'aria-pressed': String(key === current) },
      swatch ? el('span', { class: 'sw', style: `background:${swatch(key)}` }) : null, label);
    b.addEventListener('click', () => {
      buttons.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      onPick(key);
    });
    return b;
  });
  wrap.append(...buttons);
  return wrap;
}

function toggle(id, label, checked, onChange) {
  const input = el('input', { type: 'checkbox', id, role: 'switch' });
  input.checked = checked;
  input.addEventListener('change', () => onChange(input.checked));
  return el('label', { class: 'toggle', for: id }, label, input);
}

const linToCss = (rgb) => '#' + rgb.map((c) => {
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.round(Math.min(1, s * 1.6) * 255).toString(16).padStart(2, '0');
}).join('');

// ---------------------------------------------------------------- panels
function buildPanels() {
  const panels = $('#panels');
  const ui = {};

  // FACE ------------------------------------------------------------------
  let variation = 1.0;
  const modeSliders = [];
  const pFace = el('div', { class: 'panel', id: 'p-face', role: 'tabpanel', 'aria-labelledby': 'tab-face' },
    group('Identity', '100 scans → 40 PCA modes',
      el('p', { class: 'note' }, 'Each face is sampled from a statistical shape model built from light-stage scans, so it belongs to nobody.'),
      el('div', { class: 'actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { const w = app.randomIdentity(variation); modeSliders.forEach((s, i) => s.set(w[i])); } }, 'New face'),
        el('button', { class: 'btn ghost', type: 'button', onclick: () => { app.setIdentity([]); modeSliders.forEach((s) => s.set(0)); } }, 'Average face')),
      slider('variation', 'Distinctiveness', { min: 0, max: 2, step: 0.05, value: 1, fmt: (v) => `${v.toFixed(2)} σ`, onInput: (v) => { variation = v; } })),
    group('Principal shape modes', 'σ from the mean',
      ...Array.from({ length: 6 }, (_, i) => {
        const s = slider(`mode${i}`, `Mode ${i + 1}`, { min: -3, max: 3, step: 0.05, value: 0, fmt: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`, onInput: (v) => {
          const w = Array.from(app.model.identityWeights); w[i] = v; app.setIdentity(w);
        } });
        modeSliders.push(s);
        return s;
      })),
  );

  // EXPRESSION ------------------------------------------------------------
  const exprItems = Object.keys(EXPRESSIONS).map((k) => [k, EXPRESSION_LABELS[k]]);
  const au = (id, label, names, max = 1) => slider(id, label, { min: 0, max, step: 0.01, value: 0, fmt: (v) => v.toFixed(2), onInput: (v) => {
    for (const n of names) app.animator.manual[n] = v;
  } });
  const pExpr = el('div', { class: 'panel', id: 'p-expr', role: 'tabpanel', 'aria-labelledby': 'tab-expr', hidden: '' },
    group('Expression', '53 blend shapes',
      chips('Expression presets', exprItems, 'neutral', (k) => app.animator.setExpression(k)),
      slider('intensity', 'Intensity', { min: 0, max: 1.2, step: 0.01, value: 1, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { app.animator.intensity = v; } })),
    group('Behaviour', null,
      el('div', { class: 'toggles' },
        toggle('talk', 'Talk (babble visemes)', false, (v) => { app.animator.talking = v; }),
        toggle('blink', 'Spontaneous blinking', true, (v) => { app.animator.blinkEnabled = v; }),
        toggle('idle', 'Breathing and micro-motion', true, (v) => { app.animator.idle = v; }),
        toggle('follow', 'Eyes follow the cursor', true, (v) => { app.followCursor = v; }))),
    group('Action units', 'FACS',
      au('au1', 'AU1+2 Brow raiser', ['browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R']),
      au('au4', 'AU4 Brow lowerer', ['browDown_L', 'browDown_R']),
      au('au6', 'AU6 Cheek raiser', ['cheekSquint_L', 'cheekSquint_R']),
      au('au9', 'AU9 Nose wrinkler', ['noseSneer_L', 'noseSneer_R']),
      au('au12', 'AU12 Lip corner puller', ['mouthSmile_L', 'mouthSmile_R']),
      au('au15', 'AU15 Lip corner depressor', ['mouthFrown_L', 'mouthFrown_R']),
      au('au18', 'AU18 Lip pucker', ['mouthPucker']),
      au('au26', 'AU26 Jaw drop', ['jawOpen'], 0.8),
      au('au45', 'AU45 Eyelid closure', ['eyeBlink_L', 'eyeBlink_R'])),
  );

  // SKIN -------------------------------------------------------------------
  const HAIR = {
    black: [0.02, 0.018, 0.016], 'dark brown': [0.06, 0.04, 0.028], brown: [0.13, 0.08, 0.05],
    auburn: [0.2, 0.07, 0.03], blond: [0.42, 0.3, 0.16], gray: [0.3, 0.29, 0.28],
  };
  const setHair = (k) => {
    const [r, g, b] = HAIR[k];
    skinParams.uHairTint.value.setRGB(r, g, b);
    skinParams.uBrowTint.value.setRGB(r * 1.1, g * 1.1, b * 1.1);
  };
  setHair('dark brown');
  const TONES = [['I', 0.07], ['II', 0.13], ['III', 0.22], ['IV', 0.36], ['V', 0.6], ['VI', 0.95]];
  let toneSlider;
  const pSkin = el('div', { class: 'panel', id: 'p-skin', role: 'tabpanel', 'aria-labelledby': 'tab-skin', hidden: '' },
    group('Pigmentation', 'chromophore model',
      chips('Fitzpatrick type', TONES.map(([k]) => [k, `Type ${k}`]), 'II', (k) => {
        const v = TONES.find((t) => t[0] === k)[1]; skinParams.uMelanin.value = v; toneSlider.set(v);
      }),
      (toneSlider = slider('melanin', 'Melanin', { min: 0.02, max: 1.2, step: 0.01, value: skinParams.uMelanin.value, onInput: (v) => { skinParams.uMelanin.value = v; } })),
      slider('hemo', 'Hemoglobin', { min: 0.2, max: 1.2, step: 0.01, value: skinParams.uHemoglobin.value, onInput: (v) => { skinParams.uHemoglobin.value = v; } }),
      slider('flush', 'Flush (blood flow)', { min: 0, max: 1, step: 0.01, value: 0, onInput: (v) => { skinParams.uFlush.value = v; } })),
    group('Surface', null,
      slider('oil', 'Sebum / shine', { min: 0, max: 1.5, step: 0.01, value: skinParams.uOil.value, onInput: (v) => { skinParams.uOil.value = v; } }),
      slider('rough', 'Roughness scale', { min: 0.6, max: 1.4, step: 0.01, value: 1, onInput: (v) => { skinParams.uRoughness.value = v; } }),
      slider('pores', 'Pore depth', { min: 0, max: 2, step: 0.01, value: 1, onInput: (v) => { skinParams.uPoreStrength.value = v; } }),
      slider('age', 'Age lines', { min: 0, max: 1.5, step: 0.01, value: skinParams.uAge.value, onInput: (v) => { skinParams.uAge.value = v; } }),
      slider('sss', 'Scatter radius', { min: 0, max: 2, step: 0.01, value: 1, fmt: (v) => `${v.toFixed(2)}×`, onInput: (v) => { app.pipeline.skin.sssWidth = v; app.pipeline.skin.sssEnabled = v > 0.01; } })),
    group('Hair', null,
      chips('Hair colour', Object.keys(HAIR).map((k) => [k, k[0].toUpperCase() + k.slice(1)]), 'dark brown', setHair, (k) => linToCss(HAIR[k])),
      slider('beard', 'Beard stubble', { min: 0, max: 1, step: 0.01, value: skinParams.uBeard.value, onInput: (v) => { skinParams.uBeard.value = v; } }),
      slider('scalp', 'Buzz cut', { min: 0, max: 1, step: 0.01, value: skinParams.uScalp.value, onInput: (v) => { skinParams.uScalp.value = v; } })),
  );

  // EYES -------------------------------------------------------------------
  const pEyes = el('div', { class: 'panel', id: 'p-eyes', role: 'tabpanel', 'aria-labelledby': 'tab-eyes', hidden: '' },
    group('Iris', 'refracted through the cornea',
      chips('Iris colour', Object.keys(IRIS_PRESETS).map((k) => [k, k[0].toUpperCase() + k.slice(1)]), 'brown', (k) => app.setIris(k), (k) => linToCss(IRIS_PRESETS[k][1])),
      slider('pupil', 'Pupil diameter', { min: 0.18, max: 0.62, step: 0.005, value: eyeParams.uPupil.value, fmt: (v) => `${(v * 11.7).toFixed(1)} mm`, onInput: (v) => { eyeParams.uPupil.value = v; } })),
    group('Sclera', null,
      slider('veins', 'Visible vessels', { min: 0, max: 1.2, step: 0.01, value: eyeParams.uVeins.value, onInput: (v) => { eyeParams.uVeins.value = v; } })),
  );

  // LIGHT ------------------------------------------------------------------
  const lensItems = [[24, '24'], [35, '35'], [50, '50'], [85, '85'], [135, '135']];
  const fItems = [[1.4, '1.4'], [2.8, '2.8'], [4, '4'], [8, '8'], [22, '16+']];
  const pLight = el('div', { class: 'panel', id: 'p-light', role: 'tabpanel', 'aria-labelledby': 'tab-light', hidden: '' },
    group('Lighting rig', null,
      chips('Lighting preset', Object.entries(PRESETS).map(([k, p]) => [k, p.label]), 'studio', (k) => { app.setLighting(k); $('#ro-light').textContent = PRESETS[k].label; }),
      slider('rot', 'Rotate rig', { min: -180, max: 180, step: 1, value: 0, fmt: (v) => `${v.toFixed(0)}°`, onInput: (v) => { app.lighting.rotation = v; app.lighting.update(); } }),
      slider('ev', 'Exposure', { min: -2, max: 2, step: 0.1, value: 0, fmt: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)} EV`, onInput: (v) => { app.setExposure(v); $('#ro-ev').textContent = `${v >= 0 ? '' : ''}${v.toFixed(1)}`; } })),
    group('Lens', 'full-frame equivalent',
      el('div', { class: 'row' }, el('label', {}, 'Focal length (mm)')),
      chips('Focal length', lensItems.map(([k, l]) => [k, l]), 85, (k) => { app.setFocalLength(k); $('#ro-lens').textContent = k; }),
      el('div', { class: 'row' }, el('label', {}, 'Aperture (f-number)')),
      chips('Aperture', fItems, 4, (k) => { app.setAperture(k); $('#ro-f').textContent = k >= 22 ? '16+' : k; }),
      el('div', { class: 'actions' }, el('button', { class: 'btn ghost', type: 'button', onclick: () => app.resetView() }, 'Reset framing'))),
    group('Quality', null,
      slider('scale', 'Render resolution', { min: 0.5, max: 1.5, step: 0.05, value: 1, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => app.setRenderScale(v) })),
    el('p', { class: 'credits' },
      'Shape and expression data: ', el('a', { href: 'https://github.com/ICT-VGL/ICT-FaceKit', target: '_blank', rel: 'noopener' }, 'ICT-FaceKit'),
      ' (USC ICT, MIT). Skin, eyes and lighting are procedural.'),
  );

  panels.append(pFace, pExpr, pSkin, pEyes, pLight);

  // tabs
  const tabs = [...document.querySelectorAll('.tabs [role=tab]')];
  tabs.forEach((t) => t.addEventListener('click', () => {
    tabs.forEach((x) => {
      const on = x === t;
      x.setAttribute('aria-selected', String(on));
      $('#' + x.getAttribute('aria-controls')).hidden = !on;
    });
  }));
  return ui;
}

// ---------------------------------------------------------------- boot
const msg = $('#loading-msg');
app.init((m) => { msg.textContent = m; }).then(() => {
  buildPanels();
  app.setAperture(4);
  app.setIris('brown');
  // start from a pleasant sample identity
  app.randomIdentity(0.9, 20241108);
  let acc = 0, n = 0;
  app.listeners.add((dt) => {
    acc += dt; n++;
    if (acc > 0.5) { $('#ro-fps').textContent = Math.round(n / acc); acc = 0; n = 0; }
  });
  requestAnimationFrame(() => $('#loading').classList.add('done'));
  document.body.dataset.ready = '1';
}).catch((err) => {
  console.error(err);
  msg.textContent = 'The renderer could not start.';
  const e = $('#loading-err');
  e.hidden = false;
  e.textContent = /webgl/i.test(String(err)) ? 'This browser has WebGL 2 turned off or unavailable. Try a recent Chrome, Edge, Firefox or Safari.' : String(err.message || err);
});
