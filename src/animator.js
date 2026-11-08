// Procedural "aliveness": blinks, saccades, breathing, idle head motion,
// expression presets with smooth transitions, and a babble/talk cycle.
import * as THREE from 'three';

export const EXPRESSIONS = {
  neutral: {},
  smile: { mouthSmile_L: 0.72, mouthSmile_R: 0.72, cheekSquint_L: 0.45, cheekSquint_R: 0.45, eyeSquint_L: 0.2, eyeSquint_R: 0.2, mouthDimple_L: 0.2, mouthDimple_R: 0.2 },
  laugh: { mouthSmile_L: 1, mouthSmile_R: 1, jawOpen: 0.32, cheekSquint_L: 0.8, cheekSquint_R: 0.8, eyeSquint_L: 0.45, eyeSquint_R: 0.45, mouthUpperUp_L: 0.3, mouthUpperUp_R: 0.3, browInnerUp_L: 0.15, browInnerUp_R: 0.15 },
  surprise: { browInnerUp_L: 0.9, browInnerUp_R: 0.9, browOuterUp_L: 0.8, browOuterUp_R: 0.8, eyeWide_L: 0.65, eyeWide_R: 0.65, jawOpen: 0.3, mouthFunnel: 0.15 },
  angry: { browDown_L: 0.9, browDown_R: 0.9, eyeSquint_L: 0.4, eyeSquint_R: 0.4, noseSneer_L: 0.35, noseSneer_R: 0.35, mouthPress_L: 0.45, mouthPress_R: 0.45, mouthFrown_L: 0.3, mouthFrown_R: 0.3, jawForward: 0.1 },
  sad: { browInnerUp_L: 0.85, browInnerUp_R: 0.85, mouthFrown_L: 0.6, mouthFrown_R: 0.6, mouthShrugLower: 0.45, eyeSquint_L: 0.1, eyeSquint_R: 0.1, eyeLookDown_L: 0.15, eyeLookDown_R: 0.15 },
  disgust: { noseSneer_L: 0.85, noseSneer_R: 0.85, mouthUpperUp_L: 0.55, mouthUpperUp_R: 0.55, browDown_L: 0.45, browDown_R: 0.45, eyeSquint_L: 0.45, eyeSquint_R: 0.45, mouthFrown_L: 0.2, mouthFrown_R: 0.2 },
  pout: { mouthPucker: 0.9, mouthFunnel: 0.25, browInnerUp_L: 0.2, browInnerUp_R: 0.2 },
  smirk: { mouthSmile_L: 0.65, mouthDimple_L: 0.3, cheekSquint_L: 0.35, eyeSquint_L: 0.15, browOuterUp_R: 0.25, mouthPress_R: 0.15 },
};

export const EXPRESSION_LABELS = {
  neutral: 'Neutral', smile: 'Smile', laugh: 'Laugh', surprise: 'Surprise', angry: 'Anger',
  sad: 'Sadness', disgust: 'Disgust', pout: 'Pout', smirk: 'Smirk',
};

const VISEMES = [
  { jawOpen: 0.28, mouthStretch_L: 0.1, mouthStretch_R: 0.1 },          // a
  { jawOpen: 0.14, mouthSmile_L: 0.2, mouthSmile_R: 0.2, mouthStretch_L: 0.25, mouthStretch_R: 0.25 }, // e / i
  { jawOpen: 0.18, mouthFunnel: 0.55, mouthPucker: 0.25 },              // o
  { jawOpen: 0.06, mouthPucker: 0.75, mouthFunnel: 0.2 },               // u
  { mouthClose: 0.15, mouthPress_L: 0.3, mouthPress_R: 0.3 },           // m b p
  { jawOpen: 0.05, mouthRollLower: 0.45, mouthUpperUp_L: 0.15, mouthUpperUp_R: 0.15 }, // f v
  { jawOpen: 0.1, mouthShrugUpper: 0.2 },                                // rest
];

function noise1(t, seed) {
  // smooth value noise from a few incommensurate sines
  return (Math.sin(t * 1.0 + seed) * 0.5 + Math.sin(t * 2.31 + seed * 1.7) * 0.3 + Math.sin(t * 4.07 + seed * 2.9) * 0.2);
}

export class Animator {
  constructor(model) {
    this.model = model;
    this.names = model.expressions.map((e) => e.name);
    this.current = Object.fromEntries(this.names.map((n) => [n, 0]));
    this.target = { ...this.current };
    this.manual = {};          // slider overrides (added on top)
    this.expression = 'neutral';
    this.intensity = 1;
    this.t = 0;
    this.idle = true;
    this.blinkEnabled = true;
    this.talking = false;
    this.nextBlink = 1.5;
    this.blinkT = -1;
    this.doubleBlink = false;
    this.saccade = new THREE.Vector2();
    this.saccadeTarget = new THREE.Vector2();
    this.nextSaccade = 0.8;
    this.visemeT = 0;
    this.viseme = {};
    this.gazeYaw = 0;
    this.gazePitch = 0;
    this.headRot = new THREE.Euler();
    this.headPos = new THREE.Vector3();
  }

  setExpression(name, intensity = this.intensity) {
    this.expression = name;
    this.intensity = intensity;
  }

  // gazeYaw/Pitch in radians (head-local) supplied by the app each frame.
  update(dt) {
    this.t += dt;
    const t = this.t;
    const tgt = {};
    for (const n of this.names) tgt[n] = 0;
    const e = EXPRESSIONS[this.expression] || {};
    for (const [k, v] of Object.entries(e)) tgt[k] = v * this.intensity;

    // Talking: hop between visemes ~9 per second.
    if (this.talking) {
      this.visemeT -= dt;
      if (this.visemeT <= 0) {
        this.viseme = VISEMES[Math.floor(Math.random() * VISEMES.length)];
        this.visemeT = 0.07 + Math.random() * 0.12;
        if (Math.random() < 0.08) { this.viseme = VISEMES[6]; this.visemeT = 0.25 + Math.random() * 0.3; }
      }
      for (const [k, v] of Object.entries(this.viseme)) tgt[k] = Math.min(1, (tgt[k] || 0) + v);
      tgt.browInnerUp_L += 0.15 * Math.max(0, noise1(t * 1.5, 3));
      tgt.browInnerUp_R += 0.15 * Math.max(0, noise1(t * 1.5, 3));
    }

    // Blinks (closing ~70 ms, opening ~150 ms), occasionally doubled.
    let blink = 0;
    if (this.blinkEnabled) {
      this.nextBlink -= dt;
      if (this.nextBlink <= 0 && this.blinkT < 0) {
        this.blinkT = 0;
        this.doubleBlink = Math.random() < 0.12;
      }
      if (this.blinkT >= 0) {
        this.blinkT += dt;
        const c = 0.075, h = 0.03, o = 0.16;
        const b = this.blinkT;
        if (b < c) blink = THREE.MathUtils.smootherstep(b / c, 0, 1);
        else if (b < c + h) blink = 1;
        else if (b < c + h + o) blink = 1 - THREE.MathUtils.smoothstep((b - c - h) / o, 0, 1);
        else {
          this.blinkT = -1;
          this.nextBlink = this.doubleBlink ? 0.12 : 1.8 + Math.random() * 4.5;
          this.doubleBlink = false;
        }
      }
    }

    // Saccades: small fast re-fixations around the gaze target.
    if (this.idle) {
      this.nextSaccade -= dt;
      if (this.nextSaccade <= 0) {
        const big = Math.random() < 0.25;
        this.saccadeTarget.set((Math.random() * 2 - 1) * (big ? 0.12 : 0.035), (Math.random() * 2 - 1) * (big ? 0.06 : 0.02));
        this.nextSaccade = 0.4 + Math.random() * 2.2;
      }
    } else this.saccadeTarget.set(0, 0);
    this.saccade.lerp(this.saccadeTarget, 1 - Math.exp(-dt * 40));

    // Idle head motion and breathing.
    if (this.idle) {
      this.headRot.set(
        THREE.MathUtils.degToRad(1.0 * noise1(t * 0.23, 5) + 0.35 * Math.sin(t * 2 * Math.PI / 4.4)),
        THREE.MathUtils.degToRad(1.6 * noise1(t * 0.17, 1)),
        THREE.MathUtils.degToRad(0.7 * noise1(t * 0.19, 9)),
      );
      this.headPos.set(0, 0.0007 * Math.sin(t * 2 * Math.PI / 4.4), 0);
    } else {
      this.headRot.set(0, 0, 0);
      this.headPos.set(0, 0, 0);
    }

    // Eyelids follow the gaze (lid-eye coupling).
    const yaw = this.gazeYaw + this.saccade.x, pitch = this.gazePitch + this.saccade.y;
    const up = THREE.MathUtils.clamp(pitch / 0.4, 0, 1), down = THREE.MathUtils.clamp(-pitch / 0.4, 0, 1);
    tgt.eyeLookUp_L += up; tgt.eyeLookUp_R += up;
    tgt.eyeLookDown_L += down; tgt.eyeLookDown_R += down;
    const left = THREE.MathUtils.clamp(yaw / 0.5, 0, 1), right = THREE.MathUtils.clamp(-yaw / 0.5, 0, 1);
    tgt.eyeLookOut_L += left; tgt.eyeLookIn_R += left;
    tgt.eyeLookIn_L += right; tgt.eyeLookOut_R += right;
    this.eyeYaw = yaw;
    this.eyePitch = pitch;

    // Smooth towards target (blink and gaze bypass smoothing).
    const k = 1 - Math.exp(-dt * (this.talking ? 22 : 9));
    for (const n of this.names) {
      this.current[n] += (tgt[n] - this.current[n]) * k;
      let w = this.current[n] + (this.manual[n] || 0);
      if (n === 'eyeBlink_L' || n === 'eyeBlink_R') w = Math.max(w, blink * (1 - 0.6 * up));
      if (n.startsWith('eyeLook')) w = tgt[n];
      this.model.setExpression(n, THREE.MathUtils.clamp(w, -0.2, 1.2));
    }
  }

  // Wrinkle activation for the shader: forehead, smile/squint, frown/sneer.
  wrinkles(out) {
    const g = (n) => this.model.getExpression(n);
    const avg = (a, b) => 0.5 * (g(a) + g(b));
    out.set(
      Math.max(avg('browInnerUp_L', 'browInnerUp_R'), 0.85 * avg('browOuterUp_L', 'browOuterUp_R')),
      Math.max(avg('cheekSquint_L', 'cheekSquint_R'), 0.8 * avg('eyeSquint_L', 'eyeSquint_R'), 0.75 * avg('mouthSmile_L', 'mouthSmile_R')),
      Math.max(avg('browDown_L', 'browDown_R'), avg('noseSneer_L', 'noseSneer_R'), 0.6 * g('mouthShrugLower')),
    );
    return out;
  }
}
