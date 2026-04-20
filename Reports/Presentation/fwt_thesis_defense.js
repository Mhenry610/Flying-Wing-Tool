"use strict";

const path = require("path");
const fs = require("fs");

const ROOT = __dirname;
const WORK_ROOT = path.join(ROOT, "fwt_thesis_defense_work");

const pptxgen = require(path.join(WORK_ROOT, "node_modules", "pptxgenjs"));
const { autoFontSize } = require(path.join(WORK_ROOT, "pptxgenjs_helpers", "text.js"));
const { imageSizingContain } = require(path.join(WORK_ROOT, "pptxgenjs_helpers", "image.js"));
const { svgToDataUri } = require(path.join(WORK_ROOT, "pptxgenjs_helpers", "svg.js"));
const {
  warnIfSlideElementsOutOfBounds,
  warnIfSlideHasOverlaps,
} = require(path.join(WORK_ROOT, "pptxgenjs_helpers", "layout.js"));

const pptx = new pptxgen();

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;
const COLORS = {
  blue: "004684",
  gold: "FDB927",
  white: "FFFFFF",
  offWhite: "F7F8FA",
  ink: "243447",
  gray: "6B7480",
  line: "D7DCE2",
  paleBlue: "EAF2FB",
  paleGold: "FFF4D6",
  paleGray: "F2F4F7",
  green: "2E7D32",
  red: "B23A48",
};
const FONTS = {
  title: "Palatino Linotype",
  heading: "Century Gothic",
  body: "Palatino Linotype",
};

const OUTPUT = path.join(ROOT, "fwt_thesis_defense.pptx");
const ASSET_ROOT = path.join(ROOT, "fwt_thesis_defense_assets");
const FIGURE_ROOT = path.join(ROOT, "..", "Report", "figures");
const BG_TITLE = path.join(ASSET_ROOT, "title_background.png");
const BG_TRANSITION = path.join(ASSET_ROOT, "transition_background.png");
const BG_CONTENT = path.join(ASSET_ROOT, "content_background.png");

const FIG = {
  analysis: path.join(FIGURE_ROOT, "analysis_tab.png"),
  baramState: path.join(FIGURE_ROOT, "baram_state_overview.png"),
  cad: path.join(FIGURE_ROOT, "cad_isometric.png"),
  export: path.join(FIGURE_ROOT, "export_tab.png"),
  geometry: path.join(FIGURE_ROOT, "geometry_tab.png"),
  baramConv: path.join(FIGURE_ROOT, "intendedvalidation2_baram_convergence.png"),
  baramCompare: path.join(FIGURE_ROOT, "intendedvalidation2_baram_vs_aerosandbox.png"),
  clPolar: path.join(FIGURE_ROOT, "intendedvalidation2_cl_polar.png"),
  cdPolar: path.join(FIGURE_ROOT, "intendedvalidation2_cd_polar.png"),
  dragAttrib: path.join(FIGURE_ROOT, "intendedvalidation2_drag_area_attribution.png"),
  twist: path.join(FIGURE_ROOT, "intendedvalidation2_twist_distribution.png"),
  mission: path.join(FIGURE_ROOT, "mission_tab.png"),
  structures: path.join(FIGURE_ROOT, "structures_tab.png"),
};

for (const mustExist of [BG_TITLE, BG_TRANSITION, BG_CONTENT, ...Object.values(FIG)]) {
  if (!fs.existsSync(mustExist)) throw new Error(`Missing asset: ${mustExist}`);
}

pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "North Carolina Agricultural and Technical State University";
pptx.subject = "Thesis defense for the Flying Wing Tool";
pptx.title = "Preliminary Design Tool for Flying Wing Unmanned Aerial Systems";
pptx.lang = "en-US";
pptx.theme = { headFontFace: FONTS.heading, bodyFontFace: FONTS.body, lang: "en-US" };

function fitBody(text, opts) {
  return autoFontSize(text, FONTS.body, {
    fontSize: opts.fontSize ?? 16,
    minFontSize: opts.minFontSize ?? 12.5,
    maxFontSize: opts.maxFontSize ?? 18,
    mode: "shrink",
    margin: opts.margin ?? 0,
    padding: opts.padding ?? 0,
    x: opts.x,
    y: opts.y,
    w: opts.w,
    h: opts.h,
    color: opts.color ?? COLORS.ink,
    breakLine: false,
    valign: opts.valign ?? "top",
  });
}

function addBackground(slide, pageNum) {
  slide.addImage({ path: BG_CONTENT, x: 0, y: 0, w: SLIDE_W, h: SLIDE_H });
  slide.addShape(pptx.ShapeType.rect, {
    x: 12.64,
    y: 7.0,
    w: 0.54,
    h: 0.2,
    line: { color: COLORS.gold, transparency: 100 },
    fill: { color: COLORS.gold },
  });
  slide.addText(String(pageNum), {
    x: 12.78,
    y: 7.005,
    w: 0.2,
    h: 0.12,
    margin: 0,
    fontFace: FONTS.heading,
    fontSize: 7.5,
    bold: true,
    color: COLORS.blue,
    align: "center",
    valign: "mid",
  });
}

function addSlideTitle(slide, title, eyebrow = null) {
  if (eyebrow) {
    slide.addText(eyebrow, {
      x: 0.58,
      y: 0.58,
      w: 3.2,
      h: 0.18,
      margin: 0,
      fontFace: FONTS.heading,
      fontSize: 9,
      bold: true,
      color: COLORS.gray,
      allCaps: true,
      charSpace: 1.1,
    });
  }
  slide.addText(title, {
    x: 0.58,
    y: eyebrow ? 0.78 : 0.65,
    w: 8.9,
    h: 0.45,
    margin: 0,
    fontFace: FONTS.heading,
    fontSize: 24,
    bold: true,
    color: COLORS.blue,
  });
}

function addBodyText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, fitBody(text, { x, y, w, h, ...opts }));
}

function addBullets(slide, items, x, y, w, opts = {}) {
  const gap = opts.gap ?? 0.53;
  const boxH = opts.boxH ?? 0.38;
  const fontSize = opts.fontSize ?? 15.5;
  items.forEach((item, index) => {
    slide.addText(item, {
      x,
      y: y + index * gap,
      w,
      h: boxH,
      margin: 0,
      valign: "top",
      fontFace: FONTS.body,
      fontSize,
      color: opts.color ?? COLORS.ink,
      bullet: { indent: 14 },
      hanging: 2,
    });
  });
}

function addPanel(slide, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    line: { color: opts.lineColor ?? COLORS.line, pt: opts.linePt ?? 1 },
    fill: { color: opts.fillColor ?? COLORS.white },
  });
}

function addStatCard(slide, x, y, w, h, label, value, note, opts = {}) {
  addPanel(slide, x, y, w, h, { fillColor: opts.fillColor ?? COLORS.offWhite });
  slide.addText(label, {
    x: x + 0.14,
    y: y + 0.11,
    w: w - 0.28,
    h: 0.18,
    margin: 0,
    fontFace: FONTS.heading,
    fontSize: 8.5,
    bold: true,
    color: COLORS.gray,
    allCaps: true,
    charSpace: 0.8,
  });
  slide.addText(value, {
    x: x + 0.14,
    y: y + 0.31,
    w: w - 0.28,
    h: 0.36,
    margin: 0,
    fontFace: FONTS.heading,
    fontSize: opts.valueSize ?? 20,
    bold: true,
    color: opts.valueColor ?? COLORS.blue,
  });
  if (note) addBodyText(slide, note, x + 0.14, y + h - 0.32, w - 0.28, 0.18, {
    fontSize: 9.5,
    minFontSize: 8.5,
    maxFontSize: 10.5,
    color: COLORS.gray,
  });
}

function addCaption(slide, text, x, y, w) {
  slide.addText(text, {
    x,
    y,
    w,
    h: 0.18,
    margin: 0,
    align: "center",
    fontFace: FONTS.body,
    fontSize: 9,
    italic: true,
    color: COLORS.gray,
  });
}

function addImageFrame(slide, imgPath, x, y, w, h, caption = null) {
  addPanel(slide, x, y, w, h, { fillColor: COLORS.white });
  const pad = 0.07;
  slide.addImage({
    path: imgPath,
    ...imageSizingContain(imgPath, x + pad, y + pad, w - 2 * pad, h - 2 * pad),
  });
  if (caption) addCaption(slide, caption, x + 0.02, y + h + 0.03, w - 0.04);
}

function addSvgPanel(slide, svg, x, y, w, h) {
  addPanel(slide, x, y, w, h, { fillColor: COLORS.white });
  slide.addImage({ data: svgToDataUri(svg), x: x + 0.05, y: y + 0.05, w: w - 0.1, h: h - 0.1 });
}

function workflowSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650">
  <defs>
    <style>
      .title{font:700 28px 'Century Gothic';fill:#${COLORS.blue}}
      .label{font:600 20px 'Century Gothic';fill:#${COLORS.ink}}
      .body{font:18px 'Palatino Linotype';fill:#${COLORS.gray}}
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#${COLORS.blue}"/>
    </marker>
  </defs>
  <rect x="15" y="15" width="1170" height="620" rx="24" fill="#fff" stroke="#${COLORS.line}" stroke-width="3"/>
  <text x="180" y="70" class="title">Fragmented workflow</text>
  <text x="735" y="70" class="title">Integrated workflow</text>
  <rect x="75" y="85" width="355" height="90" rx="18" fill="#${COLORS.paleGray}" stroke="#${COLORS.line}" stroke-width="2"/><text x="100" y="120" class="label">Geometry</text><text x="100" y="150" class="body">Manual rebuilds between tools</text>
  <rect x="75" y="205" width="355" height="90" rx="18" fill="#${COLORS.paleGray}" stroke="#${COLORS.line}" stroke-width="2"/><text x="100" y="240" class="label">Aerodynamics</text><text x="100" y="270" class="body">Reference quantities can drift</text>
  <rect x="75" y="325" width="355" height="90" rx="18" fill="#${COLORS.paleGray}" stroke="#${COLORS.line}" stroke-width="2"/><text x="100" y="360" class="label">Structure</text><text x="100" y="390" class="body">Loads and geometry separate</text>
  <rect x="75" y="445" width="355" height="90" rx="18" fill="#${COLORS.paleGray}" stroke="#${COLORS.line}" stroke-width="2"/><text x="100" y="480" class="label">Mission</text><text x="100" y="510" class="body">Energy assumptions live elsewhere</text>
  <path d="M252 175 L252 205" stroke="#${COLORS.gray}" stroke-width="3" stroke-dasharray="8 8"/>
  <path d="M252 295 L252 325" stroke="#${COLORS.gray}" stroke-width="3" stroke-dasharray="8 8"/>
  <path d="M252 415 L252 445" stroke="#${COLORS.gray}" stroke-width="3" stroke-dasharray="8 8"/>
  <rect x="675" y="200" width="380" height="190" rx="24" fill="#${COLORS.paleBlue}" stroke="#${COLORS.blue}" stroke-width="3"/>
  <text x="725" y="252" class="title" style="font-size:34px">Shared Project Model</text>
  <text x="725" y="288" class="body" style="font-size:20px">One aircraft definition carries geometry,</text>
  <text x="725" y="316" class="body" style="font-size:20px">analysis settings, mission state, and export data.</text>
  <rect x="530" y="130" width="150" height="70" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="548" y="172" class="label">Geometry</text>
  <rect x="530" y="440" width="190" height="70" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="548" y="482" class="label">Aerodynamics</text>
  <rect x="980" y="130" width="150" height="70" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="998" y="172" class="label">Structure</text>
  <rect x="975" y="440" width="150" height="70" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="1000" y="482" class="label">Mission</text>
  <rect x="805" y="520" width="120" height="70" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="828" y="562" class="label">Export</text>
  <line x1="675" y1="250" x2="605" y2="165" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow)"/>
  <line x1="675" y1="345" x2="625" y2="475" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow)"/>
  <line x1="1055" y1="165" x2="1060" y2="200" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow)"/>
  <line x1="1045" y1="475" x2="1045" y2="390" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow)"/>
  <line x1="865" y1="520" x2="865" y2="390" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow)"/>
  </svg>`;
}

function sharedStateSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">
  <defs>
    <style>
      .title{font:700 32px 'Century Gothic';fill:#fff}
      .label{font:700 22px 'Century Gothic';fill:#${COLORS.blue}}
      .body{font:18px 'Palatino Linotype';fill:#${COLORS.gray}}
    </style>
    <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#${COLORS.blue}"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="1160" height="660" rx="24" fill="#fff" stroke="#${COLORS.line}" stroke-width="3"/>
  <rect x="360" y="225" width="480" height="250" rx="30" fill="#${COLORS.blue}" stroke="#${COLORS.blue}" stroke-width="2"/>
  <text x="432" y="295" class="title">Shared Project Model</text>
  <text x="418" y="336" class="title" style="font-size:22px">Geometry + analysis settings + mission state</text>
  <text x="446" y="370" class="title" style="font-size:22px">optimized twist + stored results + exports</text>
  <rect x="110" y="110" width="310" height="110" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="130" y="146" class="label">Geometry definition</text><text x="130" y="177" class="body">Spanwise sections, planform, control surfaces</text>
  <rect x="790" y="110" width="310" height="110" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="810" y="146" class="label">Aerodynamic analysis</text><text x="810" y="177" class="body">Cruise point, polars, twist synthesis</text>
  <rect x="110" y="480" width="310" height="110" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="130" y="516" class="label">Structural assessment</text><text x="130" y="547" class="body">Loads, mass, stress, margins, feasibility</text>
  <rect x="790" y="480" width="310" height="110" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="810" y="516" class="label">Mission and propulsion</text><text x="810" y="547" class="body">Energy use, power demand, trajectory results</text>
  <rect x="435" y="560" width="330" height="90" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="455" y="596" class="label">Export and interoperability</text><text x="455" y="627" class="body">CAD, ribs, spars, CFD-ready geometry</text>
  <line x1="360" y1="260" x2="420" y2="220" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow2)"/>
  <line x1="840" y1="260" x2="790" y2="220" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow2)"/>
  <line x1="360" y1="430" x2="420" y2="480" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow2)"/>
  <line x1="840" y1="430" x2="790" y2="480" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow2)"/>
  <line x1="600" y1="475" x2="600" y2="560" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow2)"/>
  </svg>`;
}

function architectureSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="720" viewBox="0 0 1240 720">
  <defs>
    <style>
      .layer{font:700 24px 'Century Gothic';fill:#${COLORS.blue}}
      .label{font:700 20px 'Century Gothic';fill:#${COLORS.ink}}
      .body{font:17px 'Palatino Linotype';fill:#${COLORS.gray}}
    </style>
    <marker id="arrow3" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#${COLORS.blue}"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="1200" height="680" rx="24" fill="#fff" stroke="#${COLORS.line}" stroke-width="3"/>
  <text x="65" y="72" class="layer">User interfaces</text>
  <rect x="60" y="95" width="1120" height="110" rx="20" fill="#${COLORS.paleBlue}" stroke="#${COLORS.line}" stroke-width="2"/>
  <rect x="120" y="120" width="280" height="62" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="195" y="160" class="label">GUI application</text>
  <rect x="470" y="120" width="300" height="62" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="530" y="160" class="label">Batch / script entry points</text>
  <rect x="840" y="120" width="260" height="62" rx="16" fill="#fff" stroke="#${COLORS.line}" stroke-width="2"/><text x="905" y="160" class="label">Saved project files</text>
  <text x="65" y="264" class="layer">Shared data layer</text>
  <rect x="60" y="286" width="1120" height="108" rx="22" fill="#${COLORS.blue}" stroke="#${COLORS.blue}" stroke-width="2"/>
  <text x="130" y="338" style="font:700 28px 'Century Gothic';fill:#fff">Unified project schema / aircraft definition</text>
  <text x="130" y="372" style="font:20px 'Palatino Linotype';fill:#fff">One source of truth for geometry, operating conditions, stored results, and export products</text>
  <text x="65" y="445" class="layer">Analysis services</text>
  <rect x="60" y="468" width="256" height="130" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="78" y="505" class="label">Geometry + aerodynamics</text><text x="78" y="540" class="body">AeroSandbox-based geometry, cruise, polars, twist</text>
  <rect x="348" y="468" width="256" height="130" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="366" y="505" class="label">Structures</text><text x="366" y="540" class="body">Wingbox sizing, loads, stress, margins, feasibility</text>
  <rect x="636" y="468" width="256" height="130" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="654" y="505" class="label">Mission + propulsion</text><text x="654" y="540" class="body">3-DOF mission reasoning and energy use</text>
  <rect x="924" y="468" width="256" height="130" rx="18" fill="#${COLORS.offWhite}" stroke="#${COLORS.line}" stroke-width="2"/><text x="942" y="505" class="label">Export + interoperability</text><text x="942" y="540" class="body">CAD, ribs, spars, downstream geometry</text>
  <text x="65" y="648" class="layer">External endpoints</text><text x="260" y="648" class="body" style="font-size:19px">CFD, CAD, manufacturing, and validation studies connect outward from the same definition.</text>
  <line x1="260" y1="205" x2="260" y2="286" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  <line x1="620" y1="205" x2="620" y2="286" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  <line x1="970" y1="205" x2="970" y2="286" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  <line x1="188" y1="394" x2="188" y2="468" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  <line x1="476" y1="394" x2="476" y2="468" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  <line x1="764" y1="394" x2="764" y2="468" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  <line x1="1052" y1="394" x2="1052" y2="468" stroke="#${COLORS.blue}" stroke-width="4" marker-end="url(#arrow3)"/>
  </svg>`;
}

function addCheck(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function buildDeck() {
  let page = 1;
  let slide = pptx.addSlide();
  slide.addImage({ path: BG_TITLE, x: 0, y: 0, w: SLIDE_W, h: SLIDE_H });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.62,
    y: 0.68,
    w: 7.35,
    h: 5.10,
    line: { color: COLORS.blue, transparency: 100 },
    fill: { color: COLORS.blue },
  });
  slide.addText("Thesis Defense", { x: 0.76, y: 0.88, w: 2.8, h: 0.25, margin: 0, fontFace: FONTS.heading, fontSize: 11, bold: true, color: COLORS.gold, allCaps: true, charSpace: 1.2 });
  slide.addText("Preliminary Design Tool for\nFlying Wing Unmanned Aerial Systems", { x: 0.76, y: 1.25, w: 6.6, h: 1.3, margin: 0, fontFace: FONTS.title, fontSize: 24, bold: true, color: COLORS.white });
  slide.addText("Malik D. Henry", { x: 0.8, y: 3.2, w: 3.8, h: 0.25, margin: 0, fontFace: FONTS.heading, fontSize: 16, bold: true, color: COLORS.white });
  slide.addText("Master of Science in Mechanical Engineering", { x: 0.8, y: 3.55, w: 5.7, h: 0.22, margin: 0, fontFace: FONTS.body, fontSize: 12.5, color: COLORS.white });
  slide.addText("Advisor: John P. Kizito\nNorth Carolina Agricultural and Technical State University\nGreensboro, North Carolina | April 2026", { x: 0.8, y: 4.18, w: 5.6, h: 0.85, margin: 0, fontFace: FONTS.body, fontSize: 13, color: COLORS.white });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Why Flying Wings Are Hard to Design", "Problem Context");
  addBullets(slide, [
    "The wing must simultaneously provide lift, trim authority, stability, and a practical structural load path.",
    "Sweep, twist, airfoil choice, control-surface layout, and center-of-gravity target all interact from the start.",
    "Small geometry changes can alter aerodynamics, structure, mission feasibility, and manufacturability at the same time.",
    "The early-stage challenge is therefore a workflow problem as much as a physics problem.",
  ], 0.7, 1.55, 5.35, { gap: 0.64, boxH: 0.48, fontSize: 15.3 });
  addImageFrame(slide, FIG.cad, 6.55, 1.45, 5.95, 4.4, "Representative blended flying-wing geometry");
  addPanel(slide, 0.7, 6.1, 11.8, 0.68, { fillColor: COLORS.paleGold });
  addBodyText(slide, "Preliminary design only becomes useful when the aircraft can move through coupled analyses without being rebuilt between tools.", 0.92, 6.28, 11.35, 0.24, { fontSize: 15, minFontSize: 13.5, maxFontSize: 15.5 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Research Gap: Fragmented Workflow", "Motivation");
  addBullets(slide, [
    "Existing tools are effective within their own domains, but usually capture only one slice of the design problem.",
    "Manual handoffs across aerodynamics, CAD, mission, and structures slow iteration and weaken traceability.",
  ], 0.72, 1.42, 11.5, { gap: 0.55, boxH: 0.4, fontSize: 15.2 });
  addSvgPanel(slide, workflowSvg(), 0.72, 2.42, 11.8, 3.66);
  addPanel(slide, 0.72, 6.18, 11.8, 0.5, { fillColor: COLORS.paleBlue });
  addBodyText(slide, "Research question: how can one software environment support geometry, preliminary analysis, mission evaluation, and downstream export without forcing the user to rebuild the aircraft model between stages?", 0.92, 6.31, 11.35, 0.22, { fontSize: 13.8, minFontSize: 12.5, maxFontSize: 14.5 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page);
  slide.addText("Literature Review", { x: 0.78, y: 0.88, w: 2.4, h: 0.18, margin: 0, fontFace: FONTS.heading, fontSize: 9, bold: true, color: COLORS.gray, allCaps: true, charSpace: 1.0 });
  slide.addText("Past Work and Motivation", { x: 0.78, y: 1.08, w: 5.8, h: 0.34, margin: 0, fontFace: FONTS.heading, fontSize: 22, bold: true, color: COLORS.blue });
  addPanel(slide, 0.72, 1.42, 2.75, 1.78, { fillColor: COLORS.paleBlue });
  addPanel(slide, 3.63, 1.42, 2.75, 1.78, { fillColor: COLORS.offWhite });
  addPanel(slide, 6.54, 1.42, 2.75, 1.78, { fillColor: COLORS.offWhite });
  addPanel(slide, 9.45, 1.42, 2.80, 1.78, { fillColor: COLORS.paleGold });
  slide.addText("Conceptual design baseline", { x: 0.92, y: 1.64, w: 2.3, h: 0.2, margin: 0, fontFace: FONTS.heading, fontSize: 13.5, bold: true, color: COLORS.blue });
  slide.addText("Cross-tool comparisons", { x: 3.83, y: 1.64, w: 2.2, h: 0.2, margin: 0, fontFace: FONTS.heading, fontSize: 13.5, bold: true, color: COLORS.blue });
  slide.addText("Low-order solver limits", { x: 6.74, y: 1.64, w: 2.2, h: 0.2, margin: 0, fontFace: FONTS.heading, fontSize: 13.5, bold: true, color: COLORS.blue });
  slide.addText("Optimization-first workflows", { x: 9.65, y: 1.64, w: 2.35, h: 0.2, margin: 0, fontFace: FONTS.heading, fontSize: 13.5, bold: true, color: COLORS.blue });
  addBodyText(slide, "Raymer and Torenbeek establish the standard preliminary-design decomposition; Nguyen et al. show how higher-fidelity corrections can be inserted into an MDO loop.", 0.92, 1.94, 2.32, 0.92, { fontSize: 11.2, minFontSize: 10.2, maxFontSize: 11.5 });
  addBodyText(slide, "Vegh et al. show that NDARC and SUAVE can disagree on trends and optima unless assumptions, drag models, and mission accounting are traceable.", 3.83, 1.94, 2.32, 0.92, { fontSize: 11.2, minFontSize: 10.2, maxFontSize: 11.5 });
  addBodyText(slide, "Deperrois, Yu et al., and Rosas-Cordova et al. make the same point from different angles: low-order tools are useful, but validity depends on setup and regime.", 6.74, 1.94, 2.32, 0.92, { fontSize: 11.0, minFontSize: 10.0, maxFontSize: 11.3 });
  addBodyText(slide, "Sharpe's AeroSandbox and later work on graph transformations and NeuralFoil shift the focus toward robust, differentiable, optimization-oriented analysis.", 9.65, 1.94, 2.32, 0.92, { fontSize: 11.0, minFontSize: 10.0, maxFontSize: 11.3 });
  addBullets(slide, [
    "No single source provided an open flying-wing workflow that keeps geometry, aerodynamics, structures, mission, and export connected.",
    "The literature therefore motivates two requirements at once: fast low-order analysis and disciplined traceability across tools.",
    "This thesis positions the Flying Wing Tool as a response to that workflow gap rather than as a replacement for all external solvers.",
  ], 0.92, 3.72, 11.0, { gap: 0.78, boxH: 0.6, fontSize: 14.0 });
  addPanel(slide, 0.92, 6.08, 11.25, 0.48, { fillColor: COLORS.paleBlue });
  addBodyText(slide, "Representative prior-work anchors: Raymer (2018), Nguyen et al. (2013), Vegh et al. (2019), Deperrois (2009), Yu et al. (2022), Rosas-Cordova et al. (2024), Sharpe (2021, 2024, 2025).", 1.12, 6.21, 10.85, 0.18, { fontSize: 11.4, minFontSize: 10.4, maxFontSize: 11.8 });
  addCheck(slide);

  slide = pptx.addSlide();
  slide.addImage({ path: BG_CONTENT, x: 0, y: 0, w: SLIDE_W, h: SLIDE_H });
  slide.addShape(pptx.ShapeType.rect, {
    x: 12.42,
    y: 7.0,
    w: 0.86,
    h: 0.24,
    line: { color: COLORS.gold, transparency: 100 },
    fill: { color: COLORS.gold },
  });
  slide.addText("Section Break", { x: 0.76, y: 1.98, w: 2.4, h: 0.20, margin: 0, fontFace: FONTS.heading, fontSize: 10, bold: true, color: COLORS.gray, allCaps: true, charSpace: 1.1 });
  slide.addText("Core Contribution", { x: 0.76, y: 2.28, w: 5.6, h: 0.55, margin: 0, fontFace: FONTS.title, fontSize: 26, bold: true, color: COLORS.blue });
  slide.addText("An integrated preliminary-design workflow built around one shared aircraft definition", { x: 0.76, y: 2.95, w: 7.4, h: 0.4, margin: 0, fontFace: FONTS.body, fontSize: 16, color: COLORS.gray });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Shared-State Workflow", "Contribution");
  addSvgPanel(slide, sharedStateSvg(), 0.7, 1.35, 7.1, 4.95);
  addBullets(slide, [
    "Geometry, analysis settings, mission assumptions, optimized twist, and stored outputs remain in one project model.",
    "Twist optimization writes back into the same state later used for polars, structures, and export.",
    "Batch studies can reload a saved project file without GUI-specific reconstruction.",
    "The main thesis contribution is continuity of state rather than a single new solver.",
  ], 8.15, 1.58, 4.05, { gap: 0.78, boxH: 0.6, fontSize: 14.2 });
  addPanel(slide, 8.15, 5.78, 4.05, 0.78, { fillColor: COLORS.paleGold });
  addBodyText(slide, "The value proposition is simple: one aircraft definition persists across the preliminary design loop.", 8.36, 6.02, 3.62, 0.25, { fontSize: 14.2, minFontSize: 12.5, maxFontSize: 14.5 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Implemented Software Architecture", "Architecture");
  addSvgPanel(slide, architectureSvg(), 0.7, 1.35, 7.25, 5.05);
  addBullets(slide, [
    "GUI and batch entry points operate on the same shared project schema.",
    "Core services cover geometry and aerodynamics, structures, mission and propulsion, plus export.",
    "External tools are used only when higher-fidelity analysis or downstream fabrication requires them.",
  ], 8.18, 1.72, 4.02, { gap: 0.95, boxH: 0.72, fontSize: 14.2 });
  addPanel(slide, 8.18, 5.92, 4.02, 0.64, { fillColor: COLORS.paleBlue });
  addBodyText(slide, "Built around AeroSandbox with custom geometry, structural, mission, and export services.", 8.4, 6.11, 3.6, 0.18, { fontSize: 12.8, minFontSize: 11.8, maxFontSize: 13.2 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Integrated Software Surface", "Implementation");
  addBodyText(slide, "The GUI exposes the same shared project state across geometry definition, aerodynamic analysis, mission setup, structures, and export.", 0.72, 1.30, 11.7, 0.20, { fontSize: 13.2, minFontSize: 12.0, maxFontSize: 13.8 });
  [[FIG.geometry, "Geometry", 0.72, 1.62, 3.88, 2.05],[FIG.analysis, "Analysis", 4.73, 1.62, 3.88, 2.05],[FIG.mission, "Mission", 8.74, 1.62, 3.58, 2.05],[FIG.structures, "Structures", 2.0, 4.05, 3.88, 2.05],[FIG.export, "Export", 6.96, 4.05, 3.88, 2.05]].forEach(([img, label, x, y, w, h]) => { addImageFrame(slide, img, x, y, w, h); addCaption(slide, label, x, y + h + 0.04, w); });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Geometry to Analysis to Export Is Traceable", "Traceability");
  addImageFrame(slide, FIG.geometry, 0.7, 1.45, 3.45, 2.35, "Geometry definition");
  addImageFrame(slide, FIG.cad, 4.48, 1.45, 4.35, 2.35, "Generated aircraft geometry");
  addImageFrame(slide, FIG.export, 9.18, 1.45, 3.15, 2.35, "Export products");
  slide.addText("→", { x: 4.20, y: 2.3, w: 0.22, h: 0.3, margin: 0, fontFace: FONTS.heading, fontSize: 22, bold: true, color: COLORS.blue });
  slide.addText("→", { x: 8.91, y: 2.3, w: 0.22, h: 0.3, margin: 0, fontFace: FONTS.heading, fontSize: 22, bold: true, color: COLORS.blue });
  addBullets(slide, [
    "Spanwise sections drive aerodynamic loading, structural sizing, and manufacturing geometry.",
    "Spar locations and control-surface definitions persist across analysis and export steps.",
    "The aerodynamic wing and the manufactured wing are derived from the same section model.",
  ], 0.88, 4.55, 11.2, { gap: 0.64, boxH: 0.48, fontSize: 15 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Representative Case: IntendedValidation2", "Case Study");
  addImageFrame(slide, FIG.cad, 0.72, 1.45, 5.65, 4.35, "Small blended flying-wing configuration");
  [["Planform area", "1.625 m²", "Actual planform area used for CFD rescaling"],["Aspect ratio", "5.63", "Actual span and planform-based value"],["Span", "3.025 m", "Including the centerbody"],["Gross weight", "12.0 kg", "Case-study takeoff weight"],["Lift target", "Bell", "Bell-shaped target lift distribution"],["Static margin", "8%", "Trim input retained in the same project"]].forEach((card, idx) => { const row = Math.floor(idx / 2); const col = idx % 2; addStatCard(slide, 6.72 + col * 2.82, 1.55 + row * 1.38, 2.55, 1.12, ...card, { valueSize: 16.5 }); });
  addPanel(slide, 0.72, 6.02, 11.8, 0.6, { fillColor: COLORS.paleBlue });
  addBodyText(slide, "This case is useful because it is neither trivial nor uniformly successful: it produces plausible aerodynamic results, a coherent twist distribution, and lightweight structure, but still exposes a governing local structural limit.", 0.92, 6.18, 11.35, 0.23, { fontSize: 13.4, minFontSize: 12.2, maxFontSize: 14 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Aerodynamic Results and Twist Synthesis", "Case Study");
  addImageFrame(slide, FIG.clPolar, 0.72, 1.48, 3.78, 3.12, "Lift polar");
  addImageFrame(slide, FIG.cdPolar, 4.78, 1.48, 3.78, 3.12, "Drag polar");
  addImageFrame(slide, FIG.twist, 8.84, 1.48, 3.48, 3.12, "Optimized twist");
  addStatCard(slide, 1.05, 5.28, 2.95, 1.05, "Cruise lift coefficient", "0.300", "Target design CL is met");
  addStatCard(slide, 4.3, 5.28, 2.95, 1.05, "Cruise drag coefficient", "0.0129", "Low-order cruise estimate");
  addStatCard(slide, 7.55, 5.28, 2.95, 1.05, "Cruise L/D", "23.21", "Encouraging for a compact flying wing");
  addBodyText(slide, "The same shared project object carries trim, twist, and operating-point assumptions into the aerodynamic outputs.", 1.08, 6.48, 9.8, 0.18, { fontSize: 12.8, minFontSize: 11.8, maxFontSize: 13.2, color: COLORS.gray });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "One Project Model Also Supports Structure and Mission", "Case Study");
  slide.addText("Structural outputs", { x: 0.85, y: 1.42, w: 2.5, h: 0.22, margin: 0, fontFace: FONTS.heading, fontSize: 15, bold: true, color: COLORS.blue });
  slide.addText("Mission outputs", { x: 6.95, y: 1.42, w: 2.5, h: 0.22, margin: 0, fontFace: FONTS.heading, fontSize: 15, bold: true, color: COLORS.blue });
  [["Structural mass", "2.363 kg", "Estimated wingbox-oriented mass"],["Tip deflection", "2.27 mm", "Very small relative to span"],["Min rib crushing margin", "0.97", "Governing local structural limit"],["Feasibility flag", "False", "Global structure is good, but local crushing still fails"]].forEach((card, idx) => addStatCard(slide, 0.8 + (idx % 2) * 2.55, 1.8 + Math.floor(idx / 2) * 1.38, 2.3, 1.08, ...card, { valueSize: 16.5, valueColor: idx === 3 ? COLORS.red : COLORS.blue, fillColor: idx >= 2 ? COLORS.paleGold : COLORS.offWhite }));
  [["Mission success", "True", "9 of 9 phases completed"],["Distance", "1283 m", "Stored mission result in the project file"],["Energy used", "6.64 Wh", "Mission-level electrical usage"],["Max electrical power", "2.30 kW", "Peak mission power demand"]].forEach((card, idx) => addStatCard(slide, 6.9 + (idx % 2) * 2.55, 1.8 + Math.floor(idx / 2) * 1.38, 2.3, 1.08, ...card, { valueSize: 16.5, valueColor: idx === 0 ? COLORS.green : COLORS.blue }));
  slide.addShape(pptx.ShapeType.line, { x: 6.2, y: 1.7, w: 0, h: 4.05, line: { color: COLORS.line, pt: 1.25 } });
  addPanel(slide, 0.82, 5.98, 11.65, 0.58, { fillColor: COLORS.paleBlue });
  addBodyText(slide, "This is the thesis value of the workflow: one saved design reveals favorable global trends while still exposing the local rib-crushing mechanism that prevents full structural feasibility.", 1.02, 6.14, 11.2, 0.21, { fontSize: 13.4, minFontSize: 12.2, maxFontSize: 14 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "BARAM CFD Cross-Check Setup", "Validation Path");
  addImageFrame(slide, FIG.baramState, 0.72, 1.48, 6.45, 4.3, "Saved BARAM and ParaView state for the same geometry");
  addBullets(slide, [
    "The same exported geometry was solved in BARAM instead of being rebuilt in a separate CAD workflow.",
    "The case uses an incompressible RANS solution with a standard k-epsilon model and a symmetry half-model.",
    "The stored solver log reports convergence after 604 SIMPLE iterations.",
    "For this thesis pass, CFD was the independent validation path because a demonstrator airframe was not completed within manufacturing and time constraints.",
  ], 7.55, 1.65, 4.35, { gap: 0.82, boxH: 0.65, fontSize: 13.9 });
  addPanel(slide, 7.55, 5.58, 4.35, 0.94, { fillColor: COLORS.paleGold });
  addBodyText(slide, "A four-level mesh-sensitivity study was completed for cruise and takeoff. The former fine level is relabeled Medium 2, and the final refinement is reported as Fine.", 7.76, 5.78, 3.92, 0.46, { fontSize: 12.1, minFontSize: 11.0, maxFontSize: 12.6 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Mesh-Sensitivity Results", "Validation Path");
  addPanel(slide, 0.72, 1.45, 5.78, 4.7, { fillColor: COLORS.paleBlue });
  addPanel(slide, 6.83, 1.45, 5.78, 4.7, { fillColor: COLORS.offWhite });
  slide.addText("Cruise", { x: 0.98, y: 1.73, w: 1.5, h: 0.24, margin: 0, fontFace: FONTS.heading, fontSize: 17, bold: true, color: COLORS.blue });
  slide.addText("Takeoff", { x: 7.09, y: 1.73, w: 1.7, h: 0.24, margin: 0, fontFace: FONTS.heading, fontSize: 17, bold: true, color: COLORS.blue });
  addBodyText(slide, "Cells: 0.423M -> 2.159M -> 2.773M -> 8.505M", 1.02, 2.05, 4.95, 0.22, { fontSize: 12.4, minFontSize: 11.4, maxFontSize: 12.8, color: COLORS.gray });
  addBodyText(slide, "Cells: 0.423M -> 2.159M -> 2.773M -> 33.901M", 7.13, 2.05, 4.95, 0.22, { fontSize: 12.4, minFontSize: 11.4, maxFontSize: 12.8, color: COLORS.gray });
  addStatCard(slide, 0.98, 2.48, 1.62, 1.1, "M2 -> F dCd", "2.02%", "Final refinement change", { valueSize: 18 });
  addStatCard(slide, 2.82, 2.48, 1.62, 1.1, "M2 -> F dCl", "0.57%", "Final refinement change", { valueSize: 18 });
  addStatCard(slide, 4.66, 2.48, 1.4, 1.1, "Read", "Mixed", "Lift stable, drag still moving", { valueSize: 18, valueColor: COLORS.blue, fillColor: COLORS.paleGold });
  addStatCard(slide, 7.09, 2.48, 1.62, 1.1, "M2 -> F dCd", "3.13%", "Final refinement change", { valueSize: 18 });
  addStatCard(slide, 8.93, 2.48, 1.62, 1.1, "M2 -> F dCl", "0.58%", "Final refinement change", { valueSize: 18 });
  addStatCard(slide, 10.77, 2.48, 1.4, 1.1, "Read", "Sensitive", "Drag still moves materially", { valueSize: 18, valueColor: COLORS.red, fillColor: COLORS.paleGold });
  addBullets(slide, [
    "The former fine mesh is relabeled Medium 2 because it sat very close to medium in total cell count.",
    "Medium to Medium 2 is already tight in cruise, but the final Fine mesh still reduces Cd by about 2%.",
    "Cruise is best described as lift-stable with residual drag sensitivity, not as a strict grid-independence proof.",
  ], 1.02, 3.92, 5.0, { gap: 0.68, boxH: 0.5, fontSize: 12.8 });
  addBullets(slide, [
    "The same relabeling is used on takeoff so the four-level ladder reads coarse, medium, Medium 2, and Fine.",
    "The final takeoff refinement tightens lift but still changes drag by about 3.13%.",
    "Takeoff is therefore presented as a mesh-sensitivity result rather than a settled drag prediction.",
  ], 7.13, 3.92, 5.0, { gap: 0.68, boxH: 0.5, fontSize: 12.8 });
  addPanel(slide, 0.72, 6.35, 11.9, 0.58, { fillColor: COLORS.paleGold });
  addBodyText(slide, "Important limitation: retained wall-layer cells remained zero in the accepted meshes, so this establishes mesh sensitivity of the present setup rather than full boundary-layer independence.", 0.98, 6.50, 11.4, 0.2, { fontSize: 12.8, minFontSize: 11.8, maxFontSize: 13.0 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "CFD Cross-Check Results", "BARAM Comparison");
  addImageFrame(slide, FIG.baramConv, 0.72, 1.46, 5.15, 3.22, "Coefficient convergence history");
  addImageFrame(slide, FIG.baramCompare, 6.15, 1.46, 6.15, 3.22, "BARAM versus repository cruise point");
  addStatCard(slide, 0.98, 5.12, 3.55, 1.12, "Lift coefficient", "0.273 vs 0.300", "Directionally close after rescaling");
  addStatCard(slide, 4.88, 5.12, 3.55, 1.12, "Drag coefficient", "0.0341 vs 0.0129", "Material drag gap remains");
  addStatCard(slide, 8.78, 5.12, 3.1, 1.12, "L/D", "8.00 vs 23.21", "Drag dominates the discrepancy", { valueColor: COLORS.red, fillColor: COLORS.paleGold });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Interpreting the Drag Discrepancy", "Validation Boundary");
  addImageFrame(slide, FIG.dragAttrib, 0.72, 1.5, 5.2, 4.35, "Planform share versus drag share");
  addBullets(slide, [
    "The split-patch continuation shows that drag is concentrated in the blended centerbody region rather than the outer wing.",
    "Centerbody plus junction account for 32.2% of planform area but 94.3% of the drag share.",
    "Pressure drag dominates the mismatch; the missing term is not explained by outer-wing skin friction alone.",
    "The low-order stack remains useful for lift and trim trends, but it under-represents centerbody-dominated body/interference drag.",
  ], 6.4, 1.72, 5.65, { gap: 0.8, boxH: 0.62, fontSize: 14 });
  addPanel(slide, 6.4, 5.92, 5.65, 0.64, { fillColor: COLORS.paleBlue });
  addBodyText(slide, "This narrows the modeling limitation to a specific regime: thick blended centerbodies at the present Reynolds-number scale need more faithful drag modeling than a wing-only buildup provides.", 6.62, 6.11, 5.2, 0.18, { fontSize: 12.8, minFontSize: 11.8, maxFontSize: 13.2 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Validation Boundaries and Practical Constraints", "Scope Discipline");
  addPanel(slide, 0.78, 1.55, 5.45, 3.95, { fillColor: COLORS.paleBlue });
  addPanel(slide, 6.68, 1.55, 5.45, 3.95, { fillColor: COLORS.paleGold });
  slide.addText("Demonstrated now", { x: 1.02, y: 1.82, w: 2.8, h: 0.24, margin: 0, fontFace: FONTS.heading, fontSize: 17, bold: true, color: COLORS.blue });
  slide.addText("Not yet demonstrated", { x: 6.92, y: 1.82, w: 3.2, h: 0.24, margin: 0, fontFace: FONTS.heading, fontSize: 17, bold: true, color: COLORS.blue });
  addBullets(slide, [
    "An integrated workflow across geometry, aerodynamics, structures, mission, and export.",
    "Batch reproducibility from saved project files and shared state.",
    "An external CFD cross-check tied to the same aircraft geometry.",
  ], 1.02, 2.25, 4.7, { gap: 0.78, boxH: 0.6, fontSize: 14.2 });
  addBullets(slide, [
    "A physical demonstrator and test data.",
    "Detailed structure verification or joint-level design validation.",
    "Comprehensive predictive-accuracy claims across aircraft classes and regimes.",
  ], 6.92, 2.25, 4.7, { gap: 0.78, boxH: 0.6, fontSize: 14.2 });
  addPanel(slide, 0.78, 5.8, 11.35, 0.78, { fillColor: COLORS.paleGray });
  addBodyText(slide, "Because of manufacturing and schedule constraints, this thesis pass stops at software integration plus CFD cross-check rather than a completed demonstrator airframe.", 1.02, 6.05, 10.9, 0.22, { fontSize: 14.2, minFontSize: 12.8, maxFontSize: 14.5 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page); addSlideTitle(slide, "Conclusion and Next Steps", "Close");
  addPanel(slide, 0.78, 1.48, 5.55, 4.35, { fillColor: COLORS.paleBlue });
  addPanel(slide, 6.78, 1.48, 5.35, 4.35, { fillColor: COLORS.offWhite });
  slide.addText("What this thesis establishes", { x: 1.04, y: 1.76, w: 3.6, h: 0.24, margin: 0, fontFace: FONTS.heading, fontSize: 17, bold: true, color: COLORS.blue });
  addBullets(slide, [
    "A coherent preliminary-design environment for tailless aircraft can be built around one shared project model.",
    "The shared-state workflow preserves geometry-to-analysis-to-export traceability across disciplines.",
    "A single saved design can expose aerodynamic, structural, mission, and CFD consequences without being rebuilt.",
  ], 1.04, 2.22, 4.78, { gap: 0.88, boxH: 0.7, fontSize: 14.2 });
  slide.addText("Next steps", { x: 7.04, y: 1.76, w: 2.3, h: 0.24, margin: 0, fontFace: FONTS.heading, fontSize: 17, bold: true, color: COLORS.blue });
  addBullets(slide, [
    "Validate against published cases, independent solvers, or physical test data where practical.",
    "Improve drag modeling for thick blended centerbodies and body-interference effects.",
    "Extend end-to-end design studies and optimization loops while keeping claim discipline tight.",
  ], 7.04, 2.22, 4.54, { gap: 0.88, boxH: 0.7, fontSize: 14.2 });
  addPanel(slide, 0.78, 6.02, 11.35, 0.58, { fillColor: COLORS.paleGold });
  addBodyText(slide, "The main result is integration itself: the project has crossed the threshold from disconnected analysis scripts to a defensible preliminary-design workflow.", 1.02, 6.18, 10.9, 0.2, { fontSize: 14, minFontSize: 12.8, maxFontSize: 14.2 });
  addCheck(slide);

  slide = pptx.addSlide(); addBackground(slide, ++page);
  slide.addText("Sources", { x: 0.78, y: 0.88, w: 1.8, h: 0.18, margin: 0, fontFace: FONTS.heading, fontSize: 9, bold: true, color: COLORS.gray, allCaps: true, charSpace: 1.0 });
  slide.addText("References", { x: 0.78, y: 1.08, w: 3.2, h: 0.34, margin: 0, fontFace: FONTS.heading, fontSize: 22, bold: true, color: COLORS.blue });
  addPanel(slide, 0.78, 1.52, 5.60, 4.44, { fillColor: COLORS.offWhite });
  addPanel(slide, 6.93, 1.52, 5.60, 4.44, { fillColor: COLORS.offWhite });
  const refsLeft = [
    "Raymer, D. P. Aircraft Design: A Conceptual Approach. 6th ed., AIAA, 2018.",
    "Nguyen, N. V., Gilard, V., and Bingol, O. Multidisciplinary Unmanned Combat Air Vehicle system design using Multi-Fidelity Model. Aerospace Science and Technology, 2013.",
    "Vegh, J. M., Botero, E., et al. Comparison of NDARC and SUAVE for an eVTOL aircraft conceptual design: Kitty Hawk Cora. Stanford University, 2019.",
    "Deperrois, A. Guidelines for XFLR5 v6.03. XFLR5 documentation, 2009.",
  ];
  const refsRight = [
    "Yu, A., et al. Comparing Potential Flow Solvers for Aerodynamic Characteristics Estimation of the T-Flex UAV. ICAS 2022.",
    "Rosas-Cordova, A., et al. Validation of VSPAERO for Basic Wing Simulation. RIMNI, 2024.",
    "Sharpe, P. D. AeroSandbox: A Differentiable Framework for Aircraft Design Optimization. M.S. thesis, MIT, 2021.",
    "Sharpe, P. D. Accelerating Practical Engineering Design Optimization with Computational Graph Transformations. Ph.D. thesis, MIT, 2024.",
    "Sharpe, P. D. NeuralFoil: A physics-informed machine learning model for airfoil aerodynamics. arXiv:2503.16323, 2025.",
  ];
  refsLeft.forEach((ref, idx) => addBodyText(slide, ref, 1.02, 1.84 + idx * 0.96, 5.08, 0.62, { fontSize: 10.7, minFontSize: 9.8, maxFontSize: 11.0 }));
  refsRight.forEach((ref, idx) => addBodyText(slide, ref, 7.17, 1.84 + idx * 0.80, 5.08, 0.56, { fontSize: 10.1, minFontSize: 9.2, maxFontSize: 10.4 }));
  addPanel(slide, 0.78, 6.12, 11.75, 0.44, { fillColor: COLORS.paleGold });
  addBodyText(slide, "Full bibliography is retained in the written thesis; this slide lists the references used most directly in the presentation narrative.", 1.00, 6.24, 11.30, 0.16, { fontSize: 11.4, minFontSize: 10.4, maxFontSize: 11.6 });
  addCheck(slide);
}

async function main() {
  buildDeck();
  await pptx.writeFile({ fileName: OUTPUT });
  console.log(`Wrote ${OUTPUT}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
