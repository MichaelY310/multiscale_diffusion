import { access, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import vm from "node:vm";

const DEFAULT_INPUT = "docs/research_docs/RDD_METHODS_AND_COMBINATION_HANDBOOK_ZH.md";
const BUNDLED_MARKED = path.join(
  process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js",
);

const inputPath = path.resolve(process.argv[2] ?? DEFAULT_INPUT);
const outputPath = path.resolve(
  process.argv[3] ?? inputPath.replace(/\.md$/i, ".html"),
);

async function findKatexDirectory() {
  const vscodeRoot = path.join(
    process.env.LOCALAPPDATA,
    "Programs/Microsoft VS Code",
  );
  const entries = await readdir(vscodeRoot, { withFileTypes: true });
  const roots = [vscodeRoot, ...entries.filter((entry) => entry.isDirectory())
    .map((entry) => path.join(vscodeRoot, entry.name))];

  for (const root of roots) {
    const candidate = path.join(
      root,
      "resources/app/extensions/markdown-math/notebook-out",
    );
    try {
      await access(path.join(candidate, "katex.js"));
      await access(path.join(candidate, "katex.min.css"));
      return candidate;
    } catch {
      // Try the next VS Code version directory.
    }
  }
  throw new Error("VS Code's bundled KaTeX installation was not found.");
}

function loadKatex(bundleSource) {
  const cut = bundleSource.indexOf("var cn=tn");
  if (cut < 0) {
    throw new Error("The VS Code KaTeX bundle layout is unsupported.");
  }
  const sandbox = { module: { exports: {} }, exports: {}, console };
  vm.runInNewContext(
    `${bundleSource.slice(0, cut)}\nmodule.exports = rn();`,
    sandbox,
  );
  return sandbox.module.exports;
}

async function embedKatexFonts(css, katexDirectory) {
  let output = css
    .replace(/,url\((?:\.\/)?fonts\/[^)]+\.woff\) format\("woff"\)/g, "")
    .replace(/,url\((?:\.\/)?fonts\/[^)]+\.ttf\) format\("truetype"\)/g, "");
  const fontPattern = /url\((?:\.\/)?fonts\/([^)'\"]+\.woff2)\)/g;
  const fontNames = [...output.matchAll(fontPattern)].map((match) => match[1]);
  const uniqueFontNames = [...new Set(fontNames)];
  for (const fontName of uniqueFontNames) {
    const font = await readFile(path.join(katexDirectory, "fonts", fontName));
    const dataUrl = `url(data:font/woff2;base64,${font.toString("base64")})`;
    output = output.replaceAll(`url(fonts/${fontName})`, dataUrl);
    output = output.replaceAll(`url(./fonts/${fontName})`, dataUrl);
  }
  return output;
}

function renderMarkdownMath(markdown, marked, katex) {
  const protectedCode = [];
  const codeToken = (value) => {
    const index = protectedCode.push(value) - 1;
    return `RDDCODEPLACEHOLDER${index}END`;
  };

  let source = markdown
    .replace(/```[\s\S]*?```/g, codeToken)
    .replace(/`[^`\r\n]+`/g, codeToken);

  const formulas = [];
  const mathToken = (tex, displayMode) => {
    const index = formulas.push({ tex: tex.trim(), displayMode }) - 1;
    return displayMode
      ? `\n<div data-rdd-math-block="${index}"></div>\n`
      : `<span data-rdd-math-inline="${index}"></span>`;
  };

  source = source
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, tex) => mathToken(tex, true))
    .replace(/\$([^$\r\n]+?)\$/g, (_match, tex) => mathToken(tex, false));

  source = source.replace(/RDDCODEPLACEHOLDER(\d+)END/g, (_match, index) =>
    protectedCode[Number(index)]);

  let body = marked.parse(source, { gfm: true, breaks: false });
  const errors = [];
  formulas.forEach(({ tex, displayMode }, index) => {
    let rendered;
    try {
      rendered = katex.renderToString(tex, {
        displayMode,
        output: "htmlAndMathml",
        strict: "ignore",
        throwOnError: true,
      });
    } catch (error) {
      errors.push(`Formula ${index + 1}: ${error.message}\n${tex}`);
      rendered = `<code class="katex-error">${tex}</code>`;
    }
    const marker = displayMode
      ? `<div data-rdd-math-block="${index}"></div>`
      : `<span data-rdd-math-inline="${index}"></span>`;
    const replacement = displayMode
      ? `<div class="rdd-math-block">${rendered}</div>`
      : `<span class="rdd-math-inline">${rendered}</span>`;
    body = body.replace(marker, replacement);
  });

  if (errors.length) {
    throw new Error(`KaTeX failed to render:\n${errors.join("\n\n")}`);
  }
  return { body, formulaCount: formulas.length };
}

const markdown = await readFile(inputPath, "utf8");
const { marked } = await import(pathToFileURL(BUNDLED_MARKED).href);
const katexDirectory = await findKatexDirectory();
const katexBundle = await readFile(path.join(katexDirectory, "katex.js"), "utf8");
const katex = loadKatex(katexBundle);
const katexCss = await embedKatexFonts(
  await readFile(path.join(katexDirectory, "katex.min.css"), "utf8"),
  katexDirectory,
);
const { body: renderedBody, formulaCount } = renderMarkdownMath(
  markdown,
  marked,
  katex,
);

const title = path.basename(inputPath, path.extname(inputPath));
const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title}</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fa;
      --paper: #ffffff;
      --text: #1f2328;
      --muted: #59636e;
      --border: #d0d7de;
      --accent: #0969da;
      --code-bg: #f6f8fa;
      --quote: #656d76;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
        "Microsoft YaHei", Arial, sans-serif;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(220px, 290px) minmax(0, 960px);
      justify-content: center;
      gap: 28px;
      padding: 28px;
    }
    nav {
      position: sticky;
      top: 20px;
      max-height: calc(100vh - 40px);
      overflow: auto;
      align-self: start;
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--paper);
    }
    nav strong { display: block; margin-bottom: 10px; }
    nav a {
      display: block;
      padding: 3px 0;
      color: var(--muted);
      text-decoration: none;
      line-height: 1.4;
    }
    nav a:hover { color: var(--accent); }
    nav .toc-h3 { padding-left: 14px; font-size: 0.92em; }
    main {
      min-width: 0;
      padding: 42px 52px 70px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--paper);
      box-shadow: 0 2px 8px rgb(31 35 40 / 5%);
    }
    h1, h2, h3, h4 { line-height: 1.3; scroll-margin-top: 20px; }
    h1 { margin-top: 0; font-size: 2em; }
    h2 { margin-top: 2.2em; padding-bottom: 0.35em; border-bottom: 1px solid var(--border); }
    h3 { margin-top: 1.7em; }
    a { color: var(--accent); }
    hr { height: 1px; margin: 32px 0; border: 0; background: var(--border); }
    blockquote {
      margin: 1em 0;
      padding: 0.2em 1em;
      color: var(--quote);
      border-left: 4px solid var(--border);
    }
    code {
      padding: 0.15em 0.35em;
      border-radius: 5px;
      background: var(--code-bg);
      font: 0.9em/1.5 "Cascadia Code", Consolas, monospace;
    }
    pre {
      overflow: auto;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--code-bg);
    }
    pre code { padding: 0; background: transparent; }
    table { width: 100%; border-collapse: collapse; margin: 1.25em 0; font-size: 0.94em; }
    th, td { padding: 8px 11px; border: 1px solid var(--border); vertical-align: top; }
    th { background: var(--code-bg); text-align: left; }
    img { max-width: 100%; }
    .rdd-math-block {
      overflow-x: auto;
      overflow-y: hidden;
      padding: 0.6em 0;
    }
    .status { color: var(--muted); }
    @media (max-width: 900px) {
      .layout { display: block; padding: 0; }
      nav { display: none; }
      main { padding: 24px 20px 50px; border: 0; border-radius: 0; }
    }
    @media print {
      body { background: white; }
      .layout { display: block; padding: 0; }
      nav { display: none; }
      main { border: 0; box-shadow: none; padding: 0; }
      a { color: inherit; text-decoration: none; }
    }
  </style>
  <style>${katexCss}</style>
</head>
<body>
  <div class="layout">
    <nav id="toc"><strong>目录</strong></nav>
    <main id="content">${renderedBody}</main>
  </div>
  <script>
    const content = document.getElementById("content");
    const headings = [...content.querySelectorAll("h2, h3")];
    const usedIds = new Set();
    headings.forEach((heading, index) => {
      const base = heading.textContent.trim()
        .toLowerCase()
        .replace(/[^\\p{L}\\p{N}]+/gu, "-")
        .replace(/^-|-$/g, "") || "section-" + (index + 1);
      let id = base;
      let suffix = 2;
      while (usedIds.has(id)) id = base + "-" + suffix++;
      usedIds.add(id);
      heading.id = id;
    });

    const toc = document.getElementById("toc");
    toc.innerHTML = '<strong>目录</strong>' + headings.map((heading) =>
      '<a class="toc-' + heading.tagName.toLowerCase() + '" href="#' +
        heading.id + '">' + heading.textContent + '</a>'
    ).join("");

  </script>
</body>
</html>
`;

await writeFile(outputPath, html, "utf8");
console.log(`Rendered ${inputPath}`);
console.log(`KaTeX   ${formulaCount} formulas (pre-rendered offline)`);
console.log(`Wrote    ${outputPath}`);
