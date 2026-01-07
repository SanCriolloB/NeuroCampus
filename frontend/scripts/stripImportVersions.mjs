// frontend/scripts/stripImportVersions.mjs
import fs from "node:fs/promises";
import path from "node:path";

const FRONTEND_ROOT = process.cwd();
const SRC_DIR = path.join(FRONTEND_ROOT, "src");

const exts = new Set([".ts", ".tsx", ".js", ".jsx"]);

function looksLikeSemver(s) {
  // 1.2.3, 1.2, 1, 1.2.3-beta.1, 1.2.3+meta, etc.
  return /^\d+(\.\d+){0,2}([\-+].*)?$/.test(s);
}

function stripVersion(specifier) {
  // Ignorar rutas internas/relativas/alias
  if (
    specifier.startsWith(".") ||
    specifier.startsWith("/") ||
    specifier.startsWith("@/") ||
    specifier.startsWith("http:") ||
    specifier.startsWith("https:")
  ) {
    return specifier;
  }

  const lastAt = specifier.lastIndexOf("@");
  if (lastAt <= 0) return specifier;

  const versionPart = specifier.slice(lastAt + 1);
  if (!looksLikeSemver(versionPart)) return specifier;

  return specifier.slice(0, lastAt);
}

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) files.push(...(await walk(p)));
    else files.push(p);
  }
  return files;
}

async function processFile(filePath) {
  const ext = path.extname(filePath);
  if (!exts.has(ext)) return false;

  let s = await fs.readFile(filePath, "utf8");
  const original = s;

  // import ... from "x"
  s = s.replace(
    /from\s+(['"])([^'"]+)\1/g,
    (m, q, spec) => `from ${q}${stripVersion(spec)}${q}`
  );

  // dynamic import("x")
  s = s.replace(
    /import\(\s*(['"])([^'"]+)\1\s*\)/g,
    (m, q, spec) => `import(${q}${stripVersion(spec)}${q})`
  );

  if (s !== original) {
    await fs.writeFile(filePath, s, "utf8");
    return true;
  }
  return false;
}

async function main() {
  const targets = [
    path.join(SRC_DIR, "components", "ui"),
    path.join(SRC_DIR, "components"),
  ];

  let changed = 0;

  for (const t of targets) {
    try {
      const files = await walk(t);
      for (const f of files) {
        const did = await processFile(f);
        if (did) changed++;
      }
    } catch {
      // carpeta no existe (ok)
    }
  }

  console.log(`stripImportVersions: archivos modificados = ${changed}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
