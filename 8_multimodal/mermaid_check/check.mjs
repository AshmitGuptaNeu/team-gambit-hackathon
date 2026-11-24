// Reads Mermaid code on stdin, prints {"ok": true, "type": "..."} or {"ok": false, "error": "..."}.
// Mermaid's parser needs a DOM, so jsdom provides one; nothing is rendered and nothing goes online.
import { JSDOM } from "jsdom";
const dom = new JSDOM("<!doctype html><body></body>");
globalThis.window = dom.window; globalThis.document = dom.window.document;
// Mermaid sanitises labels with "<" (e.g. List<Observer>) using browser classes: give it jsdom's.
for (const k of ["Element", "HTMLElement", "Node", "DocumentFragment", "NodeFilter", "DOMParser", "NamedNodeMap",
                 "HTMLFormElement", "HTMLTemplateElement", "Text", "Comment"]) globalThis[k] = dom.window[k];
globalThis.DOMPurify = undefined;
const { default: mermaid } = await import("mermaid");
mermaid.initialize({ startOnLoad: false });
let code = "";
for await (const chunk of process.stdin) code += chunk;
try {
  const r = await mermaid.parse(code);
  console.log(JSON.stringify({ ok: true, type: r.diagramType }));
} catch (e) {
  console.log(JSON.stringify({ ok: false, error: String(e.message || e).slice(0, 600) }));
}
