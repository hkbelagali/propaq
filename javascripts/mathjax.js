window.MathJax = {
  tex: {
    // arithmatex (generic mode) hands us \(...\) on every .md-derived page.
    // Notebook pages never reach arithmatex: mkdocs-jupyter renders markdown
    // cells with nbconvert, which normalises all math to $...$ / $$...$$ and
    // emits a MathJax *2* tex2jax config that MathJax 3 ignores. So both
    // delimiter styles have to be live here.
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Material's documented recipe is ignoreHtmlClass: ".*|" - ignore
    // everything, opt back in via processHtmlClass. That works for .md pages
    // only because arithmatex wraps each expression in a span carrying the
    // class, i.e. the class sits on the math's immediate parent. It cannot
    // work for notebooks, where math is loose $...$ inside a <p>: MathJax
    // re-evaluates the ignore state at every level,
    //     ignore = (ignore || ignoreHtmlClass.exec(cname)) && !process
    // and ".*|" matches the empty class of that <p>, re-ignoring it even
    // though its jp-RenderedMarkdown ancestor was opted in.
    //
    // So ignore a real list instead. <pre>/<code> are already skipped by
    // MathJax's default skipHtmlTags, which covers rendered code cells; the
    // entries below add Material's chrome, the notebook prompts, and the
    // CodeMirror/clipboard mirrors that hold code as plain divs.
    ignoreHtmlClass:
      "md-header|md-footer|md-nav|md-search|md-sidebar|md-source|" +
      "clipboard-copy-txt|CodeMirror|jp-InputArea-editor|" +
      "jp-InputPrompt|jp-OutputPrompt|highlight",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown",
  },
};

// Re-typeset after Material's instant navigation swaps the page body.
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.startup.output.clearCache();
    window.MathJax.typesetClear();
    window.MathJax.texReset();
    window.MathJax.typesetPromise();
  }
});
