---
id: 7ba45890-b892-47b6-b0d3-3cd8086f4dac
name: latex
description: Generate LaTeX documents with preferred academic formatting.
type: reference
created: 2026-01-29T20:53:54
updated: 2026-01-29T20:53:54
tags: [skill, latex, academic]
aliases: []
user_invocable: true
---

# LaTeX Template Generator

Generate a new LaTeX document using the user's preferred academic style.

## IMPORTANT: Always Prompt for Style

When this skill is invoked, ALWAYS use the AskUserQuestion tool to ask the user which style they want BEFORE generating any LaTeX. Present these options:

1. **ACL Style** - For *ACL conference papers (ACL, EMNLP, NAACL, etc.)
2. **General Academic** - For coursework, reports, and general documents

Only proceed with document generation after the user has selected a style.

## ACL Style Template (Conference Papers)

For ACL/EMNLP/NAACL and other *ACL conference submissions, use this template:

```latex
\documentclass[11pt]{article}

% ACL 2023 style (update year as needed)
\usepackage[hyperref]{acl2023}

% Standard packages
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}

% Recommended packages
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{subcaption}

% For final camera-ready, use: \aclfinalcopy

\title{[TITLE]}

\author{
  First Author \\
  Affiliation \\
  \texttt{email@example.com} \\\And
  Second Author \\
  Affiliation \\
  \texttt{email@example.com}
}

\begin{document}
\maketitle

\begin{abstract}
Abstract text here (100-200 words for most *ACL venues).
\end{abstract}

\section{Introduction}

\section{Related Work}

\section{Method}

\section{Experiments}

\subsection{Experimental Setup}

\subsection{Results}

\section{Analysis}

\section{Conclusion}

\section*{Limitations}
% Required for *ACL 2023+ submissions

\section*{Ethics Statement}
% Required for *ACL 2023+ submissions (if applicable)

\section*{Acknowledgments}
% Use \section*{Acknowledgments} for camera-ready

\bibliography{references}

\appendix

\section{Appendix}

\end{document}
```

### ACL Style Files

Download the official ACL style files from: https://github.com/acl-org/acl-style-files

Required files in the same directory:
- `acl2023.sty` (or current year)
- `acl_natbib.bst`

### ACL-Specific Rules

- **Page limit**: 8 pages (long) or 4 pages (short), excluding references
- **Citations**: Use `\citet{}` for textual and `\citep{}` for parenthetical
- **Anonymous**: Remove author names and acknowledgments for review
- **Limitations section**: Required after main content
- **References**: Use `\bibliography{references}` with BibTeX

## General Academic Template

For coursework, reports, and general academic documents:

```latex
\documentclass[11pt]{article}

% Font
\usepackage{palatino}
\usepackage[T1]{fontenc}

% Page layout
\usepackage[top=0.6in, bottom=0.6in, hmargin=0.75in, includehead, includefoot, headheight=14pt, footskip=0.25in]{geometry}
\usepackage{setspace}
\setstretch{1.1}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}

% Header and footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\small [CLASS_NAME]}
\fancyfoot[C]{\small \textsf{\href{https://github.com/[REPO]}{github.com/[REPO]}}}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[C]{\small [CLASS_NAME]}
  \fancyfoot[C]{\small \textsf{\href{https://github.com/[REPO]}{github.com/[REPO]}}}
  \fancyfoot[R]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
}

% Standard packages
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

% Compact section spacing
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1em}{0.5em}
\titlespacing*{\subsection}{0pt}{0.8em}{0.3em}
\titlespacing*{\subsubsection}{0pt}{0.6em}{0.2em}

% Compact list spacing
\usepackage{enumitem}
\setlist[itemize]{nosep, topsep=0.3em, partopsep=0pt, parsep=0pt, itemsep=0.2em}

% Title formatting
\usepackage{titling}
\setlength{\droptitle}{-5em}
\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\end{center}\vspace{-1em}}
\preauthor{\begin{center}}
\postauthor{\end{center}\vspace{-2em}}
\predate{}
\postdate{}

\title{[TITLE]}
\author{Name (\href{mailto:email@example.com}{email@example.com})}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Abstract text here.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}

\section{Conclusion}

\begin{thebibliography}{9}
\bibitem{example} Author Name. Year. Title. \textit{Publication}.
\end{thebibliography}

\end{document}
```

## Common Style Rules

When generating or editing LaTeX:

- **Paragraphs**: No indentation for General Academic style (use `\setlength{\parindent}{0pt}`). ACL style keeps default indentation per conference formatting requirements
- **Tables**: Always use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) with `[H]` or `[t]` positioning
- **Links**: Use `\href{}{}` for clickable links, display GitHub URLs without `https://`
- **Lists**: Keep itemize lists compact
- **Figures**: Use `[H]` or `[t]` float specifier for placement
- **Math**: Use `amsmath` environments (`align`, `equation`)
- **Build**: pdfLaTeX for ACL style; XeLaTeX via latexmk for general

## Output

Write the generated file to `doc/[filename].tex` where filename is derived from the title (lowercase, hyphens for spaces).
