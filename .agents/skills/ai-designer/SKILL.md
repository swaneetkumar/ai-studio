---
name: ai-designer
description: "AI Designer Skill suite from elayadesign/ai-design-skills. Complete system for building high-converting landing pages, structured UI design, strict visual systems (typography, spacing scale, 60-30-10 color rules, corner radius, animations), and professional Figma design workflows. Activate when designing or building any UI, marketing page, web app interface, or Figma mockup."
---

# AI Designer Skill (elayadesign/ai-design-skills)

This skill suite unifies the design intelligence from **elayadesign/ai-design-skills**:

1. **`landing-page-design`**: Complete system for building high-converting landing pages, structured page architecture, conversion copy, and rigid visual systems (type scale, spacing tokens, corner radius, backgrounds, icons, and motion).
2. **`figma-designer`**: Design inside Figma with real variables, text styles, auto-layout, and reusable component tokens.

---

## 1. Landing Page Design System (Non-Negotiable Visual Rules)

### Core Strategy: One Offer → One Audience → One Primary Action
- **Hero Section:** One dominant headline, subtext, high-contrast CTA button, and single primary visual (product mock, video demo, or interactive preview).
- **Intake Requirements:**
  - What is the ONE primary action? (trial, demo, buy, waitlist, download)
  - What counts as a conversion?
  - Top 3 objections to resolve.

### Visual Foundation Tokens:
- **60-30-10 Color Rule:** 60% dominant background, 30% structural/text cards, 10% high-contrast accent for primary CTAs.
- **Typography:** Max 2 font families (e.g., Inter + JetBrains Mono or Poppins + Open Sans). Major Third or Perfect Fourth scale.
- **Spacing Scale:** Multiples of 4px/8px (e.g., 4, 8, 12, 16, 24, 32, 48, 64, 96px). No arbitrary margins.
- **Corner Radius:** Standardized hierarchy (`rounded-xl` for cards, `rounded-lg` for inputs, `rounded-full` for badges/pills).
- **Motion:** Subtle transitions (150ms-250ms), smooth micro-interactions, full `@media (prefers-reduced-motion: reduce)` accessibility.

---

## 2. Figma Workflow
- Load local variable collections before creating loose colors.
- Use real Tailwind spacing/radius tokens.
- Apply auto-layout to all frames.
- Bind all fills and strokes to semantic tokens.

---

## Sub-Skills Reference
- Full Landing Page Specification: [`../landing-page-design/SKILL.md`](../landing-page-design/SKILL.md)
- Figma Designer Specification: [`../figma-designer/SKILL.md`](../figma-designer/SKILL.md)
