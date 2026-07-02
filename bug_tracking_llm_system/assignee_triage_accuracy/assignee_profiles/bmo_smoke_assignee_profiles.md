# Assignee Responsibility Profiles: bmo_smoke

> These profiles are inferred from historical ticket ownership only. They are not verified job titles.

- History path: `bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/data/processed/bmo_smoke_history_train.jsonl`
- Source rows: `45`
- Profile count: `15`

## Summary

| Assignee | Tickets | Confidence | Top Components | Top Keywords |
|---|---:|---|---|---|
| `dev_0002` | 9 | medium | dom: core & html (2); settings ui (2); general (2) | chrome, chrome mochikit, mochikit, mochikit content, content |
| `dev_0015` | 9 | medium | css parsing and computation (3); dom: core & html (2); graphics: canvas2d (1) | scope, sync, kscopeactivation, web-platform-tests, dcheck |
| `dev_0011` | 3 | low | search (1); messaging system (1); new tab page (1) | react-test-renderer, newtab, chai-json-schema, able, aboutwelcome |
| `dev_0001` | 2 | low | audio/video: playback (2) | media test, test test_eme_mfcdm_generate_request.html, test_eme_mfcdm_generate_request.html, media, dom media |
| `dev_0003` | 2 | low | audio/video (1); audio/video: web codecs (1) | decodertemplate, ffmpegvideoencoder, smaller, bigger, functions |
| `dev_0004` | 2 | low | pdf viewer (2) | l10n, viewer.ftl, viewer.ftl l10n, l10n viewer.ftl, pdf.js |
| `dev_0005` | 2 | low | panning and zooming (2) | minimap, rects, border, border thickness, thickness |
| `dev_0006` | 2 | low | layout: text and fonts (1); general (1) | xpcomutils.definelazygetter, pref, remove, add linter, chromeutils.definelazygetter |
| `dev_0007` | 2 | low | javascript engine (1); javascript engine: jit (1) | inlining, monomorphic, monomorphic inlining, trial, trial inlining |
| `dev_0008` | 2 | low | graphics: canvas2d (1); graphics: webrender (1) | builds worker, worker, checkouts, checkouts gecko, worker checkouts |
| `dev_0009` | 2 | low | dom: web authentication (2) | aboutwebauthn, aboutwebauthn tests, components aboutwebauthn, browser browser_aboutwebauthn_bio.js, browser_aboutwebauthn_bio.js |
| `dev_0010` | 2 | low | web audio (1); audio/video: web codecs (1) | xsimd, arch, arch generic, generic, generic xsimd_generic_math.hpp |
| `dev_0012` | 2 | low | widget: gtk (2) | boxes, hand, etc, cursor, links |
| `dev_0013` | 2 | low | panning and zooming (2) | core panning, panning, panning zooming, zooming, change |
| `dev_0014` | 2 | low | xpcom (2) | generate, core xpcom, xpidl, xpcom, arguments |

## Details

### `dev_0002`

Inferred from 9 historical tickets: likely focuses on dom: core & html, settings ui, general in firefox, core. Common work themes: chrome, chrome mochikit, mochikit, mochikit content, content.

- Ticket count: `9`
- Confidence: `medium`
- Top products: firefox (6); core (3)
- Top components: dom: core & html (2); settings ui (2); general (2); dom: ui events & focus handling (1); search (1); protections ui (1)
- Top keywords: chrome, chrome mochikit, mochikit, mochikit content, content, content tests, dom events, events
- Representative tickets:
  - `bmo_1872954` [protections ui] [Protections Panel] Protections doorhanger has a blocked tracker milestones subviewbutton that is unlabeled
  - `bmo_1872719` [dom: ui events & focus handling] Perma [Tier-2] dom/events/test/clipboard/browser_navigator_clipboard_read.js | Node is not accessible via accessibility API: id: , tagName: browser, className
  - `bmo_1872899` [search] Perma [tier 2] TEST-UNEXPECTED-FAIL | browser/components/search/test/browser/browser_searchbar_openpopup.js | Node is not accessible via accessibility API: id: , tagName: html:body, className:  -
  - `bmo_1872903` [general] Add an exception from a11y_checks for a click on a web content in Desktop UI
  - `bmo_1872743` [settings ui] Default and Default Private Engine comboboxes in Settings UI are unlabelled

### `dev_0015`

Inferred from 9 historical tickets: likely focuses on css parsing and computation, dom: core & html, graphics: canvas2d in core. Common work themes: scope, sync, kscopeactivation, web-platform-tests, dcheck.

- Ticket count: `9`
- Confidence: `medium`
- Top products: core (9)
- Top components: css parsing and computation (3); dom: core & html (2); graphics: canvas2d (1); layout: images, video, and html frames (1); layout (1); css transitions and animations (1)
- Top keywords: scope, sync, kscopeactivation, web-platform-tests, dcheck, units, rule, bugs.webkit.org
- Representative tickets:
  - `bmo_1872986` [css parsing and computation] [wpt-sync] Sync PR 43849 - [@scope] Produce kScopeActivation relations for scoped nesting
  - `bmo_1872736` [layout: images, video, and html frames] [wpt-sync] Sync PR 43838 - Wait for FCP before running color-mix tests
  - `bmo_1872649` [graphics: canvas2d] [wpt-sync] Sync PR 43835 - Ensure that drawImage(detachedCanvas) throws
  - `bmo_1872758` [layout] [wpt-sync] Sync PR 43840 - Remove ResizeObserver lifecycle DCHECK
  - `bmo_1872908` [dom: core & html] [wpt-sync] Sync PR 43845 - DOM: Introduce `ObservableEventListenerOptions` dictionary

### `dev_0011`

Inferred from 3 historical tickets: likely focuses on search, messaging system, new tab page in firefox. Common work themes: react-test-renderer, newtab, chai-json-schema, able, aboutwelcome.

- Ticket count: `3`
- Confidence: `low`
- Top products: firefox (3)
- Top components: search (1); messaging system (1); new tab page (1)
- Top keywords: react-test-renderer, newtab, chai-json-schema, able, aboutwelcome, aboutwelcome asrouter, appears react-test-renderer, asrouter
- Representative tickets:
  - `bmo_1873013` [new tab page] Remove unused react-test-renderer from newtab's node_modules
  - `bmo_1873010` [messaging system] Remove chai-json-schema module from aboutwelcome/asrouter test code as it is not used
  - `bmo_1872757` [search] Update SearchTestUtils.useTestEngines to be able to load search-config-v2 configurations

### `dev_0001`

Inferred from 2 historical tickets: likely focuses on audio/video: playback in core. Common work themes: media test, test test_eme_mfcdm_generate_request.html, test_eme_mfcdm_generate_request.html, media, dom media.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: audio/video: playback (2)
- Top keywords: media test, test test_eme_mfcdm_generate_request.html, test_eme_mfcdm_generate_request.html, media, dom media, add_task, http, http mochi.test
- Representative tickets:
  - `bmo_1873011` [audio/video: playback] Perma win ccov wmfme dom/media/test/test_eme_mfcdm_generate_request.html | failed to create media key
  - `bmo_1872973` [audio/video: playback] [wmfme] Run EME clearkey tests on wmfme

### `dev_0003`

Inferred from 2 historical tickets: likely focuses on audio/video, audio/video: web codecs in core. Common work themes: decodertemplate, ffmpegvideoencoder, smaller, bigger, functions.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: audio/video (1); audio/video: web codecs (1)
- Top keywords: decodertemplate, ffmpegvideoencoder, smaller, bigger, functions, settings, audio video, core audio
- Representative tickets:
  - `bmo_1872871` [audio/video] Split FFmpegVideoEncoder initailization into smaller functions
  - `bmo_1872951` [audio/video: web codecs] DecoderTemplate cleanup

### `dev_0004`

Inferred from 2 historical tickets: likely focuses on pdf viewer in firefox. Common work themes: l10n, viewer.ftl, viewer.ftl l10n, l10n viewer.ftl, pdf.js.

- Ticket count: `2`
- Confidence: `low`
- Top products: firefox (2)
- Top components: pdf viewer (2)
- Top keywords: l10n, viewer.ftl, viewer.ftl l10n, l10n viewer.ftl, pdf.js, github.com mozilla, mozilla pdf.js, pdf.js commit
- Representative tickets:
  - `bmo_1872797` [pdf viewer] Update PDF.js to new version 231c79800b4e0bb5b705353a98c7e8c8cff70393 from 2023-12-31 14:34:28
  - `bmo_1872721` [pdf viewer] Highlight on cropped pdf isn't visible when saving/printing

### `dev_0005`

Inferred from 2 historical tickets: likely focuses on panning and zooming in core. Common work themes: minimap, rects, border, border thickness, thickness.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: panning and zooming (2)
- Top keywords: minimap, rects, border, border thickness, thickness, apz, apz minimap, thickness rects
- Representative tickets:
  - `bmo_1872901` [panning and zooming] Increase border thickness of rects in APZ minimap
  - `bmo_1872772` [panning and zooming] Remove apz.scrollend-event.content.enabled pref

### `dev_0006`

Inferred from 2 historical tickets: likely focuses on layout: text and fonts, general in core, firefox. Common work themes: xpcomutils.definelazygetter, pref, remove, add linter, chromeutils.definelazygetter.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (1); firefox (1)
- Top components: layout: text and fonts (1); general (1)
- Top keywords: xpcomutils.definelazygetter, pref, remove, add linter, chromeutils.definelazygetter, chromeutils.definelazygetter add, default years, error xpcomutils.definelazygetter
- Representative tickets:
  - `bmo_1872784` [layout: text and fonts] Remove layout.css.font-display.enabled pref
  - `bmo_1872922` [general] Replace last few uses of XPCOMUtils.defineLazyGetter with ChromeUtils.defineLazyGetter and add linter error for XPCOMUtils.defineLazyGetter

### `dev_0007`

Inferred from 2 historical tickets: likely focuses on javascript engine, javascript engine: jit in core. Common work themes: inlining, monomorphic, monomorphic inlining, trial, trial inlining.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: javascript engine (1); javascript engine: jit (1)
- Top keywords: inlining, monomorphic, monomorphic inlining, trial, trial inlining, yaml, flag force, loading
- Representative tickets:
  - `bmo_1872666` [javascript engine] Remove OrderedDict code from YAML file loading
  - `bmo_1873027` [javascript engine: jit] Add a shell flag to force monomorphic inlining or trial inlining

### `dev_0008`

Inferred from 2 historical tickets: likely focuses on graphics: canvas2d, graphics: webrender in core. Common work themes: builds worker, worker, checkouts, checkouts gecko, worker checkouts.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: graphics: canvas2d (1); graphics: webrender (1)
- Top keywords: builds worker, worker, checkouts, checkouts gecko, worker checkouts, builds, gfx, gecko gfx
- Representative tickets:
  - `bmo_1872776` [graphics: webrender] firefox: src/rasterize.h:1157: void draw_perspective_spans(int, Point3D *, Interpolants *, Texture &, Texture &, const ClipRect &) [P = unsigned int]: Assertion `l0.y == r0.y' failed.
  - `bmo_1872646` [graphics: canvas2d] Only use DrawTargetWebgl::BeginFrame when actually mutating a canvas

### `dev_0009`

Inferred from 2 historical tickets: likely focuses on dom: web authentication in core. Common work themes: aboutwebauthn, aboutwebauthn tests, components aboutwebauthn, browser browser_aboutwebauthn_bio.js, browser_aboutwebauthn_bio.js.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: dom: web authentication (2)
- Top keywords: aboutwebauthn, aboutwebauthn tests, components aboutwebauthn, browser browser_aboutwebauthn_bio.js, browser_aboutwebauthn_bio.js, toolkit, toolkit components, test-pass toolkit
- Representative tickets:
  - `bmo_1872957` [dom: web authentication] Perma [tier 2] toolkit/components/aboutwebauthn/tests/browser/browser_aboutwebauthn_bio.js | Node is not accessible via accessibility API: id: bio-enrollments-tab-button, tagName: DIV, className: category -
  - `bmo_1873038` [dom: web authentication] Sidebar navigation tablist in about:webauthn needs to ensure proper dynamic focus handling

### `dev_0010`

Inferred from 2 historical tickets: likely focuses on web audio, audio/video: web codecs in core. Common work themes: xsimd, arch, arch generic, generic, generic xsimd_generic_math.hpp.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: web audio (1); audio/video: web codecs (1)
- Top keywords: xsimd, arch, arch generic, generic, generic xsimd_generic_math.hpp, include xsimd, third_party xsimd, windows
- Representative tickets:
  - `bmo_1872670` [web audio] Update xsimd to new version 3216c13f180e671d61b8bf7ecb96168f78592100 from 2024-01-02 07:57:02
  - `bmo_1872879` [audio/video: web codecs] Set scalability mode in VideoEncoder on Windows

### `dev_0012`

Inferred from 2 historical tickets: likely focuses on widget: gtk in core. Common work themes: boxes, hand, etc, cursor, links.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: widget: gtk (2)
- Top keywords: boxes, hand, etc, cursor, links, open hand, open, boxes etc
- Representative tickets:
  - `bmo_1872961` [widget: gtk] The cursor was shown as an open hand instead of a pointing hand above links, video, audio, boxes etc.
  - `bmo_1872963` [widget: gtk] WakeLockListener.cpp debug log messages are incorrect

### `dev_0013`

Inferred from 2 historical tickets: likely focuses on panning and zooming in core. Common work themes: core panning, panning, panning zooming, zooming, change.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: panning and zooming (2)
- Top keywords: core panning, panning, panning zooming, zooming, change, change webrenderlayerscrolldata, checking leaf, displayportutils
- Representative tickets:
  - `bmo_1872564` [panning and zooming] optimize DisplayPortUtils::MaybeCreateDisplayPortInFirstScrollFrameEncountered by checking for leaf frames
  - `bmo_1872563` [panning and zooming] change WebRenderLayerScrollData::mVisibleRegion from a region to a rect

### `dev_0014`

Inferred from 2 historical tickets: likely focuses on xpcom in core. Common work themes: generate, core xpcom, xpidl, xpcom, arguments.

- Ticket count: `2`
- Confidence: `low`
- Top products: core (2)
- Top components: xpcom (2)
- Top keywords: generate, core xpcom, xpidl, xpcom, arguments, arguments enums, avoid keywords, better separate
- Representative tickets:
  - `bmo_1872918` [xpcom] Generate typescript declarations lib from xpidl
  - `bmo_1872969` [xpcom] Rename several xpidl method arguments and enums to avoid js keywords and name collisions

