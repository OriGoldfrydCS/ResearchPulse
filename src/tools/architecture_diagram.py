"""
Architecture Diagram Generator — Comprehensive, colourful PNG for ResearchPulse.

Uses Pillow to render the full system architecture: all 25 tools organised by
functional category (each with its own colour), all databases, external services,
agent sub-modules, and feature flags.

Added for Course Project compliance: GET /api/model_architecture must return image/png.
"""

from __future__ import annotations

import io
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════
# Canvas
# ═══════════════════════════════════════════════════════════════════
CANVAS_W = 1600
CANVAS_H = 1300

# ═══════════════════════════════════════════════════════════════════
# Global colours
# ═══════════════════════════════════════════════════════════════════
BG      = (248, 249, 252)
WHITE   = (255, 255, 255)
TITLE_C = (33, 37, 41)
SUB_C   = (108, 117, 125)
ARROW_C = (73, 80, 87)
DIVIDER = (206, 212, 218)

# ═══════════════════════════════════════════════════════════════════
# Category palettes – (header_fill, box_fill, border)
# ═══════════════════════════════════════════════════════════════════
C_CORE      = ((21, 101, 192),  (227, 242, 253), (13, 71, 161))     # Blue
C_ANALYSIS  = ((46, 125, 50),   (232, 245, 233), (27, 94, 32))      # Green
C_DELIVERY  = ((230, 81, 0),    (255, 243, 224), (191, 54, 12))     # Orange
C_INBOUND   = ((0, 121, 107),   (224, 242, 241), (0, 77, 64))      # Teal
C_COLLEAGUE = ((106, 27, 154),  (243, 229, 245), (74, 20, 140))    # Purple
C_SCHEDULE  = ((198, 40, 40),   (255, 235, 238), (183, 28, 28))    # Red
C_AGENT     = ((173, 20, 87),   (252, 228, 236), (136, 14, 79))    # Pink
C_API       = ((63, 81, 181),   (232, 234, 246), (48, 63, 159))    # Indigo
C_DB        = ((55, 71, 79),    (236, 239, 241), (38, 50, 56))     # Blue-grey
C_FLAGS     = ((121, 85, 72),   (239, 235, 233), (93, 64, 55))     # Brown


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def _font(size: int):
    """Load a TrueType font or fall back to the built-in bitmap font."""
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                 "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _rrect(d, xy, r, fill, outline, w=2):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=w)


def _arrow(d, x, y0, y1, c=ARROW_C, w=2):
    d.line([(x, y0), (x, y1)], fill=c, width=w)
    d.polygon([(x - 5, y1 - 8), (x + 5, y1 - 8), (x, y1)], fill=c)


def _header_bar(d, x0, y0, x1, y_h, fill, radius=10):
    """Coloured bar at top of a rounded-rect (rounded top, flat bottom)."""
    d.rounded_rectangle((x0 + 1, y0 + 1, x1 - 1, y0 + y_h),
                        radius=radius, fill=fill)
    d.rectangle((x0 + 1, y0 + y_h - 10, x1 - 1, y0 + y_h), fill=fill)


def _tool_box(d, x, y, w, h, name, desc, pal, fn, fd):
    """Single tool box inside a category."""
    _rrect(d, (x, y, x + w, y + h), 6, WHITE, pal[2], 1)
    d.text((x + w // 2, y + 5), name, fill=pal[0], font=fn, anchor="mt")
    d.text((x + w // 2, y + 22), desc, fill=SUB_C, font=fd, anchor="mt")


def _category_box(d, x, y, w, tools, pal, title, fh, fn, fd):
    """Category group with header + vertical tool list.  Returns bottom y."""
    n = len(tools)
    h = 30 + n * 44 + 8
    _rrect(d, (x, y, x + w, y + h), 10, pal[1], pal[2], 2)
    _header_bar(d, x, y, x + w, 28, pal[0], 10)
    d.text((x + w // 2, y + 7), title, fill=WHITE, font=fh, anchor="mt")
    ty = y + 34
    tw = w - 20
    for name, desc in tools:
        _tool_box(d, x + 10, ty, tw, 38, name, desc, pal, fn, fd)
        ty += 44
    return y + h


def _db_cylinder(d, cx, cy, w, h, name, desc, pal, fn, fd):
    """Database cylinder icon."""
    hdr, bg, border = pal
    x0, x1 = cx - w // 2, cx + w // 2
    ey = 14
    d.rectangle((x0, cy + ey // 2, x1, cy + h - ey // 2), fill=bg)
    d.ellipse((x0, cy + h - ey, x1, cy + h), fill=bg, outline=border, width=2)
    d.line([(x0, cy + ey // 2), (x0, cy + h - ey // 2)], fill=border, width=2)
    d.line([(x1, cy + ey // 2), (x1, cy + h - ey // 2)], fill=border, width=2)
    d.ellipse((x0, cy, x1, cy + ey), fill=hdr, outline=border, width=2)
    d.text((cx, cy + h // 2 + 2), name, fill=TITLE_C, font=fn, anchor="mm")
    d.text((cx, cy + h // 2 + 18), desc, fill=SUB_C, font=fd, anchor="mm")


# ═══════════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def generate_architecture_png() -> bytes:
    """Generate the comprehensive ResearchPulse architecture diagram (PNG bytes)."""
    W, H = CANVAS_W, CANVAS_H
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f24, f15, f13, f12, f11, f10 = (
        _font(24), _font(15), _font(13), _font(12), _font(11), _font(10),
    )
    cx = W // 2

    # ─── TITLE ────────────────────────────────────────────────────
    d.text((cx, 16), "ResearchPulse  -  Full System Architecture",
           fill=TITLE_C, font=f24, anchor="mt")
    d.text((cx, 44),
           "Autonomous ReAct Agent for Research Paper Discovery, Delivery & Collaboration",
           fill=SUB_C, font=f13, anchor="mt")

    # ─── WEB UI ───────────────────────────────────────────────────
    uy = 68
    _rrect(d, (cx - 180, uy, cx + 180, uy + 34), 8, C_API[1], C_API[2])
    d.text((cx, uy + 9), "Web UI  (static/index.html)",
           fill=C_API[2], font=f15, anchor="mt")
    _arrow(d, cx, uy + 34, uy + 52)

    # ─── FASTAPI SERVER ───────────────────────────────────────────
    ay = uy + 52
    _rrect(d, (cx - 380, ay, cx + 380, ay + 52), 10, C_API[0], C_API[2], 2)
    d.text((cx, ay + 5), "FastAPI Server",
           fill=WHITE, font=f15, anchor="mt")
    d.text((cx, ay + 22),
           "routes.py  |  dashboard_routes.py  |  colleague_routes.py  |  run_manager.py",
           fill=(200, 210, 255), font=f11, anchor="mt")
    d.text((cx, ay + 37),
           "/api/execute  |  /api/team_info  |  /api/agent_info  |  "
           "/api/model_architecture  |  /api/dashboard/*",
           fill=(180, 190, 240), font=f10, anchor="mt")
    _arrow(d, cx, ay + 52, ay + 70)

    # ─── REACT AGENT ──────────────────────────────────────────────
    ry = ay + 70
    _rrect(d, (cx - 420, ry, cx + 420, ry + 82), 12, C_AGENT[1], C_AGENT[2], 3)
    d.text((cx, ry + 5), "ReAct Agent  (react_agent.py)",
           fill=C_AGENT[0], font=f15, anchor="mt")
    d.text((cx, ry + 23),
           "Thought -> Action -> Observation  |  Bounded Episodic Execution",
           fill=SUB_C, font=f12, anchor="mt")
    subs = [
        ("ScopeGate", C_AGENT), ("StopController", C_AGENT),
        ("PromptController", C_AGENT), ("ProfileEvolution", C_ANALYSIS),
        ("ReplyParser", C_INBOUND),
    ]
    sx = cx - 380
    for name, pal in subs:
        _rrect(d, (sx, ry + 44, sx + 138, ry + 70), 6, pal[1], pal[2], 1)
        d.text((sx + 69, ry + 50), name, fill=pal[0], font=f11, anchor="mt")
        sx += 156
    _arrow(d, cx, ry + 82, ry + 100)

    # ─── TOOL REGISTRY ────────────────────────────────────────────
    tr_y = ry + 100
    _rrect(d, (cx - 480, tr_y, cx + 480, tr_y + 28), 6, ARROW_C,
           (52, 58, 64), 2)
    d.text((cx, tr_y + 6),
           "Tool Registry  --  25 tools across 6 categories",
           fill=WHITE, font=f13, anchor="mt")

    # ═══════════════════════════════════════════════════════════════
    # TOOL CATEGORIES — three rows
    # ═══════════════════════════════════════════════════════════════
    cat_w = 280     # width of each category box
    cat_gap = 20    # gap between categories

    # ── Row 1: Core Pipeline (9 tools in 3 × 3 grid) ─────────────
    t1y = tr_y + 43
    core_w = 3 * cat_w + 2 * cat_gap   # 880
    core_x = cx - core_w // 2
    core_h = 30 + 3 * 44 + 8           # 170
    _rrect(d, (core_x, t1y, core_x + core_w, t1y + core_h), 10,
           C_CORE[1], C_CORE[2], 2)
    _header_bar(d, core_x, t1y, core_x + core_w, 28, C_CORE[0], 10)
    d.text((core_x + core_w // 2, t1y + 7), "Core Pipeline Tools",
           fill=WHITE, font=f13, anchor="mt")
    for off in [-260, 0, 260]:
        _arrow(d, cx + off, tr_y + 28, t1y, C_CORE[2])
    core_tools = [
        ("FetchArxivPapers", "arXiv API query"),
        ("CheckSeenPapers", "Dedup filter"),
        ("RetrieveSimilar", "RAG retrieval"),
        ("ScoreRelevance", "Relevance scoring"),
        ("LLMNovelty", "Deep novelty AI"),
        ("DecideDelivery", "Policy engine"),
        ("PersistState", "State persistence"),
        ("GenerateReport", "Report builder"),
        ("TerminateRun", "Run termination"),
    ]
    col_w = (core_w - 24 - 20) // 3    # ~ 278
    for i, (name, desc) in enumerate(core_tools):
        col, row = i % 3, i // 3
        bx = core_x + 12 + col * (col_w + 10)
        by = t1y + 34 + row * 44
        _tool_box(d, bx, by, col_w, 38, name, desc, C_CORE, f12, f10)

    # ── Row 2: Paper Analysis | Delivery & Email | Inbound ────────
    t2y = t1y + core_h + 15
    r2_sx = cx - (3 * cat_w + 2 * cat_gap) // 2

    pa_x = r2_sx
    _category_box(d, pa_x, t2y, cat_w,
                  [("SummarizePaper", "PDF -> AI summary"),
                   ("LiveDocument", "Research briefing"),
                   ("AuditLog", "Run audit logs")],
                  C_ANALYSIS, "Paper Analysis", f13, f12, f10)

    de_x = r2_sx + cat_w + cat_gap
    de_bot = _category_box(d, de_x, t2y, cat_w,
                  [("OutboundEmail", "SMTP gateway"),
                   ("EmailTemplates", "HTML templates"),
                   ("CalendarInvite", "ICS invitations"),
                   ("ICSGenerator", "RFC 5545 .ics"),
                   ("RecipientResolver", "Recipient logic")],
                  C_DELIVERY, "Delivery & Email", f13, f12, f10)

    ib_x = r2_sx + 2 * (cat_w + cat_gap)
    _category_box(d, ib_x, t2y, cat_w,
                  [("EmailPoller", "IMAP polling"),
                   ("InboundProcessor", "Reply parsing"),
                   ("InboxScheduler", "Poll scheduler")],
                  C_INBOUND, "Inbound & Inbox", f13, f12, f10)

    # ── Row 3: Colleague | Scheduling | Feature Flags ────────────
    t3y = de_bot + 15
    r3_sx = r2_sx

    co_x = r3_sx
    co_bot = _category_box(d, co_x, t3y, cat_w,
                  [("ColleagueParser", "Signup parser"),
                   ("ColleagueTokens", "HMAC tokens"),
                   ("JoinCodeCrypto", "AES encryption")],
                  C_COLLEAGUE, "Colleague Management", f13, f12, f10)

    sc_x = r3_sx + cat_w + cat_gap
    _category_box(d, sc_x, t3y, cat_w,
                  [("SchedulerService", "Background runs"),
                   ("ArxivCategories", "Category taxonomy")],
                  C_SCHEDULE, "Scheduling", f13, f12, f10)

    # Feature Flags box
    ff_x = r3_sx + 2 * (cat_w + cat_gap)
    ff_h = co_bot - t3y          # match Colleague height
    _rrect(d, (ff_x, t3y, ff_x + cat_w, t3y + ff_h), 10,
           C_FLAGS[1], C_FLAGS[2], 2)
    _header_bar(d, ff_x, t3y, ff_x + cat_w, 28, C_FLAGS[0], 10)
    d.text((ff_x + cat_w // 2, t3y + 7), "Feature Flags",
           fill=WHITE, font=f13, anchor="mt")
    flags = ["LLM Novelty Scoring", "Audit Logging",
             "Profile Evolution", "Live Document"]
    fy = t3y + 38
    for fl in flags:
        d.text((ff_x + 20, fy), "* " + fl, fill=C_FLAGS[0], font=f12)
        fy += 24

    # ═══════════════════════════════════════════════════════════════
    # SEPARATOR
    # ═══════════════════════════════════════════════════════════════
    sep_y = co_bot + 20
    d.line([(60, sep_y), (W - 60, sep_y)], fill=DIVIDER, width=2)

    # ═══════════════════════════════════════════════════════════════
    # DATABASES
    # ═══════════════════════════════════════════════════════════════
    db_ly = sep_y + 10
    d.text((cx, db_ly), "Databases & Storage",
           fill=TITLE_C, font=f15, anchor="mt")
    dby = db_ly + 25
    dbs = [
        ("PostgreSQL / Supabase", "Primary Database", C_CORE),
        ("Pinecone Vector DB", "RAG Embeddings", C_ANALYSIS),
        ("JSON File Store", "Dev / Offline Fallback", C_DB),
    ]
    db_w, db_h, db_gap = 250, 65, 70
    db_total = len(dbs) * db_w + (len(dbs) - 1) * db_gap
    db_sx = cx - db_total // 2
    for i, (name, desc, pal) in enumerate(dbs):
        dx = db_sx + i * (db_w + db_gap)
        _db_cylinder(d, dx + db_w // 2, dby, db_w, db_h, name, desc,
                     pal, f12, f10)

    # ═══════════════════════════════════════════════════════════════
    # EXTERNAL SERVICES
    # ═══════════════════════════════════════════════════════════════
    ext_ly = dby + db_h + 20
    d.text((cx, ext_ly), "External Services & APIs",
           fill=TITLE_C, font=f15, anchor="mt")
    exy = ext_ly + 25
    exts = [
        ("arXiv API", "Papers + PDFs", C_CORE),
        ("OpenAI / LLM API", "GPT + Embeddings", C_COLLEAGUE),
        ("SMTP (Gmail)", "Outbound Email", C_DELIVERY),
        ("IMAP (Gmail)", "Inbound Email", C_INBOUND),
    ]
    ex_w, ex_h, ex_gap = 230, 50, 45
    ex_total = len(exts) * ex_w + (len(exts) - 1) * ex_gap
    ex_sx = cx - ex_total // 2
    for i, (name, desc, pal) in enumerate(exts):
        ex = ex_sx + i * (ex_w + ex_gap)
        _rrect(d, (ex, exy, ex + ex_w, exy + ex_h), 8, pal[1], pal[2], 2)
        _header_bar(d, ex, exy, ex + ex_w, 20, pal[0], 8)
        d.text((ex + ex_w // 2, exy + 4), name,
               fill=WHITE, font=f11, anchor="mt")
        d.text((ex + ex_w // 2, exy + 32), desc,
               fill=SUB_C, font=f10, anchor="mt")

    # ═══════════════════════════════════════════════════════════════
    # LEGEND
    # ═══════════════════════════════════════════════════════════════
    leg_y = exy + ex_h + 20
    d.text((80, leg_y), "Legend:", fill=TITLE_C, font=f15)
    legend = [
        ("Core Pipeline", C_CORE),   ("Paper Analysis", C_ANALYSIS),
        ("Delivery & Email", C_DELIVERY), ("Inbound & Inbox", C_INBOUND),
        ("Colleague Mgmt", C_COLLEAGUE),  ("Scheduling", C_SCHEDULE),
        ("Feature Flags", C_FLAGS),       ("Database", C_DB),
    ]
    lx = 160
    for name, pal in legend:
        _rrect(d, (lx, leg_y + 2, lx + 18, leg_y + 18), 4, pal[0], pal[2])
        d.text((lx + 24, leg_y + 2), name, fill=TITLE_C, font=f11)
        lx += 28 + len(name) * 7

    # ─── Encode to PNG ────────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
