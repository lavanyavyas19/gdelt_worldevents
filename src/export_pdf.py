
from __future__ import annotations

import io
import datetime
from typing import Dict, List, Optional


try:
    from fpdf import FPDF as _FPDF_CLASS
    _HAS_FPDF = True
except ImportError:
    _FPDF_CLASS = None
    _HAS_FPDF = False


_BasePDF = _FPDF_CLASS if _HAS_FPDF else object



_NAVY      = (26,  45,  82)    # #1A2D52
_TEAL      = (13, 148, 136)    # #0D9488
_CORAL     = (192, 80,  77)    # #C0504D
_LIGHT     = (240, 244, 248)   # #F0F4F8
_DARKGRAY  = (55,  65,  81)    # #374151
_MIDGRAY   = (107, 114, 128)   # #6B7280
_WHITE     = (255, 255, 255)


def is_available() -> bool:
    """Return True if fpdf2 is installed."""
    return _HAS_FPDF




def _safe_text(text: str, max_len: int = 2000) -> str:
    """Strip non-Latin characters and truncate for PDF safety."""
    safe = "".join(c if ord(c) < 256 else "?" for c in text)
    return safe[:max_len]


class _BriefingPDF(_BasePDF):
    """Custom FPDF subclass for the analyst briefing layout."""

    def __init__(self, title: str, country: str, date_str: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.add_page()
        self._brief_title   = title
        self._brief_country = country
        self._brief_date    = date_str

  
    def header(self):
       
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 210, 28, "F")

        
        self.set_y(5)
        self.set_x(10)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*_WHITE)
        self.cell(0, 7, _safe_text(self._brief_title, 80), ln=True)

       
        self.set_x(10)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(180, 200, 220)
        self.cell(0, 5, f"{self._brief_country}   |   {self._brief_date}", ln=True)

       
        self.set_draw_color(*_TEAL)
        self.set_line_width(0.8)
        self.line(10, 29, 200, 29)
        self.ln(6)

   
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_MIDGRAY)
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        self.cell(0, 5,
                  f"GDELT Event Intelligence Dashboard  |  Generated {ts}  |  "
                  "Data: GDELT Project (gdeltproject.org)  |  For research use only",
                  align="C")

   
    def section_heading(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_TEAL)
        self.cell(0, 5, text.upper(), ln=True)
        
        x, y = self.get_x(), self.get_y()
        self.set_draw_color(*_TEAL)
        self.set_line_width(0.4)
        self.line(10, y, 200, y)
        self.ln(3)

    
    def metrics_table(self, metrics: Dict):
        self.section_heading("Key Metrics")
        self.set_font("Helvetica", "", 9)
        col_w = 95
        items = list(metrics.items())
        
        if len(items) % 2:
            items.append(("", ""))

        for i in range(0, len(items), 2):
            for j in range(2):
                k, v = items[i + j]
                x_pos = 10 + j * col_w
                self.set_xy(x_pos, self.get_y())

                
                if i % 4 == 0:
                    self.set_fill_color(*_LIGHT)
                else:
                    self.set_fill_color(*_WHITE)

                self.set_draw_color(203, 213, 225)
                self.rect(x_pos, self.get_y(), col_w, 7, "FD")

                self.set_text_color(*_MIDGRAY)
                self.set_font("Helvetica", "", 8)
                self.set_xy(x_pos + 2, self.get_y() + 1)
                self.cell(col_w // 2, 4, _safe_text(str(k), 30))

                self.set_text_color(*_DARKGRAY)
                self.set_font("Helvetica", "B", 8)
                self.set_xy(x_pos + col_w // 2, self.get_y())
                self.cell(col_w // 2, 4, _safe_text(str(v), 30))

            self.ln(7)
        self.ln(2)

    
    def summary_section(self, text: str):
        self.section_heading("Analyst Summary")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_DARKGRAY)
        self.set_x(10)
       
        lines_est = len(text) // 90 + 1
        box_h = min(lines_est * 5 + 6, 60)
        self.set_fill_color(*_LIGHT)
        self.rect(10, self.get_y(), 190, box_h, "F")
        self.set_xy(12, self.get_y() + 2)
        self.multi_cell(186, 5, _safe_text(text, 800))
        self.ln(3)

   
    def keywords_section(self, keywords: List[str]):
        if not keywords:
            return
        self.section_heading("Top Keywords")
        self.set_x(10)
        for kw in keywords[:12]:
            kw_text = _safe_text(kw, 25)
            # Chip-style
            w = self.get_string_width(kw_text) + 6
            if self.get_x() + w > 195:
                self.ln(7)
                self.set_x(10)
            self.set_fill_color(*_TEAL)
            self.set_text_color(*_WHITE)
            self.set_font("Helvetica", "", 8)
            self.cell(w, 6, kw_text, border=0, fill=True, align="C")
            self.set_x(self.get_x() + 2)
        self.ln(8)

    
    def comparison_section(self, text: str):
        if not text:
            return
        self.section_heading("Comparative Analysis")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_DARKGRAY)
        self.set_x(10)
        self.multi_cell(190, 5, _safe_text(text, 600))
        self.ln(2)

    
    def evidence_section(self, urls: List[str]):
        if not urls:
            return
        self.section_heading("Evidence Sources")
        self.set_font("Helvetica", "", 8)
        for i, url in enumerate(urls[:6], 1):
            self.set_x(10)
            self.set_text_color(*_MIDGRAY)
            self.cell(6, 5, f"{i}.")
            self.set_text_color(37, 99, 235)   # link blue
            self.cell(0, 5, _safe_text(url, 100), ln=True)
        self.ln(2)



def generate_briefing_pdf(
    spike_data: Dict,
    summary: str,
    keywords: Optional[List[str]] = None,
    evidence_urls: Optional[List[str]] = None,
    comparison_note: str = "",
) -> bytes:
    """
    Generate a professional analyst briefing PDF.

    Parameters
    ----------
    spike_data      : Dict with keys:
                        country (str), date_str (str),
                        event_count (int), z_score (float),
                        baseline (float), avg_tone (float),
                        conflict_pct (float)
    summary         : Analyst summary text (from summarizer.py)
    keywords        : Top TF-IDF keywords
    evidence_urls   : List of source article URLs
    comparison_note : Comparative analysis text

    Returns
    -------
    bytes  — raw PDF content, ready for st.download_button()

    Raises
    ------
    ImportError if fpdf2 is not installed.
    """
    if not _HAS_FPDF:
        raise ImportError(
            "fpdf2 not installed. Run: pip install fpdf2"
        )

    country   = spike_data.get("country", "Unknown")
    date_str  = spike_data.get("date_str", "Unknown Date")
    count     = spike_data.get("event_count", 0)
    z_score   = spike_data.get("z_score", 0.0)
    baseline  = spike_data.get("baseline", 0.0)
    avg_tone  = spike_data.get("avg_tone", 0.0)
    conf_pct  = spike_data.get("conflict_pct", 0.0)

    title = f"Geopolitical Intelligence Briefing — {country}"

    pdf = _BriefingPDF(title=title, country=country, date_str=date_str)

    pdf.metrics_table({
        "Country"         : country,
        "Burst Date"      : date_str,
        "Event Count"     : f"{count:,}",
        "Baseline (7d avg)": f"{baseline:.0f}",
        "Z-Score"         : f"{z_score:.2f} σ",
        "Avg Tone"        : f"{avg_tone:.2f}",
        "Conflict Events" : f"{conf_pct:.1f}%",
        "Severity"        : "High" if z_score >= 3 else "Moderate",
    })

  
    pdf.summary_section(summary)

  
    pdf.keywords_section(keywords or [])


    pdf.comparison_section(comparison_note)

 
    pdf.evidence_section(evidence_urls or [])

    return bytes(pdf.output())
