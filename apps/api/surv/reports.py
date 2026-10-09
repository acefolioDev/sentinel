from datetime import datetime, timezone
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from .config import DATA_DIR


def build_report(path: Path, title: str, notes: str, evidence: list[dict]):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallMuted', parent=styles['BodyText'], fontSize=8, leading=11, textColor=colors.HexColor('#64748b')))
    styles['Title'].textColor = colors.HexColor('#102a43')
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48,
                            title=title, author='Surveillance Q&A')
    story = [Paragraph(_safe(title), styles['Title']), Spacer(1, 8),
             Paragraph('Evidence review report · ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), styles['SmallMuted']),
             Spacer(1, 18), Paragraph('Investigator notes', styles['Heading2']),
             Paragraph(_safe(notes or 'No notes supplied.'), styles['BodyText']), Spacer(1, 16),
             Paragraph('Evidence', styles['Heading2'])]
    for index, item in enumerate(evidence, 1):
        story += [Spacer(1, 12), Paragraph(f'{index}. {_safe(item["camera_name"])} · {_safe(item["location"])}', styles['Heading3'])]
        image_path = DATA_DIR / item['thumbnail_key']
        if image_path.exists():
            source = ImageReader(str(image_path))
            w, h = source.getSize()
            display_w = min(330, w)
            story.append(Image(str(image_path), width=display_w, height=display_w * h / w))
            story.append(Spacer(1, 6))
        data = [
            ['Evidence ID', item['id']], ['Source video', item['video_id']],
            ['Source SHA-256', item['sha256']], ['Source interval', f'{item["start_ms"] / 1000:.3f}–{item["end_ms"] / 1000:.3f} seconds'],
            ['Source UTC', item.get('source_utc') or 'Unknown (video offset only)'],
            ['Review state', item['review_status']], ['Raw similarity', f'{item["similarity_raw"]:.4f} (uncalibrated)'],
            ['Reviewer note', item.get('review_note') or 'None']]
        table = Table([[Paragraph(_safe(str(a)), styles['SmallMuted']), Paragraph(_safe(str(b)), styles['BodyText'])] for a,b in data],
                      colWidths=[105, 390])
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),
                                   ('LINEBELOW',(0,-1),(-1,-1),0.4,colors.HexColor('#dce4ea'))]))
        story.append(table)
    story += [Spacer(1, 18), Paragraph('Interpretation limits', styles['Heading2']),
              Paragraph('Similarity scores rank possible visual matches only. Automated descriptions and unreviewed candidates are provisional. A verified review state records an investigator decision and does not establish identity on its own.', styles['BodyText'])]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(48, 30, 'SURVEILLANCE Q&A  ·  EVIDENCE REPORT')
    canvas.drawRightString(A4[0]-48, 30, f'Page {doc.page}')
    canvas.restoreState()

def _safe(value):
    from xml.sax.saxutils import escape
    return escape(value).replace('\n', '<br/>')
