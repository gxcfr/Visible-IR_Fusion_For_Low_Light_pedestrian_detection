import os
import sys
import copy
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
INPUT_DOCX = PROJECT_ROOT / "downloads_input" / "LLVIP_Research_Report.docx"
OUT_DIR = PROJECT_ROOT / "research" / "final_report_delivery"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DOCX = OUT_DIR / "LLVIP_Research_Report_Final.docx"
FIG_DIR = OUT_DIR / "figures"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement("w:tblBorders")
    # Top border: 1.5 pt
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), "12")
    top.set(qn("w:space"), "0")
    top.set(qn("w:color"), "222222")
    tblBorders.append(top)
    # Bottom border: 1.5 pt
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "222222")
    tblBorders.append(bottom)
    # Inside horizontal: 0.5 pt
    insideH = OxmlElement("w:insideH")
    insideH.set(qn("w:val"), "single")
    insideH.set(qn("w:sz"), "4")
    insideH.set(qn("w:space"), "0")
    insideH.set(qn("w:color"), "D0D0D0")
    tblBorders.append(insideH)
    # None for verticals
    for b_name in ["left", "right", "insideV"]:
        b = OxmlElement(f"w:{b_name}")
        b.set(qn("w:val"), "none")
        tblBorders.append(b)
    tblPr.append(tblBorders)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m_name, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m_name}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)

def create_table_element(doc, headers, data, col_widths, alignments, title_text, notes_text):
    """Creates a beautifully styled academic table with title and notes paragraphs."""
    # Title paragraph
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run(title_text)
    run_title.bold = True
    run_title.font.size = Pt(10)
    run_title.font.name = "Arial"

    # Table
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    # Header Row
    hdr_row = table.rows[0]
    hdr_tr = hdr_row._tr.get_or_add_trPr()
    hdr_tr.append(OxmlElement("w:tblHeader")) # repeat header across pages

    for c_idx, h_text in enumerate(headers):
        cell = hdr_row.cells[c_idx]
        cell.width = col_widths[c_idx]
        set_cell_background(cell, "F2F4F7")
        set_cell_margins(cell, top=120, bottom=120, left=120, right=120)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = alignments[c_idx]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.name = "Arial"
        run.font.color.rgb = RGBColor(30, 41, 59)

    # Data Rows
    for r_idx, row_values in enumerate(data):
        row = table.rows[r_idx + 1]
        is_highlight = "E007" in str(row_values[0]) or "Proposed" in str(row_values[0]) or "IA001" in str(row_values[0])
        bg_col = "F8FAFC" if (r_idx % 2 == 1 and not is_highlight) else ("EFF6FF" if is_highlight else "FFFFFF")

        for c_idx, val in enumerate(row_values):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = alignments[c_idx]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.size = Pt(8.5)
            run.font.name = "Arial"
            if is_highlight or (c_idx == 0 and "E00" in str(val)):
                run.bold = True
            if is_highlight:
                run.font.color.rgb = RGBColor(15, 23, 42)

    # Notes paragraph
    p_notes = doc.add_paragraph()
    p_notes.paragraph_format.space_before = Pt(4)
    p_notes.paragraph_format.space_after = Pt(12)
    run_notes = p_notes.add_run(notes_text)
    run_notes.italic = True
    run_notes.font.size = Pt(8.0)
    run_notes.font.name = "Arial"
    run_notes.font.color.rgb = RGBColor(100, 116, 139)

    return p_title, table, p_notes

def create_figure_elements(doc, img_path, title_text, caption_text, width_in=6.0):
    """Creates a centered figure paragraph and a caption paragraph."""
    # Image Paragraph
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(12)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_in))

    # Caption Paragraph
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(14)
    p_cap.paragraph_format.line_spacing = 1.15

    run_t = p_cap.add_run(title_text + " ")
    run_t.bold = True
    run_t.font.size = Pt(9.0)
    run_t.font.name = "Arial"
    run_t.font.color.rgb = RGBColor(30, 41, 59)

    run_c = p_cap.add_run(caption_text)
    run_c.font.size = Pt(8.5)
    run_c.font.name = "Arial"
    run_c.font.color.rgb = RGBColor(71, 85, 105)

    return p_img, p_cap

def main():
    print("Loading source document:", INPUT_DOCX)
    doc = docx.Document(str(INPUT_DOCX))

    print(f"Loaded document: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables.")

    # 1. Identify Table 1 (Internal Instructions Callout) and mark for deletion
    callout_table = None
    for t in doc.tables:
        if len(t.rows) > 0 and len(t.columns) > 0:
            if "How to finalize this report" in t.rows[0].cells[0].text:
                callout_table = t
                break

    if callout_table:
        print("Removing Table 1 ('How to finalize this report' internal handoff callout)...")
        callout_table._element.getparent().remove(callout_table._element)

    # 2. Map of Table Callouts to Replacements
    # We will locate tables by the marker in their first cell text
    table_replacements = {
        # Section VI Figures
        "[[FIGURE_VI_1_WORKFLOW]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vi_1_workflow.png",
            "title": "Figure VI-1. End-to-end research workflow.",
            "caption": "Standardized single-pass multimodal pedestrian detection research pipeline across the LLVIP benchmark. The workflow illustrates paired dataset audit, scene-aware train/validation/test split, LetterBox preprocessing, 7-model zoo implementation, unified native-coordinate evaluation, diagnostic interventions (IA001–IA004), 25-condition partial robustness stress-testing, cryptographic model freeze, and held-out official test set evaluation.",
            "width": 6.2
        },
        "[[FIGURE_VI_2_PAIRED_DATASET_SAMPLES]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vi_2_paired_dataset_samples.png",
            "title": "Figure VI-2. Paired RGB and thermal samples with ground-truth annotations.",
            "caption": "Authentic validation image pairs from LLVIP. Top row: Pair 020001 (Scene 02: high thermal contrast under deep visible darkness). Bottom row: Pair 050001 (Scene 05: streetlight illumination with distant pedestrians). Green bounding boxes denote verified ground-truth pedestrian annotations in native 1280×1024 coordinate space.",
            "width": 6.1
        },
        "[[FIGURE_VI_3_EXPERIMENT_ARCHITECTURES]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vi_3_experiment_architectures.png",
            "title": "Figure VI-3. Implemented experiment families (E001–E007).",
            "caption": "Comparative topological overview of the six architectural paradigms evaluated in this study: (a) Unimodal RGB (E001 YOLO11n / E002 YOLO26n); (b) Unimodal Thermal (E003 YOLO26n); (c) Late Fusion WBF (E004 YOLO26n); (d) Early Fusion 4CH (E005 YOLO26n with 4-channel input stem); (e) Feature Concatenation (E006 YOLO26n with symmetric channel concat); and (f) Proposed Asymmetric Gated Fusion (E007 YOLO26n with dynamic reliability gating and unmodulated thermal residual).",
            "width": 6.2
        },
        "[[FIGURE_VI_4_E007_ARCHITECTURE]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vi_4_e007_architecture.png",
            "title": "Figure VI-4. E007 thermal-primary asymmetric gated fusion architecture.",
            "caption": "Detailed architecture of the proposed E007 model derived directly from source code (asymmetric_gated_fusion_yolo26.py). Thermal features serve as the primary semantic anchor and pass through an unmodulated additive identity shortcut (F_T). Visible features are projected via 1×1 convolution and modulated by a learned spatial reliability gate map (G_V in (0, 1)) computed from concatenated multi-scale representations at P3, P4, and P5 before entering the shared PANet neck and head.",
            "width": 6.2
        },
        "[[FIGURE_VI_5_ROBUSTNESS_EXAMPLES]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vi_5_robustness_examples.png",
            "title": "Figure VI-5. Examples of selected synthetic corruption levels.",
            "caption": "Authentic sample visualizations across the five completed synthetic degradation suites (25 completed conditions): RGB Brightness reduction, RGB Gaussian Noise, RGB Gaussian Blur, Thermal Contrast compression, and Thermal Striping fixed-pattern noise at Severity Levels 1 (mild), 3 (moderate), and 5 (extreme). Dead-pixel evaluation was halted incomplete and joint-modality suites were deferred.",
            "width": 5.9
        },

        # Section VII Tables
        "[[TABLE_VII_1_COMPUTATIONAL_METRICS]]": {
            "type": "table",
            "title": "Table VII-1. Computational performance and hardware profiling (Stage 6)",
            "headers": ["Model ID", "Architecture", "Modality", "Params (M)", "FLOPs (G)", "Model Fwd (ms)", "Pipeline Lat (ms)", "Pipeline FPS", "Peak VRAM (MB)"],
            "col_widths": [Inches(0.65), Inches(1.1), Inches(0.85), Inches(0.65), Inches(0.65), Inches(0.85), Inches(0.85), Inches(0.7), Inches(0.9)],
            "alignments": [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
            "data": [
                ["E001", "YOLO11n", "RGB-only", "2.59", "6.50", "7.59 ± 0.40", "9.97 ± 0.54", "100.3", "144 / 529"],
                ["E002", "YOLO26n", "RGB-only", "2.50", "5.89", "10.76 ± 0.81", "13.03 ± 0.93", "76.7", "280 / 527"],
                ["E003", "YOLO26n", "Thermal-only", "2.50", "5.89", "9.83 ± 0.46", "12.31 ± 0.62", "81.3", "250 / 528"],
                ["E004", "YOLO26n", "Late Fusion (WBF)", "5.01", "11.78", "20.38 ± 0.65", "23.89 ± 0.83", "41.9", "113 / 398"],
                ["E005", "YOLO26n", "Early Fusion (4CH)", "2.57", "6.27", "10.16 ± 0.46", "14.41 ± 0.69", "69.4", "463 / 4,618"],
                ["E006", "YOLO26n", "Feature Concat", "4.14", "9.88", "13.39 ± 0.58", "17.48 ± 0.77", "57.2", "4,600 / 4,600"],
                ["E007", "Proposed", "Asym Gated Fusion", "4.97", "12.16", "14.17 ± 0.75", "17.94 ± 0.89", "55.7", "2,866 / 3,311"],
                ["E008*", "YOLO26n", "1-Class Gated (Def.)", "4.90", "12.04", "14.05 ± 0.72", "17.80 ± 0.85", "56.2", "98 / 442"]
            ],
            "notes": "Note: Measured on an NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) across 200 synchronized evaluation iterations (batch size B=1; peak VRAM also reported for B=16). E008* denotes the theoretical single-class pruned model specification (nc=1 vs historical nc=80 head in E007)."
        },
        "[[TABLE_VII_2_GATE_INTERVENTION_RESULTS]]": {
            "type": "table",
            "title": "Table VII-2. Inference-time gate interventions on E007 (Stage 3)",
            "headers": ["Condition ID", "Gate Mode Description", "mAP50 (%)", "mAP50-95 (%)", "Recall@0.25 (%)", "Prec@0.25 (%)", "F1@0.25 (%)", "FP / Image", "Misses (FN)"],
            "col_widths": [Inches(0.8), Inches(1.5), Inches(0.7), Inches(0.75), Inches(0.75), Inches(0.75), Inches(0.7), Inches(0.65), Inches(0.65)],
            "alignments": [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
            "data": [
                ["IA001", "Learned Gate (Original E007)", "93.70", "55.98", "90.25", "87.52", "88.86", "0.366", "496"],
                ["IA002", "Gate Zero (G_V = 0.0)", "72.27", "38.45", "59.49", "85.31", "70.10", "0.333", "2,060"],
                ["IA003", "Gate Half (G_V = 0.5)", "84.24", "48.12", "80.32", "75.17", "77.66", "0.753", "1,001"],
                ["IA004", "Gate One (G_V = 1.0)", "78.03", "40.37", "78.31", "60.37", "68.18", "1.455", "1,103"]
            ],
            "notes": "Note: Evaluated on the 1,803 validation pairs (5,086 ground truth pedestrians) in native 1280×1024 coordinate space. These conditions represent post-training inference-only interventions on the frozen E007 checkpoint, not independently trained fixed-gate models."
        },
        "[[TABLE_VII_3_ROBUSTNESS_SUMMARY]]": {
            "type": "table",
            "title": "Table VII-3. Selected synthetic-corruption results across 25 completed conditions (Stage 4)",
            "headers": ["Model ID", "Clean mAP50", "RGB Brightness", "RGB Noise", "RGB Blur", "Thermal Contrast", "Thermal Striping", "5-Suite Mean"],
            "col_widths": [Inches(1.1), Inches(0.75), Inches(0.85), Inches(0.8), Inches(0.75), Inches(0.95), Inches(0.95), Inches(0.85)],
            "alignments": [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
            "data": [
                ["E002 (RGB)", "73.04%", "86.74%", "47.46%", "75.98%", "100.00%*", "100.01%*", "82.04%"],
                ["E003 (Thermal)", "94.12%", "100.00%*", "100.00%*", "100.00%*", "74.00%", "38.10%", "82.42%"],
                ["E004 (Late WBF)", "93.74%", "99.92%", "99.70%", "98.88%", "92.02%", "80.30%", "94.16%"],
                ["E005 (Early 4CH)", "72.24%", "101.90%", "99.24%", "96.12%", "51.84%", "51.21%", "80.06%"],
                ["E006 (Concat)", "93.24%", "99.20%", "99.30%", "99.14%", "75.46%", "73.99%", "89.42%"],
                ["E007 (Proposed)", "93.70%", "98.52%", "98.72%", "99.18%", "73.66%", "83.47%", "90.71%"]
            ],
            "notes": "Note: Selected-corruption robustness study: exactly 25 of 40 planned conditions completed (5 complete suites × 5 severity levels: R001, R002, R003, R004, R006). Thermal dead-pixel evaluation (R005) was halted incomplete; simultaneous darkness+washout (R007) and noise+striping (R008) were deferred. Asterisks (*) denote clean performance preserved under orthogonal-modality corruption. Evaluated on the 1,803 validation pairs in native 1280×1024 coordinate space."
        },

        # Section VII Figures
        "[[FIGURE_VII_1_RGB_THERMAL_PAIR]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_1_rgb_thermal_pair.png",
            "title": "Figure VII-1. LLVIP visible and thermal input pair.",
            "caption": "Validation Pair 250001 (Scene 25) in visible and thermal modalities. Scene 25 presents challenging vehicular headlight glare, deep shadows, and low visible SNR. Green bounding boxes mark ground-truth pedestrian locations in native 1280×1024 coordinate space.",
            "width": 6.1
        },
        "[[FIGURE_VII_2_DATASET_AUDIT]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_2_dataset_audit.png",
            "title": "Figure VII-2. Dataset split and annotation audit.",
            "caption": "LLVIP dataset partition distribution and annotation audit. The dataset consists of 15,488 verified pairs (100% paired) partitioned by geographic scene to prevent spatial-temporal leakage: Training (15 scenes, 10,222 pairs, 29,044 valid GTs), Validation (4 scenes: 02, 05, 07, 25; 1,803 pairs, 5,086 valid GTs), and Official Test (7 scenes: 19, 20, 21, 22, 23, 24, 26; 3,463 pairs, 8,302 valid GTs).",
            "width": 6.1
        },
        "[[FIGURE_VII_3_TRAINING_CURVES]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_3_training_curves.png",
            "title": "Figure VII-3. Training convergence across key architectures.",
            "caption": "Authentic training and validation loss and metric convergence curves for E003 (Thermal YOLO26n, best epoch 48 / stopping epoch 58), E006 (Feature Concat YOLO26n), and E007 (Proposed Asymmetric Gated YOLO26n). Evaluated at 640×640 input resolution with early stopping.",
            "width": 5.8
        },
        "[[FIGURE_VII_4_QUALITATIVE_DETECTION_COMPARISON]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_4_qualitative_detection_comparison.png",
            "title": "Figure VII-4. Qualitative detection comparison on validation Scene 02.",
            "caption": "Ground-truth-matched 8-panel detection comparison on validation Pair 020001 (Scene 02: deep visible darkness). Panels display: (1) Raw Visible, (2) Raw Thermal, (3) Ground Truth, (4) E002 RGB (misses pedestrians due to darkness), (5) E003 Thermal (high confidence detection), (6) E004 Late Fusion WBF, (7) E006 Feature Concat, and (8) E007 Proposed Asymmetric Gated Fusion. Confidences and bounding boxes shown in native 1280×1024 coordinate space.",
            "width": 6.1
        },
        "[[FIGURE_VII_5_GATE_ANALYSIS]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_5_gate_analysis.png",
            "title": "Figure VII-5. Gate activation and scene-conditioned analysis.",
            "caption": "Corrected gate dynamics across validation scenes 02, 05, 07, and 25. Panel A: Scene-conditioned scatter showing within-scene positive correlation with ambient brightness (+0.09 to +0.49). Panel B: Within-scene centered residuals resolving Simpson's paradox (r = +0.116). Panel C: Foreground vs background gate distributions, demonstrating selective attenuation inside pedestrian heat signatures (G_V ~ 0.505 vs 0.513). Panel D: Moving block bootstrap distribution accounting for temporal frame autocorrelation (SE = 0.075).",
            "width": 6.0
        },
        "[[FIGURE_VII_6_GATE_INTERVENTION]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_6_gate_intervention.png",
            "title": "Figure VII-6. Inference-only gate intervention comparison (IA001–IA004).",
            "caption": "Detection metrics under inference-only gate clamping on the frozen E007 checkpoint: Learned Gate (IA001, 93.70% mAP50), Gate Zero (IA002, 72.27% mAP50), Gate Half (IA003, 84.24% mAP50), and Gate One (IA004, 78.03% mAP50). Demonstrates that G_V=0 causes representation starvation due to network co-adaptation, while G_V=1 quadruples false alarms.",
            "width": 6.1
        },
        "[[FIGURE_VII_7_ROBUSTNESS_RESULTS]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_7_robustness_results.png",
            "title": "Figure VII-7. Selected synthetic-corruption robustness results.",
            "caption": "Metric retention curves (mAP50 / Clean mAP50) across the 25 completed corruption conditions across 5 degradation suites: RGB Brightness, RGB Gaussian Noise, RGB Gaussian Blur, Thermal Contrast compression, and Thermal Striping fixed-pattern noise. E007 retains 51.17% mAP50 (54.6% retention) under severe thermal striping (L5 sigma=80), outperforming feature concatenation E006 (30.88% mAP50, 33.1% retention) by +20.29% absolute mAP50.",
            "width": 6.2
        },
        "[[FIGURE_VII_8_LATENCY_COMPARISON]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_8_latency_comparison.png",
            "title": "Figure VII-8. Computational performance and hardware profiling.",
            "caption": "Measured latency, throughput (FPS), parameter count, and FLOPs on an NVIDIA GeForce RTX 4060 Laptop GPU. Left: Model forward vs full pipeline latency (all models comfortably exceed the 30 FPS real-time threshold; E007 runs at 55.7 FPS end-to-end). Right: Parameter count (M) vs GFLOPs at 640×640 input resolution.",
            "width": 6.1
        },
        "[[FIGURE_VII_9_SOFTWARE_EXECUTION]]": {
            "type": "figure",
            "img": FIG_DIR / "figure_vii_9_software_execution.png",
            "title": "Figure VII-9. Software execution evidence and audit ledger.",
            "caption": "Authentic project execution artifacts: Panel A: PyTorch implementation of ReliabilityGatingModule and AsymmetricGatedFusionYOLO26. Panel B: Terminal execution logs of the official Stage 8 held-out test evaluation. Panel C: Project directory hierarchy and checkpoint registry. Panel D: Cryptographic SHA256 model freeze ledger and sealed test protocol attestation.",
            "width": 6.1
        }
    }

    # Iterate through all tables in the document and perform replacements
    replaced_count = 0
    # We find all matching tables first
    matched_tables = []
    for t_idx, t in enumerate(doc.tables):
        if len(t.rows) > 0 and len(t.columns) > 0:
            c_text = t.rows[0].cells[0].text.strip()
            for marker_key, rep_info in table_replacements.items():
                if marker_key in c_text:
                    matched_tables.append((t, marker_key, rep_info))
                    break

    print(f"Found {len(matched_tables)} marker callout tables to replace.")

    for t, marker_key, rep_info in matched_tables:
        if rep_info["type"] == "figure":
            p_img, p_cap = create_figure_elements(doc, rep_info["img"], rep_info["title"], rep_info["caption"], rep_info["width"])
            # Insert before t._element
            t._element.addprevious(p_img._element)
            t._element.addprevious(p_cap._element)
            t._element.getparent().remove(t._element)
            replaced_count += 1
            print(f"  [+] Replaced {marker_key} with Figure")
        elif rep_info["type"] == "table":
            p_title, tbl, p_notes = create_table_element(doc, rep_info["headers"], rep_info["data"], rep_info["col_widths"], rep_info["alignments"], rep_info["title"], rep_info["notes"])
            # Insert before t._element
            t._element.addprevious(p_title._element)
            t._element.addprevious(tbl._element)
            t._element.addprevious(p_notes._element)
            t._element.getparent().remove(t._element)
            replaced_count += 1
            print(f"  [+] Replaced {marker_key} with Table")

    print(f"Completed {replaced_count} of 17 marker replacements.")

    # 3. Post-replacement cleanup: scan all paragraphs and table cells for any remaining [[...]] markers or Antigravity instructions
    cleaned_markers = 0
    for p in doc.paragraphs:
        if "[[" in p.text:
            print("WARNING: Found marker in paragraph:", p.text[:60])
            for k in table_replacements.keys():
                p.text = p.text.replace(k, "")
            cleaned_markers += 1

    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                if "[[" in cell.text:
                    print("WARNING: Found marker in table cell:", cell.text[:60])
                    for k in table_replacements.keys():
                        cell.text = cell.text.replace(k, "")
                    cleaned_markers += 1

    # 4. Save Final DOCX
    print("Saving finalized document to:", OUT_DOCX)
    doc.save(str(OUT_DOCX))

    # Also copy to root project directory and downloads_input
    import shutil
    shutil.copy2(OUT_DOCX, PROJECT_ROOT / "LLVIP_Research_Report_Final.docx")
    shutil.copy2(OUT_DOCX, PROJECT_ROOT / "downloads_input" / "LLVIP_Research_Report_Final.docx")
    print("Mirrored final DOCX to root and downloads_input.")

if __name__ == "__main__":
    main()
