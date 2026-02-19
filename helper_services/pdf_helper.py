import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer, HRFlowable
from datetime import datetime, timezone

from common.common_constants import TEMP_DIR


def generate_pdf(outcome_column, causal_analysis_results, unique_id, run_ids, filters):
    # Directory of the script
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create a PDF document
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{TEMP_DIR}/causal_explanation_report_{timestamp}-{unique_id}.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    # Colors
    tab_h_bg_col = colors.HexColor("#95979d")    # table header background color
    tab_h_fg_col = colors.white                    # table header foreground color   
    accent_col = colors.HexColor("#00ccff")      # accent color

    # Styles for text
    styles = getSampleStyleSheet()

    # Define styles
    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Title"],
        alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles["Heading3"],
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=10
    )
    header_style = ParagraphStyle(
        name="Header",
        parent=styles["Heading3"],
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=10,
        textColor=accent_col
    )
    body_style = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=10
    )
    legend_style = ParagraphStyle(
        name="Legend",
        parent=styles["BodyText"],
        alignment=TA_LEFT,
    )
    right_style = ParagraphStyle(
        name="BodyRight",
        parent=styles["BodyText"],
        alignment=TA_RIGHT
    )

    # Define elements
    spacer = Spacer(1, 10)

    separator = HRFlowable(
        width="100%",
        thickness=1,
        lineCap='butt',
        color=accent_col,
        spaceBefore=20,
        spaceAfter=20
    )

    # Load images using absolute paths
    img_logo_path = os.path.join(script_dir, "images", "logo", "logo.png")
    img_up_path = os.path.join(script_dir, "images", "causal_effect", "blue_up.png")
    img_down_path = os.path.join(script_dir, "images", "causal_effect", "blue_down.png")
    img_zero_path = os.path.join(script_dir, "images", "causal_effect", "zero.png")

    img_logo = Image(img_logo_path, width=52, height=52)
    img_up = Image(img_up_path, width=16, height=16)
    img_down = Image(img_down_path, width=16, height=16)
    img_zero = Image(img_zero_path, width=16, height=16)

    # Heading
    title = Paragraph("CausalBench+: Causal Explanation Report", title_style)
    subtitle = Paragraph(
        f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        subtitle_style
    )

    heading = Table([[[title, subtitle], img_logo]], colWidths=["*", None], hAlign="LEFT", splitByRow=0)

    table_style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]

    heading.setStyle(TableStyle(table_style))

    # Legend
    legend_data = [
        [img_up, f"This variable increases {outcome_column}"],
        [img_down, f"This variable decreases {outcome_column}"],
        [img_zero, f"This variable has no effect on {outcome_column}"],
    ]

    legend = Table(legend_data, hAlign="LEFT", splitByRow=0)

    legend_style = [
        # Column alignment
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),

        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), "MIDDLE"),
    ]

    legend.setStyle(TableStyle(legend_style))

    # Process data
    grouped_tables = []

    for group, group_data in causal_analysis_results.items():
        table_data = [["Variable", "Effect", "Strength"]]
        grouped_tables.append(Paragraph(f"<b>Analysis:</b> {group_data['summary']}", header_style))
        
        for k, v in group_data["effects"].items():
            if abs(v) > 1000 or (abs(v) < 0.01 and v != 0):
                        value_str = f"{v:.4e}"
            else:
                value_str = f"{v:.4f}"
                
            if v > 0:
                table_data.append([k, img_up, value_str])
            elif v < 0:
                table_data.append([k, img_down, value_str])
            else:
                table_data.append([k, img_zero, value_str])
        
        # Create the table
        table = Table(table_data, colWidths="*", hAlign="LEFT", repeatRows=1)

        # Style the table
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), tab_h_bg_col),
            ("TEXTCOLOR", (0, 0), (-1, 0), tab_h_fg_col),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            # Column alignment
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),

            # Vertical alignment
            ('VALIGN', (0, 0), (-1, -1), "MIDDLE"),

            # Outer box
            ("BOX", (0, 0), (-1, -1), 0.5, tab_h_bg_col),
        ]

        # Line below each row
        for i in range(len(table_data) - 1):
            table_style.append(("LINEBELOW", (0, i), (-1, i), 0.5, tab_h_bg_col))

        table.setStyle(TableStyle(table_style))
        
        grouped_tables.append(table)
        grouped_tables.append(spacer)

        if len(group_data["recommendations"]) > 0:
            grouped_tables.append(spacer)
            grouped_tables.append(Paragraph("Additional hyperparameter settings to consider for your experiments:", body_style))
                        
            table_data = [['Hyperparameters'] + group_data['recommend_dims']]

            for i, recommendation in enumerate(group_data["recommendations"]):
                table_data.append((f"#{i + 1}",) + recommendation)

            table_data = list(map(list, zip(*table_data)))

            for row in table_data[1:]:
                for j in range(1, len(row)):
                    row[j] = Paragraph(f"{row[j]}", right_style)

            # Create the table
            col_widths = [None] + ["*" for i in range(len(group_data['recommend_dims']))]
            table = Table(table_data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)

            # Style the table
            table_style = [
                ("BACKGROUND", (0, 0), (-1, 0), tab_h_bg_col),
                ("TEXTCOLOR", (0, 0), (-1, 0), tab_h_fg_col),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                # Column alignment
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),

                # Vertical alignment
                ('VALIGN', (0, 0), (-1, -1), "MIDDLE"),

                # Outer box
                ("BOX", (0, 0), (-1, -1), 0.5, tab_h_bg_col),
            ]

            for col in range(len(table_data[0]) - 1):
                table_style.append(("LINEAFTER", (col, 0), (col, -1), 0.5, tab_h_bg_col))

            table.setStyle(TableStyle(table_style))

            grouped_tables.append(table)
            grouped_tables.append(spacer)
        
        grouped_tables.append(separator)

    run_id_header = Paragraph("<b>Run IDs:</b>", header_style)
    
    run_ids = sorted(run_ids)
    
    run_ids = [str(run_id) for run_id in run_ids]
    run_id_list = Paragraph(f"{', '.join(run_ids)}", body_style)
    
    filter_header = Paragraph("<b>Filters Applied:</b>", header_style)

    table_data = []

    for k, v in filters.items():
        # if len(v) > 0:
        table_data.append([k, Paragraph(", ".join(v))])

    # Create the table
    filters_table = Table(table_data, colWidths=[None, "*"], hAlign="LEFT", splitByRow=0)

    # Style the table
    table_style = [
        ("BACKGROUND", (0, 0), (0, -1), tab_h_bg_col),
        ("TEXTCOLOR", (0, 0), (0, -1), tab_h_fg_col),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),

        # Column alignment
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),

        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), "TOP"),

        # Outer box
        ("BOX", (0, 0), (-1, -1), 0.5, tab_h_bg_col),
    ]

    # Line below each row
    for i in range(len(table_data) - 1):
        table_style.append(("LINEBELOW", (0, i), (-1, i), 0.5, tab_h_bg_col))

    filters_table.setStyle(TableStyle(table_style))

    # Build the PDF
    doc.build([heading, separator, legend, separator] + grouped_tables + [run_id_header, run_id_list, separator, filter_header, filters_table])

    return pdf_filename
