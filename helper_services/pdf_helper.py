import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer, HRFlowable
from datetime import datetime, timezone

from common_constants import TEMP_DIR

def generate_pdf(outcome_column, causal_analysis_results, unique_id, run_ids, filters):
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

    # Styles for text
    styles = getSampleStyleSheet()

    # Define styles
    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Title"],
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles["Heading3"],
        alignment=0,
        spaceBefore=0,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        alignment=0
    )
    legend_style = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        alignment=0,
    )

    # Define elements
    spacer = Spacer(1, 25)

    separator = HRFlowable(
        width="100%",
        thickness=1,
        lineCap='round',
        color=colors.grey,
        spaceBefore=20,
        spaceAfter=20
    )

    # Text above the table
    title = Paragraph("CausalBench+: Causal Explanation Report", title_style)
    subtitle = Paragraph(
        f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        subtitle_style
    )
    
    grouped_tables = []

    # Load images from media folder using absolute paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    img_up_path = os.path.join(script_dir, "media", "causal_explanation_pdf_images", "blue_up.png")
    img_down_path = os.path.join(script_dir, "media", "causal_explanation_pdf_images", "blue_down.png")
    img_zero_path = os.path.join(script_dir, "media", "causal_explanation_pdf_images", "zero.png")
    img_up = Image(img_up_path, width=16, height=16)
    img_down = Image(img_down_path, width=16, height=16)
    img_zero = Image(img_zero_path, width=16, height=16)

    # Legend
    legend_data = [
        [img_up, f"This variable increases {outcome_column}"],
        [img_down, f"This variable decreases {outcome_column}"],
        [img_zero, f"This variable has no effect on {outcome_column}"],
    ]

    legend = Table(legend_data, hAlign="LEFT")

    legend_style = [
        # Column alignment
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),

        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), "MIDDLE"),
    ]

    legend.setStyle(TableStyle(legend_style))

    # Process data
    for group, group_data in causal_analysis_results.items():
        table_data = [["Variable", "Effect", "Strength"]]
        grouped_tables.append(Paragraph(f"<b>Summary:</b> {group_data['summary']}", subtitle_style))
        
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
        table = Table(table_data, colWidths="*")

        # Style the table
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            # Column alignment
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),

            # Vertical alignment
            ('VALIGN', (0, 0), (-1, -1), "MIDDLE"),

            # Outer box
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ]

        # Line below each row
        for i in range(len(table_data) - 1):
            table_style.append(("LINEBELOW", (0, i), (-1, i), 0.5, colors.grey))

        table.setStyle(TableStyle(table_style))
        
        grouped_tables.append(table)
        

        if len(group_data["recommendations"]) > 0:
            table_data = [['#'] + group_data['recommend_dims']]

            for i, recommendation in enumerate(group_data["recommendations"]):
                table_data.append((i + 1,) + recommendation)

            recommendations_header = Paragraph(
                "<br /><b>Additional hyperparameter settings to consider for your experiments:</b><br /><br />",
                body_style
            )

            # Create the table
            table = Table(table_data, colWidths="*")

            # Style the table
            table_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                # Column alignment
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),

                # Vertical alignment
                ('VALIGN', (0, 0), (-1, -1), "MIDDLE"),

                # Outer box
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ]

            # Line below each row
            for i in range(len(table_data) - 1):
                table_style.append(("LINEBELOW", (0, i), (-1, i), 0.5, colors.grey))

            table.setStyle(TableStyle(table_style))

            grouped_tables.append(recommendations_header)
            grouped_tables.append(table)
        
        grouped_tables.append(separator)

    run_id_header = Paragraph(
        "<b>Run IDs:</b>",
        subtitle_style
    )
    
    run_ids = sorted(run_ids)
    
    run_ids = [str(run_id) for run_id in run_ids]
    run_id_list = Paragraph(
        f"{', '.join(run_ids)}",
        body_style
    )
    
    filter_header = Paragraph(
        "<b>Filters Applied:</b>",
        subtitle_style
    )
    
    filters_list = filters
    
    filters_list = Paragraph(
        "<br />".join(filters_list),
        body_style
    )

    doc.build([title, subtitle, separator, legend, separator] + grouped_tables + [run_id_header, run_id_list, separator, filter_header, filters_list])

    return pdf_filename
