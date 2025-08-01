import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
import uuid
from datetime import datetime

def generate_pdf(yaml_data, causal_recommendation_results, causal_recommendation_vars, unique_id, run_ids, filters):
    # Create a PDF document
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"/tmp/causal_anallysis_report_{timestamp}-{unique_id}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)

    # Styles for text
    styles = getSampleStyleSheet()

    # Define left-aligned styles
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
        spaceAfter=20
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

    # Text above the table
    title = Paragraph("CausalBench: Causal Analysis Report", title_style)
    subtitle = Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), subtitle_style)
    effect = Paragraph(f"<b>Summary:</b> {yaml_data["causal_effects"]["summary"]}", body_style)
    spacer = Spacer(1, 25)
    
    sub_headings = []
    grouped_tables = []

    # Load images from media folder using absolute paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    img_up_path = os.path.join(script_dir, "media", "causal_analysis_pdf_images", "up.png")
    img_down_path = os.path.join(script_dir, "media", "causal_analysis_pdf_images", "down.png")
    img_zero_path = os.path.join(script_dir, "media", "causal_analysis_pdf_images", "zero.png")
    img_up = Image(img_up_path, width=16, height=16)
    img_down = Image(img_down_path, width=16, height=16)
    img_zero = Image(img_zero_path, width=16, height=16)

    if not yaml_data['grouped']:
        raw_table_data = yaml_data["causal_effects"]["effects"]
        table_data = [["Variable", "Effect", "Strength"]]
        
        for k, v in raw_table_data.items():
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
    else:
        raw_table_data = yaml_data["causal_effects"]["overall_effects"]
        table_data = [["Variable", "Effect", "Strength"]]
        grouped_tables.append(Paragraph(f"<b>Summary:</b> {yaml_data["causal_effects"]["summary"]}", subtitle_style))
        
        for k, v in raw_table_data.items():
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
        grouped_tables.append(spacer)
        
        for key in yaml_data["causal_effects"]["group_specific_effects"]:
            raw_table_data = yaml_data["causal_effects"]["group_specific_effects"][key]
            table_data = [["Variable", "Effect", "Strength"]]
            grouped_tables.append(Paragraph(f"<b>Summary:</b> {key}", subtitle_style))
            
            for k, v in raw_table_data.items():
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
            grouped_tables.append(spacer)
    # Legend
    legend_data = [
        [img_up, f"This variable improves {yaml_data["causal_effects"]["summary"].split()[2]}"],
        [img_down, f"This variable worsens {yaml_data["causal_effects"]["summary"].split()[2]}"],
        [img_zero, f"This variable has no effect on {yaml_data["causal_effects"]["summary"].split()[2]}"],
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
    
    recommendations = Paragraph(
        f"<b>Recommendations:</b> Additional Hyperparameter settings to consider for your experiments: <br /> "
        f"[{','.join(map(str, causal_recommendation_vars))}]: [{','.join(map(str, causal_recommendation_results[0]))}]",
        body_style
    )
    run_id_header = Paragraph(
        "<b>Run IDs:</b>",
        subtitle_style
    )
    
    run_ids = sorted(run_ids)
    
    run_ids = [str(run_id) for run_id in run_ids]
    run_id_list = Paragraph(
        f"<b>Run IDs:</b> {', '.join(run_ids)}",
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
    # Build PDF
    if not yaml_data['grouped']:
        doc.build([title, subtitle, effect, spacer, table, spacer, legend, spacer, recommendations, spacer, run_id_header, run_id_list, spacer, filter_header, filters_list])
    else:
        doc.build([title, subtitle] + [i for i in grouped_tables] + [legend, spacer, recommendations, spacer, run_id_header, run_id_list, spacer, filter_header, filters_list])

    return pdf_filename