import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import common.common_constants as common_constants


def start_smtp_connection(email, password):
    global smtp_connection
    start = time.time()
    smtp_connection = smtplib.SMTP('smtp.gmail.com', 587)
    print(f"SMTP connection started: {str(time.time() - start)}")
    smtp_connection.starttls()
    print(f"TLS started: {str(time.time() - start)}")
    smtp_connection.login(email, password)
    print(f"Logged in: {str(time.time() - start)}")
    print("SMTP connection established.")
    print(smtp_connection)
    return smtp_connection


def send_email(to: str, subject: str, body: str, attachments: list = None):
    server = start_smtp_connection(common_constants.EMAIL, common_constants.EMAIL_PASSWORD)

    print(f'To ID {str(to)}')

    message = MIMEMultipart()
    message["From"] = f"CausalBench Admin <{common_constants.EMAIL}>"
    message["To"] = to
    message["Subject"] = subject
    message.add_header('reply-to', common_constants.REPLY_TO_ADDRESS)
    message.attach(MIMEText(body, "plain"))
    
    # attach the attachments if provided
    if attachments:
        for attachment in attachments:
            try:
                with open(attachment, "rb") as file:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(file.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment)}")
                    message.attach(part)
            except Exception as e:
                print(f"Failed to attach file {attachment}. Error: {e}")
                return {"status": f"Failed to attach file {attachment}. Error: {e}"}

    # Create SMTP session for sending the mail
    try:
        text = message.as_string()
        server.sendmail(common_constants.EMAIL, to, text)
        print("Email Sent Successfully.")
        return {"status": "Email sent successfully."}
    except Exception as e:
        print("Email Failed to Send. Error: ", e)
        return {"status": f"Failed to send email. Error: {e}"}
