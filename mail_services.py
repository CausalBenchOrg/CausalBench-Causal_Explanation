import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from mail_helper_services import start_smtp_connection
import common_constants

def send_email(to: str, subject: str, body: str, attachments: list = None):
    server = start_smtp_connection(common_constants.EMAIL, common_constants.EMAIL_PASSWORD)

    print(f'To ID {str(to)}')

    message = MIMEMultipart()
    message["From"] = f"CausalBench Admin <{common_constants.SEND_AS_EMAIL}>"
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