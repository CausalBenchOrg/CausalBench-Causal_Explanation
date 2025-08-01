import os
import smtplib
import time
import asyncio
import common_constants

def start_smtp_connection(email, password):
    global smtp_connection
    start = time.time()
    smtp_connection = smtplib.SMTP('smtp.gmail.com', 587)
    print(f"SMTP connection started: {str(time.time() - start)}")
    smtp_connection.starttls()
    print(f"TLS started: {str(time.time() - start)}")
    smtp_connection.login(common_constants.EMAIL, common_constants.EMAIL_PASSWORD)
    print(f"Logged in: {str(time.time() - start)}")
    print("SMTP connection established.")
    print(smtp_connection)
    return smtp_connection

async def keep_smtp_alive():
    smtp_client = start_smtp_connection(common_constants.EMAIL, common_constants.EMAIL_PASSWORD)
    while True:
        try:
            if smtp_client:
                print("SMTP Connection alive!")
                smtp_client.noop()  # Send NOOP command to keep the connection alive
            await asyncio.sleep(300)  # Wait for 5 minutes before sending the next NOOP
        except Exception as e:
            # Handle exceptions and potentially re-establish the connection
            print(f"Error keeping SMTP alive: {e}")
            await start_smtp_connection(common_constants.EMAIL, common_constants.EMAIL_PASSWORD)


def formulate_verification_mail(verification_id):
    backend_url = common_constants.HOST
    verification_url = f"{backend_url}/verify/{verification_id}"
    subject = "Please verify your account for CausalBench"
    body = f"Hi,\nUse this URL to finish verification of your CausalBench account:\n{verification_url}\n\nThank you,\nCausalBench Team \n\n -- THIS IS AN AUTO GENERATED EMAIL. DO NOT REPLY TO THIS EMAIL! --"
    return subject, body

def formulate_password_reset_mail(token):
    backend_url = common_constants.HOST
    reset_url = f"{backend_url}/forgotPassword/{token}"
    subject = "Password reset request for CausalBench"
    body = f"Hi,\nUse this URL to reset password for your CausalBench account:\n{reset_url}\n\nThank you,\nCausalBench Team"
    return subject, body