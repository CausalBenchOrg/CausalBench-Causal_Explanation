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