import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from datetime import datetime

def send_defect_alert(defect_type, timestamp, image_path, smtp_config=None):
    """
    Send email alert with defect image.
    Default: Gmail SMTP. Update smtp_config for your provider.
    smtp_config = {
        'server': 'smtp.gmail.com',
        'port': 587,
        'user': 'pongobi17@gmail.com',
        'password': 'pgsmari*17'
    }
    """
    default_config = {
        'server': 'smtp.gmail.com',
        'port': 587,
        'user': 'pongobi17@gmail.com',  # UPDATE WITH YOUR GMAIL
        'password': 'pgsmari*17'  # GENERATE APP PASSWORD: https://myaccount.google.com/apppasswords
    }
    # print("NOTE: Update email_alert.py with real Gmail + App Password")  # Disabled spam
    config = smtp_config or default_config
    
    msg = MIMEMultipart()
    msg['From'] = config['user']
    msg['To'] = config['user']  # UPDATE recipient
    msg['Subject'] = f'3D Print Defect Detected: {defect_type}'
    
    body = f"""
    Defect Alert!
    Type: {defect_type}
    Timestamp: {timestamp}
    Check the attached image.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data)
        image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(image)
    
    try:
        server = smtplib.SMTP(config['server'], config['port'])
        server.starttls()
        server.login(config['user'], config['password'])
        text = msg.as_string()
        server.sendmail(config['user'], config['user'], text)  # UPDATE recipient
        server.quit()
        print(f"Email sent for {defect_type}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

