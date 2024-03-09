import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body):
    smtp_server = "smtp.gmail.com"
    port = 587  # Gmail SMTP 포트

    sender_email = "hoo217606@gmail.com"
    receiver_email = "tmdgn2002@gmail.com"
    app_password = "yoch spra idlc stki"  # 애플리케이션 비밀번호 사용

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject  # 수정된 부분

    message.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()

# 이메일 보내기
send_email("테스트 제목", "테스트 본문")