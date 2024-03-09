import smtplib, ssl
import datetime

def send_email(subject, body):
    smtp_server = "smtp.gmail.com"
    port = 587  # Gmail SMTP 포트

    sender_email = "hoo217606@gmail.com"
    receiver_email = "tmdgn2002@gmail.com"
    password = "tlsrh112!"

    message = f"Subject: {subject}\n\n{body}"

    context = ssl.create_default_context()

    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls(context=context)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


# 현재 시간 가져오기
now = datetime.datetime.now()

# 이메일로 보낼 내용 설정
subject = "일일 보고"
body = f"현재 시간: {now}"

# 이메일 보내기
send_email(subject, body)
